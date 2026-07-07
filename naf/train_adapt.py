import os
import os.path as osp
import pdb
import nibabel as nib
import random
import torch
import copy
import imageio.v2 as iio
import numpy as np
import argparse
import sys
import torch.nn.functional as F
from src.config.configloading import load_config
from src.render import render, run_network #, render_utils
from src.trainer_adapt import Trainer
from src.loss import calc_mse_loss, calc_mse_loss_volume, calc_mse_loss_no_add, calc_tv_loss_GPT, calc_tv_loss, calc_second_derivative_loss, calc_l1_loss_no_add
from src.utils import get_psnr, get_mse, get_psnr_3d, get_ssim_3d, cast_to_image
from src.network import get_lipschitz_constants
from src.render import matcher


def visualize(img):

    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default= "./config/jaw_50.yaml",
                        help="configs file path")
    parser.add_argument("--num", type=int, required=True)
    return parser

parser = config_parser()
args = parser.parse_args()

cfg = load_config(args.config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class BasicTrainer(Trainer):
    def __init__(self):
        """
        Basic network trainer.
        """
        num = sys.argv[2]
        super().__init__(cfg, num, device)
        print(f"[Start] exp: {cfg['exp']['expname']}, net: Basic network")


    def get_random_voxel_patch(self, geo, patch_size=128):

        n1, n2, n3 = geo.nVoxel
        n3 = 200
        s1, s2, s3 = geo.sVoxel / 2 - geo.dVoxel / 2

        # ---- Choose random start indices ----
        max_i = n1 - patch_size
        max_j = n2 - patch_size
        max_k = n3 - patch_size

        i0 = np.random.randint(0, max_i + 1)
        j0 = np.random.randint(0, max_j + 1)
        k0 = np.random.randint(0, max_k + 1)

        # ---- Compute actual coordinate ranges for the patch ----
        xs = np.linspace(-s1, s1, n1)[i0: i0 + patch_size]
        ys = np.linspace(-s2, s2, n2)[j0: j0 + patch_size]
        zs = np.linspace(-s3, s3, n3)[k0: k0 + patch_size]

        # ---- Create meshgrid only for the patch ----
        xyz = np.meshgrid(xs, ys, zs, indexing="ij")
        patch = np.asarray(xyz).transpose([1, 2, 3, 0])
        patch = torch.tensor(patch, dtype=torch.float32, device=device)

        return patch, (i0, j0, k0)


    def adapt(self, data, img_init, global_step, idx_epoch, inner_steps_adapt, meta_batch_size, use_siren):

        params = None

        for inrs in range(inner_steps_adapt):

            self.optimizer.zero_grad()
            voxels = self.train_dset.voxels
            rays = data["rays"][:,inrs,:,:].reshape(-1, 8)
            projs = data["projs"][:,inrs,:].reshape(-1)
            adapt_state = True
            ret = render(idx_epoch,use_siren, adapt_state, rays, voxels, img_init, self.net, self.net_fine, params, self.encoder, self.conf, **self.conf["render"])
            projs_pred = ret["acc"]
            raw = ret["raw_tv"]

            loss_tv = calc_tv_loss_GPT(raw, k=1e-5)

            loss = calc_mse_loss_no_add(projs, projs_pred)

            loss["loss_mse"] = loss["loss"]
            loss["loss_tv_in"] = loss_tv["loss_tv"]


            loss["loss"] = loss["loss_mse"] + loss["loss_tv_in"]
            loss["loss"].backward()


            self.optimizer.step()
            self.lr_scheduler.step()

            if (global_step % 50 == 0):
                self.writer.add_scalar(f"inner_loss_total", loss["loss"], global_step+inrs)
                self.writer.add_scalar(f"inner_loss_mse", loss["loss_mse"], global_step+inrs)
                self.writer.add_scalar(f"inner_loss_tv", loss["loss_tv_in"], global_step + inrs)


        return self.net


    def eval_step(self, img_init, global_step, idx_epoch, adapt_epoch, use_siren):
        """
        Evaluation step
        """
        # Evaluate projection
        select_ind = np.random.choice(len(self.eval_dset))
        projs = self.eval_dset.projs[select_ind]
        rays = self.eval_dset.rays[select_ind].reshape(-1, 8)
        H, W = projs.shape
        projs_pred = []
        voxels = self.train_dset.voxels

        #params = {name: p.clone() for name, p in self.net.named_parameters()}
        params = None
        adapt_state = True
        for i in range(0, rays.shape[0], self.n_rays):
            projs_pred.append(render(idx_epoch, use_siren, adapt_state, rays[i:i+self.n_rays], voxels, img_init, self.net, self.net_fine, params, self.encoder, self.conf, **self.conf["render"])["acc"])

        projs_pred = torch.cat(projs_pred, 0).reshape(H, W) #H, W

        # Evaluate density
        pts = self.eval_dset.voxels
        ##======================== NAF+ ========================##
        #rhos = self.img_init.unsqueeze(-1)        # exactly in the pts (mode=nearest)
        #pts = torch.cat((pts, rhos), dim=-1)
        ##======================== NAF+ ========================##
        image_pred = run_network(idx_epoch,pts, self.net_fine if self.net_fine is not None else self.net, self.netchunk)
        image_pred = image_pred.squeeze()
        loss = {
            "proj_mse": get_mse(projs_pred, projs),
            "proj_psnr": get_psnr(projs_pred, projs),
            #"psnr_3d": get_psnr_3d(image_pred, image),
            #"ssim_3d": get_ssim_3d(image_pred, image),
        }

        img1 = visualize(projs_pred)
        img2 = visualize(projs)
        img_concat = torch.stack([img1, img2], dim=0)
        img_concat = img_concat.unsqueeze(1)
        self.writer.add_images(tag="prediction_vs_gt",img_tensor=img_concat,global_step=self.global_step, dataformats="NCHW")

        show_proj = torch.concat([projs, projs_pred], dim=1)


        for ls in loss.keys():
            self.writer.add_scalar(f"eval/{ls}", loss[ls], global_step)
            
        # Save
        if adapt_epoch == True:
            eval_save_dir = osp.join(self.evaldir, f"adapt_epoch_{idx_epoch:05d}")
        else:
            eval_save_dir = osp.join(self.evaldir, f"step_{idx_epoch:05d}")
        os.makedirs(eval_save_dir, exist_ok=True)
        new_image = nib.Nifti1Image(image_pred.cpu().detach().numpy(), affine=np.eye(4))
        nib.save(new_image, osp.join(eval_save_dir, "image_pred.nii.gz"))
        iio.imwrite(osp.join(eval_save_dir, "proj_show_left_gt_right_pred.png"), (cast_to_image(show_proj)*255).astype(np.uint8))
        with open(osp.join(eval_save_dir, "stats.txt"), "w") as f: 
            for key, value in loss.items(): 
                f.write("%s: %f\n" % (key, value.item()))

        return loss



    def save_outpainted_projections(self, img_init, idx_epoch, train_dset, adapted_model, adapt_epoch, use_siren):

        params = None
        list_projs_out = []
        adapt_state = True
        voxels = self.train_dset.voxels

        for j in range(0, len(train_dset)):

            rays = train_dset.get_outpaint_rays(train_dset.geo_train, j).reshape(-1, 8)
            projs_pred = []
            new_rays_shape_1 = self.eval_dset.projs[0].shape[0]
            new_rays_shape_2 = self.eval_dset.projs[0].shape[1]
            for i in range(0, rays.shape[0], self.n_rays):
                projs_pred.append(
                    render(idx_epoch,use_siren, adapt_state, rays[i:i + self.n_rays], voxels, img_init, adapted_model, self.net_fine, params, self.encoder, self.conf, **self.conf["render"])["acc"])

            projs_pred = torch.cat(projs_pred, 0).reshape(new_rays_shape_1, new_rays_shape_2)  # H, W
            list_projs_out.append(projs_pred)
            print(j)


        for j in range(0,len(self.eval_dset)):
            rays = self.eval_dset.get_outpaint_rays(self.eval_dset.geo_val, j)[j].reshape(-1, 8)
            projs_pred = []
            new_rays_shape_1 = self.eval_dset.projs[0].shape[0]
            new_rays_shape_2 = self.eval_dset.projs[0].shape[1]
            for i in range(0, rays.shape[0], self.n_rays):
                projs_pred.append(
                    render(idx_epoch,use_siren, adapt_state, rays[i:i + self.n_rays], voxels, img_init, adapted_model, self.net_fine, params, self.encoder, self.conf, **self.conf["render"])["acc"])

            projs_pred = torch.cat(projs_pred, 0).reshape(new_rays_shape_1, new_rays_shape_2)  # H, W
            list_projs_out.append(projs_pred)

        if adapt_epoch == True:
            eval_save_dir = osp.join(self.evaldir, f"adapt_epoch_{idx_epoch:05d}")
        else:
            eval_save_dir = osp.join(self.evaldir, f"epoch_{idx_epoch:05d}")


        os.makedirs(eval_save_dir, exist_ok=True)
        list_cpu = [t.cpu().detach().numpy() for t in list_projs_out]
        outpainted_projs= np.array(list_cpu)
        new_image = nib.Nifti1Image(outpainted_projs, affine=np.eye(4))
        nib.save(new_image, osp.join(eval_save_dir, "outpainted_projections.nii.gz"))



trainer = BasicTrainer()
trainer.start()
        
