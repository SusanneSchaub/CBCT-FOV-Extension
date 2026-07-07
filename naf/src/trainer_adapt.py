import os
import os.path as osp
import json
import pdb
import time
#import muon
#print(dir(muon))

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from shutil import copyfile
#from muon import MuonWithAuxAdam
#from muon import SingleDeviceMuonWithAuxAdam

import numpy as np
import random
import nibabel as nib
from .dataset_not_multibatch import TIGREDataset as Dataset
from .encoder import get_encoder


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()



class Trainer:
    def __init__(self, cfg, num, device="cuda"):

        # Args
        self.device = device
        self.global_step = 0
        self.conf = cfg
        self.outpaint = cfg["outpainting"]["outpaint"]
        self.steps = cfg["train"]["steps"]
        self.n_fine = cfg["render"]["n_fine"]
        self.epochs_train = cfg["train"]["epochs_train"]
        self.epochs_adapt = cfg["train"]["epochs_adapt"]
        self.i_eval = cfg["log"]["i_eval"]
        self.i_eval_adapt = cfg["log"]["i_eval_adapt"]
        self.i_save = cfg["log"]["i_save"]
        self.netchunk = cfg["render"]["netchunk"]
        self.n_rays = cfg["train"]["n_rays"]
        self.lr_adapt = cfg["train"]["lr_adapt"]
        self.lr_adapt_muon = cfg["train"]["lr_adapt_muon"]
        self.lr_adapt_adam_muon = cfg["train"]["lr_adapt_adam_muon"]
        self.inner_steps = cfg["train"]["inner_steps_adapt"]
        self.batch_size_adapt = cfg["train"]["n_batch_adapt"]
        self.use_siren = cfg["network"]["use_siren"]
        # Log direcotry
        self.expdir = osp.join(cfg["exp"]["expdir"], cfg["exp"]["expname"])

        self.num = num


        self.ckptdir = osp.join("", "ckpt.tar")
        self.ckptdir_backup = osp.join(self.expdir, "ckpt_backup.tar")
        self.evaldir = osp.join(self.expdir, "eval")
        os.makedirs(self.evaldir, exist_ok=True)

    
        # Network
        if self.use_siren == True:
            print("use SIREN")
            from .network_siren import get_network
            network = get_network(cfg["network"]["net_type"])
            self.encoder = get_encoder(**cfg["encoder"])
            inr_config = {
                "in_features": cfg["encoder"]["input_dim"],
                "hidden_dim": cfg["network"]["hidden_dim"],
                "hidden_layers": cfg["network"]["num_layers"]-1,
                "out_dim": cfg["network"]["out_dim"],
                "first_omega_0": cfg["network"]["first_omega_0"],
                "hidden_omega_0": cfg["network"]["hidden_omega_0"],
                "outermost_linear": True,
                "bound": cfg["network"]["bound"],
            }
            self.net = network(inr_type="siren", inr_config=inr_config).to(device)

        else:
            from .network import get_network
            print("no SIREN")
            network = get_network(cfg["network"]["net_type"])
            cfg["network"].pop("net_type", None)
            self.encoder = get_encoder(**cfg["encoder"])
            self.net = network(self.encoder, **cfg["network"]).to(device)



        self.net_fine = None

        if cfg["train"]["use_muon"] == False:
            print("no Muon")
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr_adapt)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=cfg["train"]["lrate_step"], gamma=cfg["train"]["lrate_gamma"])
        else:
            print("use Muon")
            all_params = list(self.net.named_parameters())
            if self.use_siren == False:
                muon_params = [f"layers.{i}.weight" for i, layer in enumerate(self.net.layers) if isinstance(layer, torch.nn.Linear) ]
            else:
                muon_params = [f"inr.net.{i}.linear.weight" for i, layer in enumerate(self.net.inr.net_w_layers) if hasattr(layer, "linear")]

            muon_params = muon_params[1:-1]  # don't include first and last weight for training
            hidden_weights = [ param for name, param in all_params if name in muon_params]
            if hidden_weights == []:
                breakpoint()
            non_hidden_params = [ param for name, param in all_params if name not in muon_params]
            param_groups = [
                dict(params=hidden_weights, use_muon=True,
                     lr=cfg["train"]["lr_adapt_muon"]), #0.9,0.95
                dict(params=non_hidden_params, use_muon=False,
                     lr=cfg["train"]["lr_adapt_adam_muon"], betas=(0.9, 0.999) ) #weight_decay=0.01),
            ]

            self.optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg["train"]["epochs_adapt"],eta_min=1e-6)



        # Load checkpoints
        self.epoch_start = 0
        if cfg["train"]["resume"] and osp.exists(self.ckptdir):
            print(f"Load checkpoints from {self.ckptdir}.")
            ckpt = torch.load(self.ckptdir)
            self.epoch_start = ckpt["epoch"] + 1
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.global_step = self.epoch_start * len(self.train_dloader)
            self.net.load_state_dict(ckpt["network"])
            if self.n_fine > 0:
                self.net_fine.load_state_dict(ckpt["network_fine"])

        # Summary writer
        self.writer = SummaryWriter(self.expdir)
        self.writer.add_text("parameters", self.args2string(cfg), global_step=0)

    def args2string(self, hp):
        """
        Transfer args to string.
        """
        json_hp = json.dumps(hp, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))

    def start(self):
        """
        Main loop.
        """
        def fmt_loss_str(losses):
            return "".join(", " + k + ": " + f"{losses[k].item():.3g}" for k in losses)

        start_time_adaption = time.perf_counter()

        projection_path_adapt_volume = "" + "_projections.pickle"
        print(projection_path_adapt_volume)


        adapt_state = True
        self.train_dset = Dataset(projection_path_adapt_volume, self.conf["train"]["n_rays"], "train", self.device, self.inner_steps, adapt_state, outpaint=self.outpaint)
        self.eval_dset = Dataset(projection_path_adapt_volume, self.conf["train"]["n_rays"], "val", self.device,self.inner_steps, adapt_state, outpaint=self.outpaint) if self.i_eval > 0 else None
        train_dloader = torch.utils.data.DataLoader(self.train_dset, batch_size=self.conf["train"]["n_batch_adapt"],shuffle=True)

        init_sirt = np.zeros((256,256,256))

        self.img_init = (torch.tensor(init_sirt).to(self.device)).float()

        for idx_epoch_adapt in range(self.epochs_adapt):

            # Train
            self.net.train()
            print("epoch :", idx_epoch_adapt)

            print(f"Epoch {idx_epoch_adapt}: LR = {self.optimizer.param_groups[0]['lr']}")

            # Evaluate
            if ((idx_epoch_adapt % self.i_eval_adapt == 0) and (idx_epoch_adapt!=0)):
                self.net.eval()
                with torch.no_grad():
                    if idx_epoch_adapt!=0:#(idx_epoch_adapt % 25 == 0):
                            self.save_outpainted_projections(self.img_init, idx_epoch=idx_epoch_adapt, train_dset=self.train_dset, adapted_model=self.net,adapt_epoch=True, use_siren=self.use_siren)
                    loss_test = self.eval_step(self.img_init, global_step=self.global_step, idx_epoch=idx_epoch_adapt, adapt_epoch=True, use_siren=self.use_siren)
                    #torch.save(self.net.state_dict(), self.expdir + "adapted_model.pth")
                self.net.train()
                tqdm.write(f"[EVAL] epoch: {idx_epoch_adapt}/{self.epochs_train}{fmt_loss_str(loss_test)}")


            for data in train_dloader:

                breakpoint()

                self.global_step += 1
                self.maml_adapt(data, self.img_init, global_step=self.global_step, idx_epoch=idx_epoch_adapt)


        tqdm.write(f"Adpatation complete! See logs in {self.expdir}")

        elapsed_adapted = time.perf_counter() - start_time_adaption
        print(f"elapsed time for adaption: {elapsed_adapted:.4f} s")


    def maml_adapt(self, data_b, img_init, global_step, idx_epoch):

        adapt_model = self.adapt(data_b, img_init, global_step, idx_epoch, self.inner_steps, self.batch_size_adapt, use_siren=self.use_siren)

        return adapt_model


    def train_step_volume(self, global_step, idx_epoch):
        """
        Training step
        """
        self.optimizer.zero_grad()
        loss = self.compute_loss_volume(global_step, idx_epoch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_step_projections(self, data, global_step, idx_epoch):
        """
        Training step
        """
        self.optimizer.zero_grad()
        loss = self.compute_loss(data, global_step, idx_epoch)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def compute_loss(self, data, global_step, idx_epoch, use_siren):
        """
        Training step
        """
        raise NotImplementedError()

    def compute_loss_volume(self, global_step, idx_epoch):
        """
        Training step
        """
        raise NotImplementedError()


    def eval_step(self, global_step, idx_epoch, adapt_epoch, use_siren):
        """
        Evaluation step
        """
        raise NotImplementedError()


    def adapt(self, data, global_step, idx_epoch, inner_steps, meta_batch_size, use_siren):
        """
        inner_loop
        """
        return NotImplementedError()