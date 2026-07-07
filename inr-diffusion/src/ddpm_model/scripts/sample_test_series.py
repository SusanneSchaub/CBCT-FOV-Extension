import os
from datetime import datetime

import cbct_artifact_reduction.config as cfg
import cbct_artifact_reduction.pigjawdataset as dataset
import nibabel as nib
import numpy as np
import torch
import time
from cbct_artifact_reduction import lakefs_own
from cbct_artifact_reduction.argparser_config import create_sample_argparser
from cbct_artifact_reduction.guided_diffusion import dist_util, logger
from cbct_artifact_reduction.guided_diffusion.script_util import (
     args_to_dict,
     create_model_and_diffusion,
     model_and_diffusion_defaults,
 )
from torch.utils.data import DataLoader
import pickle
import visdom
cfgg = {"server": "localhost", "port": 8823}
viz = visdom.Visdom('http://' + cfgg["server"], port = cfgg["port"])

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def min_max_normalize_common(img: np.ndarray):
    """Function used to normalize image to range [0, 1]."""

    _min = img.min()
    _max = 3.5 #1.2
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def min_max_normalize_common_reverse(img: np.ndarray, cond_min):
    """Function used to normalize image to range [0, 1]."""

    _min = cond_min
    _max = 3.5 #1.2
    denormalized = img*(_max - _min) + _min
    return denormalized


def main():

    args = create_sample_argparser().parse_args()
    #dist_util.setup_dist()
    logger.configure(os.path.expanduser(args.log_dir))

    SAMPLE_DIR = os.path.expanduser(args.log_dir)

    today = datetime.now()
    logger.log(f"SAMPLING {today}")
    logger.log(f"args: {args}")
    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    logger.log("creating dataloader...")

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path1, map_location="cpu")
    )

    num_samples = 10

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    list_samples=[]

    start = time.perf_counter()


    for i in range(num_samples):


        test_numbers = []


        for tst_nr in test_numbers:

            cond_vol = nib.load("" + str(tst_nr) + "_inr_vol.nii.gz").get_fdata()

            for j in range(0,cond_vol.shape[0]):
                cond = cond_vol[j,...]
                cond = cond.astype(np.float32)
                cond_org_min = cond.min()
                cond = min_max_normalize_common(cond)
                cond = torch.from_numpy(cond).to(dist_util.dev())
                cond=cond[None, None, ...]

                model_kwargs = {}

                sample_fn = diffusion.p_sample_loop_inpainting

                noise_only_outpaint=False

                sample, _ = sample_fn(
                    noise_only_outpaint,
                    model,
                    cond,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )

                print(j)
                denorm = min_max_normalize_common_reverse(sample, cond_org_min)
                list_samples.append(denorm)

            end = time.perf_counter()
            print(f"Elapsed time: {end - start:.6f} seconds")



            stacked = torch.stack(list_samples, dim=0)
            stacked = stacked[:,0,0,:,:]
            stacked = stacked.cpu().detach().numpy()



            new_image = nib.Nifti1Image(stacked, affine=np.eye(4))

            nib.save(new_image,"" + str(tst_nr) + "_dmodel.nii.gz")
            list_samples = []



    pass


if __name__ == "__main__":
    main()
