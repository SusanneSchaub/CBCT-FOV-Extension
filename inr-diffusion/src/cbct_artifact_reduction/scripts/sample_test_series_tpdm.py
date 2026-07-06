import os
from datetime import datetime

import nibabel as nib
import numpy as np
import torch
import time
from cbct_artifact_reduction.argparser_config_tpdm import create_sample_argparser
from cbct_artifact_reduction.guided_diffusion import dist_util, logger
from cbct_artifact_reduction.guided_diffusion.script_util_tpdm import (
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
    _max = 3.5
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


    model1, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model2, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    logger.log("creating dataloader...")

    model1.load_state_dict(dist_util.load_state_dict(args.model_path1, map_location="cpu"))
    model2.load_state_dict(dist_util.load_state_dict(args.model_path2, map_location="cpu"))

    num_samples = 1

    model1.to(dist_util.dev())
    model2.to(dist_util.dev())
    if args.use_fp16:
        model1.convert_to_fp16()
        model2.convert_to_fp16()

    model1.eval()
    model2.eval()


    start = time.perf_counter()


    for i in range(num_samples):

        test_numbers = [4, 8, 9, 10, 19, 23, 26, 28, 33, 46,  # 1 to 171
                   199, 205, 208, 209, 250, 253, 255, 262, 278, 305,  # 172 to 328
                   333, 340, 342, 344, 352, 385, 410, 415, 428, 496,  # 330 to 497
                   508, 519, 528, 534, 545, 546, 561, 558, 559, 574,  # 500 to 658
                   ]

        for tst_nr in test_numbers:

            cond_vol1 = nib.load("/raid/cian/susanne.schaub/MMDental/lakefs_volumes_reco/testing/" + str(tst_nr) + "_ossart_vol.nii.gz").get_fdata()

            cond_norm = np.zeros_like(cond_vol1)

            cond_org_min = cond_vol1.min()

            for j in range(0,cond_vol1.shape[0]): #69 x800
                cond1 = cond_vol1[j,...]
                cond1 = cond1.astype(np.float32)
                cond1 = min_max_normalize_common(cond1) # check if thats only preprocessing!!!
                cond_norm[j,...] = cond1


            cond_norm = torch.from_numpy(cond_norm).to(dist_util.dev())
            cond_norm = cond_norm[None, None, ...]

            model_kwargs = {}

            sample_fn = diffusion.p_sample_loop_inpainting

            sample, _ = sample_fn(
                model1,
                model2,
                cond_norm,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )



            denorm = min_max_normalize_common_reverse(sample, cond_org_min)

            end = time.perf_counter()
            print(f"Elapsed time: {end - start:.6f} seconds")


            sam = denorm[0,0,...].cpu().detach().numpy()

            new_image = nib.Nifti1Image(sam, affine=np.eye(4))
            nib.save(new_image, "/raid/cian/susanne.schaub/INR-Diffusion/train_ossart/tpdm_samples_40000_0/" + str(tst_nr) + "_reco_tpdm.nii.gz")


        print("Done")


    pass


if __name__ == "__main__":
    main()
