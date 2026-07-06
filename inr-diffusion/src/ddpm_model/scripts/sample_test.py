import os
from datetime import datetime

import cbct_artifact_reduction.config as cfg
import cbct_artifact_reduction.pigjawdataset as dataset
import nibabel as nib
import numpy as np
import torch
from cbct_artifact_reduction import lakefs_own
from cbct_artifact_reduction.argparser_config import create_sample_argparser
from cbct_artifact_reduction.guided_diffusion import dist_util, logger
from cbct_artifact_reduction.guided_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from torch.utils.data import DataLoader
import visdom
cfgg = {"server": "localhost", "port": 8823}
viz = visdom.Visdom('http://' + cfgg["server"], port = cfgg["port"])

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


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
    client = lakefs_own.CustomBoto3Client(f"{cfg.LAKEFS_DATA_REPOSITORY}")
    inpaintingSliceDataset = dataset.InpaintingSliceDataset(
        client,
        os.path.join(cfg.ROOT_DIR, args.data_csv),
        "processed_data/frames/256x256",
        random_masks=args.random_masks,
        augment_data=args.augment_data,
        log_transform=args.log_transform,
    )

    dataloader = DataLoader(
        inpaintingSliceDataset, batch_size=args.batch_size, shuffle=True
    )

    data = iter(dataloader)

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    num_samples = 10

    model.to("cuda")
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    for i in range(num_samples):
        item = next(data)
        ground_truth, mask, info = item["slice"], item["mask"], item["info"]
        #if args.batch_size == 1:
        #    info = dataset.clean_dict(info)

        mask = nib.load("/home/s.schaub/cbct-artifact-reduction-rework/preprocessed_mask.nii.gz").get_fdata()
        mask = np.flipud(mask)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask).to("cuda")
        mask = mask[None, None, ...]
        ground_truth = ground_truth.to("cuda")
        #masked_image = ground_truth * (1 - mask)
        masked_image = ground_truth * (mask)
        mask=1-mask

        use_yulan = True
        if use_yulan == True:
            ground_truth = nib.load("/home/s.schaub/cbct-artifact-reduction-rework/preprocessed.nii.gz").get_fdata()
            mask = nib.load("/home/s.schaub/cbct-artifact-reduction-rework/preprocessed_mask.nii.gz").get_fdata()
            mask=mask[...,0]
            ground_truth = np.flipud(ground_truth)
            mask = np.flipud(mask)
            ground_truth = ground_truth.astype(np.float32)
            mask = mask.astype(np.float32)
            ground_truth = torch.from_numpy(ground_truth).to("cuda")
            mask = torch.from_numpy(mask).to("cuda")
            ground_truth=ground_truth[None, None, ...]
            mask = mask[None, None, ...]
            masked_image = ground_truth * (1-mask)



        model_kwargs = {}

        sample_fn = diffusion.p_sample_loop_inpainting

        sample, _ = sample_fn(
            model,
            masked_image,
            mask,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        breakpoint()
        if sample is not None:
            sample = sample.detach().cpu()
            mask = mask.detach().cpu()
            ground_truth = ground_truth.detach().cpu()
            sample = torch.cat([sample, ground_truth, mask], dim=1).numpy()
            sample_nifti_object = nib.nifti1.Nifti1Image(sample, None)
            nib.save(
                sample_nifti_object, os.path.join(SAMPLE_DIR, f"sample_{i}.nii.gz")
            )

            logger.log(f"Saved sample {i}")
            logger.log(info)

    pass


if __name__ == "__main__":
    main()
