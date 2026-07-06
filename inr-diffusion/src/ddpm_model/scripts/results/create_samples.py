import datetime
import os
from datetime import datetime

import cbct_artifact_reduction.config as cfg
import cbct_artifact_reduction.pigjawdataset as dataset
import cbct_artifact_reduction.scripts.results.create_sample_csv as create_sample_csv
import nibabel as nib
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


def create_folder_based_on_start_time(base_dir):
    # 1. Capture the current date and time
    start_time = datetime.now()

    # 2. Convert this time to a formatted string (e.g., YYYYMMDD_HHMMSS)
    folder_name = start_time.strftime("%Y%m%d_%H%M%S")

    # 3. Create the folder (using exist_ok to avoid errors if the folder already exists)
    os.makedirs(os.path.join(base_dir, folder_name))

    # 4. Return the folder name for reference
    return os.path.join(base_dir, folder_name)


def main():
    args = create_sample_argparser().parse_args()
    dist_util.setup_dist()
    SAMPLE_DIR = os.path.expanduser(args.log_dir)
    folder = create_folder_based_on_start_time(SAMPLE_DIR)
    logger.configure(os.path.expanduser(folder))

    slice_list = create_sample_csv.get_sampling_names()

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
        slice_list,
        "processed_data/frames/256x256",
        random_masks=False,
        augment_data=False,
        log_transform=True,
    )

    dataloader = DataLoader(
        inpaintingSliceDataset, batch_size=args.batch_size, shuffle=False
    )

    data = iter(dataloader)

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    for item in data:
        ground_truth, mask, info = item["slice"], item["mask"], item["info"]
        masked_image = ground_truth * (1 - mask)

        model_kwargs = {}

        sample_fn = diffusion.p_sample_loop_inpainting

        sample, _ = sample_fn(
            model,
            masked_image,
            mask,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        if sample is not None:
            batch_size = sample.shape[0]
            sample = sample.detach().cpu()
            mask = mask.detach().cpu()
            ground_truth = ground_truth.detach().cpu()
            sample = torch.cat([sample, ground_truth, mask], dim=1).numpy()
            for i in range(batch_size):
                slice_name = info["filename"][i]
                sample_nifti_object = nib.nifti1.Nifti1Image(sample[i, ...], None)
                nib.save(
                    sample_nifti_object,
                    os.path.join(folder, f"sample_{slice_name}"),
                )

                logger.log(f"Saved sample sample_{slice_name}")
                logger.log(info)

    pass


if __name__ == "__main__":
    main()
