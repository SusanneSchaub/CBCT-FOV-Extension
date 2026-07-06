import os

import numpy as np
import torch
import tqdm
from cbct_artifact_reduction.config import ROOT_DIR
from cbct_artifact_reduction.dataprocessing import (
    single_nifti_to_numpy,
)
from cbct_artifact_reduction.scripts.results.ssim_custom import ssim as ssim_mask
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

SAMPLE_DIR = os.path.join(ROOT_DIR, "samples", "20250328_225115")

# Get all names of the samples in sample directory
sample_names = [f for f in os.listdir(SAMPLE_DIR) if f.endswith(".nii.gz")]


NUMBER_OF_SAMPLES = len(sample_names)

data = []  # List of dicts {"name": name, "data": np_array, "ssim": ssim, "psnr": psnr, "mse": mse}

for i, file_name in enumerate(tqdm((sample_names), desc="Calculating scores")):
    batch = single_nifti_to_numpy(os.path.join(SAMPLE_DIR, file_name))

    sample, gt, mask = [i[0, ...] for i in np.split(batch, 3, axis=0)]

    data_range = gt.max() - gt.min()

    # if sample contains nan values
    # if np.isnan(sample).any():
    #     continue

    ssim_score = ssim(gt, sample, data_range=data_range)
    ssim_mask_score = ssim_mask(
        torch.from_numpy(gt)[None, None, ...],
        torch.from_numpy(sample)[None, None, ...],
        data_range=data_range,
        mask=torch.from_numpy(mask)[None, None, ...] > 0,
    )
    psnr_score = psnr(gt, sample, data_range=data_range)
    mse_score = mse(gt, sample)
    data.append(
        {
            "name": file_name,
            "data": sample,
            "ssim": ssim_score,
            "ssim_mask": ssim_mask_score,
            "psnr": psnr_score,
            "mse": mse_score,
        }
    )
    # print(f"{ssim_score=}, {ssim_mask_score=}, {psnr_score=}, {mse_score=}")

# Calculate the mean and standard deviation of the all scores
metrics = ["ssim", "ssim_mask", "psnr", "mse"]
stats = {
    metric: {
        "mean": float(np.mean([d[metric] for d in data])),
        "std": float(np.std([d[metric] for d in data])),
        "min": float(np.min([d[metric] for d in data])),
        "max": float(np.max([d[metric] for d in data])),
        "argmin": int(np.argmin([d[metric] for d in data])),
    }
    for metric in metrics
}

print(stats)
