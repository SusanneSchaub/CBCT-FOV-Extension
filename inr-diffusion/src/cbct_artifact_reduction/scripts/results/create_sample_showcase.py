import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from cbct_artifact_reduction.config import ROOT_DIR
from cbct_artifact_reduction.dataprocessing import (
    single_nifti_to_numpy,
)
from matplotlib import rc
from tqdm import tqdm

SAMPLE_DIR = os.path.join(ROOT_DIR, "samples", "20250328_225115")

# Get all names of the samples in sample directory
sample_names = [f for f in os.listdir(SAMPLE_DIR) if f.endswith(".nii.gz")]


NUMBER_OF_SAMPLES = len(sample_names)


rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)
rc_fonts = {
    "font.family": "serif",
    "font.serif": "Linux Libertine O",
    "font.size": 20,
}
matplotlib.rcParams.update(rc_fonts)


def plot_triplet_column(masked_list, sample_list, gt_list, n_rows=4):
    """
    Plots N triplets in a vertical layout:
    Masked | Sample | Ground Truth
    """
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3 * n_rows), dpi=256)
    titles = ["Masked Image", "Sample", "Ground Truth"]

    for row in range(n_rows):
        imgs = [masked_list[row], sample_list[row], gt_list[row]]
        for col in range(3):
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.imshow(imgs[col], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")

            # Add column labels to top row
            if row == 0:
                ax.set_title(titles[col])

    plt.tight_layout()
    plt.savefig("sample_showcase.png")
    plt.show()


masked_list = []
output_list = []
gt_list = []
for i, file_name in enumerate(tqdm((sample_names), desc="Calculating scores")):
    batch = single_nifti_to_numpy(os.path.join(SAMPLE_DIR, file_name))
    sample, gt, mask = [i[0, ...] for i in np.split(batch, 3, axis=0)]
    output_list.append(sample)
    gt_list.append(gt)
    masked_list.append(gt * (1 - mask))

plot_triplet_column(masked_list, output_list, gt_list)
