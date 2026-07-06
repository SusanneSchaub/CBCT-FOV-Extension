import os
import random

import cbct_artifact_reduction.config as cfg
import cbct_artifact_reduction.csvcreator as csvcreator
import cbct_artifact_reduction.lakefs_own as lakefs_own
import matplotlib
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np
from cbct_artifact_reduction.dataprocessing import (
    single_nifti_to_numpy,
)
from cbct_artifact_reduction.utils import ROOT_DIR, lookup_num_in_datatable
from matplotlib import rc
from tqdm import tqdm

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)
rc_fonts = {
    "font.family": "serif",
    "font.serif": "Linux Libertine O",
    "font.size": 20,
}
matplotlib.rcParams.update(rc_fonts)

data_csv = os.path.join(ROOT_DIR, "data.csv")
client = lakefs_own.CustomBoto3Client(f"{cfg.LAKEFS_DATA_REPOSITORY}")
lakefs_folder = "processed_data/frames/256x256/"
# local_path = client.get_file("processed_data/frames/256x256/121.nii.gz")

pd = csvcreator.filter_csv(
    data_csv, output_path=None, exclude_filled_radiation=True, implants=0
)

slice_paths = [
    os.path.join(lakefs_folder, i)
    for i in csvcreator.get_slices_from_ids(data_csv, pd.id.tolist())  # type: ignore
]

N = 1000

log_hist_counts_normalized = {
    "axeos_large": np.zeros(N),
    "axeos_small": np.zeros(N),
    "planmeca_large": np.zeros(N),
    "planmeca_small": np.zeros(N),
    "x800_large": np.zeros(N),
    "x800_small": np.zeros(N),
    "accuitomo_large": np.zeros(N),
    "accuitomo_small": np.zeros(N),
}
log_hist_counts = {
    "axeos_large": np.zeros(N),
    "axeos_small": np.zeros(N),
    "planmeca_large": np.zeros(N),
    "planmeca_small": np.zeros(N),
    "x800_large": np.zeros(N),
    "x800_small": np.zeros(N),
    "accuitomo_large": np.zeros(N),
    "accuitomo_small": np.zeros(N),
}

hist_counts_normalized = {
    "axeos_large": np.zeros(N),
    "axeos_small": np.zeros(N),
    "planmeca_large": np.zeros(N),
    "planmeca_small": np.zeros(N),
    "x800_large": np.zeros(N),
    "x800_small": np.zeros(N),
    "accuitomo_large": np.zeros(N),
    "accuitomo_small": np.zeros(N),
}
hist_counts = {
    "axeos_large": np.zeros(N),
    "axeos_small": np.zeros(N),
    "planmeca_large": np.zeros(N),
    "planmeca_small": np.zeros(N),
    "x800_large": np.zeros(N),
    "x800_small": np.zeros(N),
    "accuitomo_large": np.zeros(N),
    "accuitomo_small": np.zeros(N),
}
occurences = {
    "axeos_large": 0,
    "axeos_small": 0,
    "planmeca_large": 0,
    "planmeca_small": 0,
    "x800_large": 0,
    "x800_small": 0,
    "accuitomo_large": 0,
    "accuitomo_small": 0,
}

log_transforms = {
    "x800_large": lambda k: -np.log(k / k.max()),
    "x800_small": lambda k: -np.log(k / k.max()),
    "planmeca_large": lambda k: -np.log(k / (4326 * 2.27)),
    "planmeca_small": lambda k: -np.log(k / (3591 * 2.27)),
    "axeos_large": lambda k: -np.log(k / (2 * 10**16)),
    "axeos_small": lambda k: -np.log(k / (2 * 10**16)),
    "accuitomo_large": lambda k: -np.log(k / (k.max())),
    "accuitomo_small": lambda k: -np.log(k / (k.max())),
}

max = 500
i = 0
random.seed(42)
random.shuffle(slice_paths)
for path in tqdm(slice_paths, desc="Collecting information about slices"):
    i += 1
    id = int(os.path.basename(path).split("_")[0])
    info = lookup_num_in_datatable(id)
    scanner = info["scanner"]  # type: ignore
    fov = info["fov"]  # type: ignore

    local_path = client.get_file(path)

    np_array = single_nifti_to_numpy(local_path)  # type: ignore
    pixels = np_array.flatten()

    mean = np.mean(pixels)
    if mean < 100:
        continue

    # pixels = remove_outliers(pixels)

    _min = pixels.min()
    _max = pixels.max()

    pixels_normalized = (pixels - _min) / (_max - _min)

    log_pixels = log_transforms[f"{scanner}_{fov}"](pixels)
    log_pixels_normalized = (log_pixels - log_pixels.min()) / (
        log_pixels.max() - log_pixels.min()
    )

    log_counts, bin_edges_log = np.histogram(log_pixels, bins=N, range=(0, 6))
    counts, bin_edges = np.histogram(pixels, bins=N, range=(0, 53000))
    log_counts_normalized, bin_edges_log_normalized = np.histogram(
        log_pixels_normalized, bins=N, range=(0, 1)
    )
    counts_normalized, bin_edges_normalized = np.histogram(
        pixels_normalized, bins=N, range=(0, 1)
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_centers_log = (bin_edges_log[:-1] + bin_edges_log[1:]) / 2
    bin_centers_normalized = (bin_edges_normalized[:-1] + bin_edges_normalized[1:]) / 2
    bin_centers_log_normalized = (
        bin_edges_log_normalized[:-1] + bin_edges_log_normalized[1:]
    ) / 2

    hist_counts[f"{scanner}_{fov}"] += counts
    log_hist_counts[f"{scanner}_{fov}"] += log_counts
    hist_counts_normalized[f"{scanner}_{fov}"] += counts_normalized
    log_hist_counts_normalized[f"{scanner}_{fov}"] += log_counts_normalized
    occurences[f"{scanner}_{fov}"] += 1
    if i == max:
        break


fig, axs = plt.subplots(4, figsize=(15.1, 10), dpi=256)

plot_log = False
normalized = False

colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
i = 0
for key, value in hist_counts.items():
    if True:
        single_occurence = occurences.get(key)
        print(np.sum(value) / single_occurence)  # type: ignore

        # y = value / single_occurence  # type: ignore
        if plot_log:
            if normalized:
                y = log_hist_counts_normalized[key]
                x = bin_centers_log_normalized
                w = np.diff(bin_edges_log_normalized)
            else:
                y = log_hist_counts[key]
                x = bin_centers_log
                w = np.diff(bin_edges_log)
        else:
            if normalized:
                y = hist_counts_normalized[key]
                x = bin_centers_normalized
                w = np.diff(bin_edges_normalized)
            else:
                y = hist_counts[key]
                x = bin_centers
                w = np.diff(bin_edges)
        axs[0].bar(
            x,
            y / single_occurence,  # type: ignore
            width=w,
            align="center",
            alpha=0.5,
            label=key,
            color=colors[i],
        )
        i = i + 1


plot_log = False
normalized = True
i = 0
for key, value in hist_counts.items():
    if True:
        single_occurence = occurences.get(key)
        print(np.sum(value) / single_occurence)  # type: ignore

        # y = value / single_occurence  # type: ignore
        if plot_log:
            if normalized:
                y = log_hist_counts_normalized[key]
                x = bin_centers_log_normalized
                w = np.diff(bin_edges_log_normalized)
            else:
                y = log_hist_counts[key]
                x = bin_centers_log
                w = np.diff(bin_edges_log)
        else:
            if normalized:
                y = hist_counts_normalized[key]
                x = bin_centers_normalized
                w = np.diff(bin_edges_normalized)
            else:
                y = hist_counts[key]
                x = bin_centers
                w = np.diff(bin_edges)
        axs[1].bar(
            x,
            y / single_occurence,  # type: ignore
            width=w,
            align="center",
            alpha=0.5,
            label=key,
            color=colors[i],
        )
        i = i + 1

plot_log = True
normalized = False
i = 0
for key, value in hist_counts.items():
    if True:
        single_occurence = occurences.get(key)
        print(np.sum(value) / single_occurence)  # type: ignore

        # y = value / single_occurence  # type: ignore
        if plot_log:
            if normalized:
                y = log_hist_counts_normalized[key]
                x = bin_centers_log_normalized
                w = np.diff(bin_edges_log_normalized)
            else:
                y = log_hist_counts[key]
                x = bin_centers_log
                w = np.diff(bin_edges_log)
        else:
            if normalized:
                y = hist_counts_normalized[key]
                x = bin_centers_normalized
                w = np.diff(bin_edges_normalized)
            else:
                y = hist_counts[key]
                x = bin_centers
                w = np.diff(bin_edges)
        axs[2].bar(
            x,
            y / single_occurence,  # type: ignore
            width=w,
            align="center",
            alpha=0.5,
            label=key,
            color=colors[i],
        )
        i = i + 1


plot_log = True
normalized = True
i = 0
for key, value in hist_counts.items():
    if True:
        single_occurence = occurences.get(key)
        print(np.sum(value) / single_occurence)  # type: ignore

        # y = value / single_occurence  # type: ignore
        if plot_log:
            if normalized:
                y = log_hist_counts_normalized[key]
                x = bin_centers_log_normalized
                w = np.diff(bin_edges_log_normalized)
            else:
                y = log_hist_counts[key]
                x = bin_centers_log
                w = np.diff(bin_edges_log)
        else:
            if normalized:
                y = hist_counts_normalized[key]
                x = bin_centers_normalized
                w = np.diff(bin_edges_normalized)
            else:
                y = hist_counts[key]
                x = bin_centers
                w = np.diff(bin_edges)
        axs[3].bar(
            x,
            y / single_occurence,  # type: ignore
            width=w,
            align="center",
            alpha=0.5,
            label=key,
            color=colors[i],
        )
        i = i + 1


fig.text(0.5, 0.04, "Pixel value", ha="center")
fig.text(0.04, 0.5, "Counts", va="center", rotation="vertical")
handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(labels, loc="best", ncol=4, fontsize="small")
# plt.xlim(0, 53000)


axs[0].set_ylim(0, 500)
axs[1].set_ylim(0, 500)
axs[2].set_ylim(0, 500)
axs[3].set_ylim(0, 500)

axs2 = axs[0].twinx()
axs2.set_ylabel("raw", rotation=270, labelpad=20)
axs2.set_yticks([])
axs3 = axs[1].twinx()
axs3.set_ylabel("normalized", rotation=270, labelpad=20)
axs3.set_yticks([])
axs4 = axs[2].twinx()
axs4.set_ylabel("log", rotation=270, labelpad=20)
axs4.set_yticks([])
axs5 = axs[3].twinx()
axs5.set_ylabel("log normalized", rotation=270, labelpad=20)
axs5.set_yticks([])

axs[0].set_title("Histograms of pixel values")
plt.savefig("hist.png")
