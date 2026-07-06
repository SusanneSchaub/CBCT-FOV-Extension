import os
import glob
import numpy as np
import nibabel as nib
import torch
import random
from torch.utils.data import Dataset
from cbct_artifact_reduction.dataprocessing import (
    min_max_normalize_common,
    min_max_normalize_common_pig,
    remove_outliers,
    single_nifti_to_numpy,
)


import os
import glob
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

import visdom
cfgg = {"server": "localhost", "port": 8823}
viz = visdom.Visdom('http://' + cfgg["server"], port = cfgg["port"])

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


class PairedNiftiDataset(Dataset):
    def __init__(self, root_dir, augment_data=True):
        self.root_dir = root_dir
        self.augment_data = augment_data

        self.samples = []

        gt_files = sorted(
            glob.glob(os.path.join(root_dir, "*_gt_*.nii.gz"))
        )

        for gt_path in gt_files:
            inr_path = gt_path.replace("_gt_", "_inr_")

            if os.path.exists(inr_path):
                self.samples.append(
                    {
                        "input": inr_path,
                        "target": gt_path,
                    }
                )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No gt/inr pairs found in {root_dir}"
            )

        print(f"Found {len(self.samples)} pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        input_path = sample["input"]
        target_path = sample["target"]

        input_vol = nib.load(input_path).get_fdata().astype(np.float32)
        target_vol = nib.load(target_path).get_fdata().astype(np.float32)

        # Apply identical augmentation to both
        if self.augment_data and random.random() < 0.5:
            input_vol = input_vol[..., :, ::-1].copy()
            target_vol = target_vol[..., :, ::-1].copy()

        input_vol = min_max_normalize_common_pig(input_vol)
        target_vol = min_max_normalize_common_pig(target_vol)

        input_vol = torch.from_numpy(input_vol).float().unsqueeze(0)
        target_vol = torch.from_numpy(target_vol).float().unsqueeze(0)


        return {
            "slice": input_vol,
            "slice_gt": target_vol,
        }