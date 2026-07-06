import os
import random

import cbct_artifact_reduction.csvcreator
import cbct_artifact_reduction.utils as utils
import pandas as pd

path_to_data_csv = os.path.join(utils.ROOT_DIR, "data.csv")

data_csv: pd.DataFrame = pd.read_csv(path_to_data_csv)


A = data_csv

selected_scanners = ["axeos", "planmeca"]

control_ids = data_csv[data_csv["implants"] == 0][
    data_csv["scanner"].isin(selected_scanners)
]["id"]
print(control_ids)
amount_of_slices = data_csv[data_csv["id"].isin(control_ids)]["frames"]
print(amount_of_slices)


slice_paths = []

for iter, id in enumerate(control_ids):
    num_slices = amount_of_slices.values[iter]
    for n in range(num_slices):
        slice_paths.append(f"{id}_{n}.nii.gz")

length = len(slice_paths)
mask_ids = random.choices([i for i in range(1, 1000)], k=length)

mask_paths = [f"mask_{id}.nii.gz" for id in mask_ids]

print(len(slice_paths), len(mask_paths))
csvCreator = cbct_artifact_reduction.csvcreator.createSliceMaskCSV(
    slice_paths, mask_paths, os.path.join(utils.ROOT_DIR, "training_data.csv")
)
