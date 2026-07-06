import csv
import os

import cbct_artifact_reduction.dataprocessing as dp
import cbct_artifact_reduction.utils as utils

path_to_data_folder = os.path.join(utils.ROOT_DIR, "output/rotated")

item_names = [f for f in os.listdir(path_to_data_folder) if f.endswith(".nii.gz")]
# item_names = ["1.nii.gz"]


get_id_function = dp.filename_without_extension

item_names.sort(key=lambda x: int(get_id_function(x)))


with open(os.path.join(utils.ROOT_DIR, "data_shape.csv"), "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["id", "x", "y", "z"])
    for item in item_names:
        id = get_id_function(item)
        print(f"Processing {id}")
        item_path = os.path.join(path_to_data_folder, item)
        item_numpy = dp.single_nifti_to_numpy(item_path)
        shape = item_numpy.shape

        row = [id, shape[0], shape[1], shape[2]]
        writer.writerow(row)
