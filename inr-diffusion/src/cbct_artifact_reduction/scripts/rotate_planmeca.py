import os

import cbct_artifact_reduction.dataprocessing as dp
import cbct_artifact_reduction.utils as utils
import nibabel as nib
import numpy as np

path_to_data_folder = os.path.join(utils.ROOT_DIR, "output/rotated")
output_path = os.path.join(utils.ROOT_DIR, "output/truly_rotated")

if not os.path.exists(output_path):
    os.makedirs(output_path)

item_names = [f for f in os.listdir(path_to_data_folder) if f.endswith(".nii.gz")]


get_id_function = dp.filename_without_extension

item_names.sort(key=lambda x: int(get_id_function(x)))

planmecaIds = utils.getAllplanmecaIDs()
for item in item_names:
    id = get_id_function(item)
    if id not in planmecaIds:
        continue

    print(f"Processing {id}")
    item_path = os.path.join(path_to_data_folder, item)
    item_numpy = dp.single_nifti_to_numpy(item_path)

    shape = item_numpy.shape
    img_corrected = np.rot90(item_numpy, 1)

    nib_corrected = nib.nifti1.Nifti1Image(img_corrected, np.eye(4))
    nib.nifti1.save(nib_corrected, os.path.join(output_path, item))

    # row = [id, shape[0], shape[1], shape[2]]
    # writer.writerow(row)
