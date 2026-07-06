import os

from skimage import io

import cbct_artifact_reduction.dataprocessing as dp
import cbct_artifact_reduction.utils as utils
from cbct_artifact_reduction.dataprocessing import NiftiDataFolder

# Define the input and output directories
data_input_path = os.path.join(utils.OUTPUT_DIR, "frames", "256x256")
data_output_path = os.path.join(utils.OUTPUT_DIR, "mask", "256x256")


if not os.path.exists(data_output_path):
    os.makedirs(data_output_path)

# Create a NiftiDataFolder object for the input directory
df = NiftiDataFolder(data_input_path)

data_path_list = df.data_path_list

for path in data_path_list:
    print(f"processing {path}")
    data_numpy = dp.single_nifti_to_numpy(path)
    normalized_data_numpy = dp.min_max_normalize(data_numpy)
    mask = dp.create_binary_threshold_mask(normalized_data_numpy, threshold=0.5)
    mask_path = os.path.join(data_output_path, os.path.basename(path))
    dp.numpy_to_nifti(mask, mask_path)
    io.imshow(mask)
    io.show()
    break
