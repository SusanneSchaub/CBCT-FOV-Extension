import os

import cbct_artifact_reduction.utils as utils
from cbct_artifact_reduction.dataprocessing import NiftiDataFolder

# Define the input and output directories
data_input_path = os.path.join(utils.OUTPUT_DIR, "resized", "256x256")
data_output_path = os.path.join(utils.OUTPUT_DIR, "frames", "256x256")

# Create a NiftiDataFolder object for the input directory
df = NiftiDataFolder(data_input_path)
df.split_all_volumes_into_frames(data_output_path)
