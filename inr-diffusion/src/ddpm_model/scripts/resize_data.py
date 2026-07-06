import os

from cbct_artifact_reduction.dataprocessing import NiftiDataFolder
from cbct_artifact_reduction.utils import OUTPUT_DIR

data_path = os.path.join(OUTPUT_DIR, "rotated")
output_folder_path = os.path.join(OUTPUT_DIR, "resized")
new_dimensions = (256, 256)
overwrite_files = True
preserve_range = False

df = NiftiDataFolder(data_path)
df.resize_all_files(output_folder_path, new_dimensions, overwrite_files, preserve_range)
