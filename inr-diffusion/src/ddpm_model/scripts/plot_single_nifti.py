import os

import cbct_artifact_reduction.dataprocessing as dp
import cbct_artifact_reduction.utils as utils
import matplotlib.pyplot as plt

path_to_file = os.path.join(utils.ROOT_DIR, "output/raw_nifti_data/529.nii.gz")

numpy_array = dp.single_nifti_to_numpy(path_to_file)
plt.imshow(numpy_array[:, :, 100])
plt.show()
