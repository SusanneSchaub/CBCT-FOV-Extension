import os

import cbct_artifact_reduction.dataprocessing as dp
import cbct_artifact_reduction.utils as utils
import matplotlib.pyplot as plt

np_array = dp.single_nifti_to_numpy(
    os.path.join(utils.ROOT_DIR, "output/new_frames_resized/51_100.nii.gz")
)

plt.imshow(np_array)
plt.show()

print(np_array.shape)
