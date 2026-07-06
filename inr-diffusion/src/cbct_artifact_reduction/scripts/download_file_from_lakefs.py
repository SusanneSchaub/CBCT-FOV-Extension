import cbct_artifact_reduction.config as cfg
import cbct_artifact_reduction.lakefs_own as lakefs_own
import matplotlib.pyplot as plt
from cbct_artifact_reduction.dataprocessing import single_nifti_to_numpy

client = lakefs_own.CustomBoto3Client(f"{cfg.LAKEFS_DATA_REPOSITORY}")
local_path = client.get_file("processed_data/rotated/121.nii.gz")
print(local_path)
if local_path:
    np_array = single_nifti_to_numpy(local_path)
    plt.imshow(np_array[:, :, 100])
    plt.show()
