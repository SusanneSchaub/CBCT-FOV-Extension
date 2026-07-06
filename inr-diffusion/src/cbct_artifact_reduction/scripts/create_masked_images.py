import os

import cbct_artifact_reduction.dataprocessing as dp
import cbct_artifact_reduction.implantmaskcreator as imc
import cbct_artifact_reduction.scanner as scanner
import cbct_artifact_reduction.utils as utils
import matplotlib.pyplot as plt

OUTPUT_DIR = utils.OUTPUT_DIR
creator = imc.ImplantMaskCreator((256, 256))
image = dp.single_nifti_to_numpy(
    # 582, und 591 haben implantate obwohl sie keine haben sollten
    os.path.join(OUTPUT_DIR, "frames", "256x256", "545_100.nii.gz")
)
planmeca_folder = os.path.join(
    utils.DATA_DIR,
    "pigjaw 7",
    "planmeca pigjaw 7",
    "projection",
    "sll ti 2",
)
dirs = os.listdir(planmeca_folder)
tar_gz_path = ""
for d in dirs:
    if d.endswith(".tar.gz"):
        tar_gz_path = os.path.join(planmeca_folder, d)

original_planmeca_image = scanner.planmeca_folder_to_numpy(tar_gz_path)
plt.imshow(original_planmeca_image[:, :, 100])
plt.show()
