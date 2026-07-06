import os

import cbct_artifact_reduction.dataprocessing as dp
import cbct_artifact_reduction.implantmaskcreator as imc
import matplotlib.pyplot as plt
import numpy as np
from cbct_artifact_reduction.utils import OUTPUT_DIR

RES = 256
output_folder_path = os.path.join(OUTPUT_DIR, "masks", f"{RES}x{RES}")

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)


def create_masks():
    imcTest = imc.ImplantMaskCreator((RES, RES))

    n = 1000

    for i in range(n):
        amount_implants = np.random.randint(1, 4)
        mask = imcTest.generate_mask_with_n_implants(amount_implants)
        dp.numpy_to_nifti(mask, os.path.join(output_folder_path, f"mask_{i}.nii.gz"))
        print(f"Generated mask {i + 1}/{n}")


if __name__ == "__main__":
    imcTest = imc.ImplantMaskCreator((RES, RES))
    mask = imcTest.generate_mask_with_n_implants(10)
    plt.imshow(mask)
    plt.show()
