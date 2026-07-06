import os
import numpy as np
import ants
import matplotlib.pyplot as plt
from cbct_artifact_reduction.utils import (
    OUTPUT_DIR,
    nifti_to_numpy,
)


fixed_path = os.path.join(OUTPUT_DIR, "rotated", "10.nii.gz")
moving_paths = [
    os.path.join(OUTPUT_DIR, "rotated", f"{index}.nii.gz") for index in range(1, 2)
]
moving_path = moving_paths[0]


fixedNumpy = nifti_to_numpy(fixed_path)
movingNumpy = nifti_to_numpy(moving_path)

registeredNumpy = np.zeros(movingNumpy.shape)

for i in range(fixedNumpy.shape[2]):
    fixedSliceNumpy = fixedNumpy[:, :, i]
    movingSliceNumpy = movingNumpy[:, :, i]

    antsSliceFixed = ants.from_numpy(fixedSliceNumpy)
    antsSliceMoving = ants.from_numpy(movingSliceNumpy)

    registration = ants.registration(
        fixed=antsSliceFixed, moving=antsSliceMoving, type_of_transform="Affine"
    )

    registeredNumpy[:, :, i] = registration["warpedfixout"].numpy()

    # if i % 10 == 0:
    plt.imshow(fixedNumpy[:, 0 : 324 - 7, i] - registeredNumpy[:, :, i])
    plt.colorbar()
    plt.show()
