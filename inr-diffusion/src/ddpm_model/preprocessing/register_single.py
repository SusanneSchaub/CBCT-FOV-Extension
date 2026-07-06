import ants
from cbct_artifact_reduction.utils import (
    OUTPUT_DIR,
)
import os

fixed_path = os.path.join(OUTPUT_DIR, "rotated", "10.nii.gz")
moving_path = os.path.join(OUTPUT_DIR, "rotated", "1.nii.gz")

fixedAntsImage = ants.image_read(fixed_path)
movingAntsImage = ants.image_read(moving_path)

registrationAnts = ants.registration(
    fixed=fixedAntsImage, moving=movingAntsImage, type_of_transform="Affine"
)

differenceNumpy = fixedAntsImage.numpy() - registrationAnts["warpedmovout"].numpy()
differenceAntsImage = ants.from_numpy(differenceNumpy)
ants.image_write(
    differenceAntsImage, os.path.join(OUTPUT_DIR, "registered", "difference.nii.gz")
)
