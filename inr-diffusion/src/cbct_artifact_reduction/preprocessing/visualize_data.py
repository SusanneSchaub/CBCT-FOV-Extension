import os
from cbct_artifact_reduction.utils import (
    OUTPUT_DIR,
    nifti_to_numpy,
)

import matplotlib.pyplot as plt
import ants

data_path = os.path.join(OUTPUT_DIR, "registered")
files = [f for f in os.listdir(data_path) if f == "Affine.nii.gz"]
control_image = nifti_to_numpy(os.path.join(OUTPUT_DIR, "rotated", "10.nii.gz"))
moving_image = nifti_to_numpy(os.path.join(OUTPUT_DIR, "rotated", "1.nii.gz"))
frame = 7
# create subplot for all files in files
fig, axs = plt.subplots(1, len(files) + 3)
for index, file in enumerate(files):
    file_np_array = nifti_to_numpy(os.path.join(data_path, file))
    # plot in subplot
    plot_1 = axs[index].imshow(moving_image[:, :, frame] - file_np_array[:, :, frame])
    print(file_np_array.shape)
    print(control_image.shape)
    axs[index].set_title(file)
    plt.colorbar(plot_1, ax=axs[index])
# set colorbar

plot_control = axs[-1].imshow(control_image[:, :, frame])
axs[-1].set_title("Control")
plt.colorbar(plot_control, ax=axs[-1])
plot_moving = axs[-2].imshow(moving_image[:, :, frame])
axs[-2].set_title("Moving")
plt.colorbar(plot_moving, ax=axs[-2])
plot_registered = axs[-3].imshow(file_np_array[:, :, frame])
axs[-3].set_title("Registered")
plt.colorbar(plot_registered, ax=axs[-3])

# -------
fixed_path = os.path.join(OUTPUT_DIR, "rotated", "10.nii.gz")
moving_paths = [
    os.path.join(OUTPUT_DIR, "rotated", f"{index}.nii.gz") for index in range(1, 2)
]
moving_path_1 = moving_paths[0]

fixedAntsImage = ants.image_read(fixed_path)
movingAntsImage = ants.image_read(moving_path_1)

# print(fixedAntsImage.numpy().shape)
# print(movingAntsImage.numpy().shape)
# -------
plt.show()
