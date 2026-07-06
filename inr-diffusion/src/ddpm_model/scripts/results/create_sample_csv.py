import os
import random

import cbct_artifact_reduction.config as cfg
import cbct_artifact_reduction.csvcreator as csvcreator
import cbct_artifact_reduction.pigjawdataset as dataset
import numpy as np
from cbct_artifact_reduction import lakefs_own
from cbct_artifact_reduction.dataprocessing import single_nifti_to_numpy
from torch.utils.data import DataLoader

data_csv = os.path.join(cfg.ROOT_DIR, "data.csv")
random.seed(42)
np.random.seed(42)
N = 100
axeos_small = csvcreator.get_random_entries(
    data_csv,
    1,
    exclude_filled_radiation=True,
    scanner="axeos",
    mandible="1",
    implants="0",
    fov="small",
)
axeos_large = csvcreator.get_random_entries(
    data_csv,
    1,
    exclude_filled_radiation=True,
    scanner="axeos",
    mandible="1",
    implants="0",
    fov="large",
)
planmeca_small = csvcreator.get_random_entries(
    data_csv,
    1,
    exclude_filled_radiation=True,
    scanner="planmeca",
    mandible="1",
    implants="0",
    fov="small",
)
planmeca_large = csvcreator.get_random_entries(
    data_csv,
    1,
    exclude_filled_radiation=True,
    scanner="planmeca",
    mandible="1",
    implants="0",
    fov="large",
)
x800_small = csvcreator.get_random_entries(
    data_csv,
    1,
    exclude_filled_radiation=True,
    scanner="x800",
    mandible="1",
    implants="0",
    fov="small",
)
x800_large = csvcreator.get_random_entries(
    data_csv,
    1,
    exclude_filled_radiation=True,
    scanner="x800",
    mandible="1",
    implants="0",
    fov="large",
)
accuitomo_small = csvcreator.get_random_entries(
    data_csv,
    1,
    exclude_filled_radiation=True,
    scanner="accuitomo",
    mandible="1",
    implants="0",
    fov="small",
)
accuitomo_large = csvcreator.get_random_entries(
    data_csv,
    1,
    exclude_filled_radiation=True,
    scanner="accuitomo",
    mandible="1",
    implants="0",
    fov="large",
)

axeos_small_slices = csvcreator.get_slice_ids(
    data_csv, axeos_small["id"].tolist(), shuffle=True
)[0:N]
planmeca_small_slices = csvcreator.get_slice_ids(
    data_csv, planmeca_small["id"].tolist(), shuffle=True
)[0:N]
x800_small_slices = csvcreator.get_slice_ids(
    data_csv, x800_small["id"].tolist(), shuffle=True
)[0:N]
accuitomo_small_slices = csvcreator.get_slice_ids(
    data_csv, accuitomo_small["id"].tolist(), shuffle=True
)[0:N]

axeos_large_slices = csvcreator.get_slice_ids(
    data_csv, axeos_large["id"].tolist(), shuffle=True
)[0:N]
planmeca_large_slices = csvcreator.get_slice_ids(
    data_csv, planmeca_large["id"].tolist(), shuffle=True
)[0:N]
x800_large_slices = csvcreator.get_slice_ids(
    data_csv, x800_large["id"].tolist(), shuffle=True
)[0:N]
accuitomo_large_slices = csvcreator.get_slice_ids(
    data_csv, accuitomo_large["id"].tolist(), shuffle=True
)[0:N]
print(axeos_small_slices)


dict_of_slices = {
    "axeos_small": axeos_small_slices,
    "planmeca_small": planmeca_small_slices,
    "x800_small": x800_small_slices,
    "accuitomo_small": accuitomo_small_slices,
    "axeos_large": axeos_large_slices,
    "planmeca_large": planmeca_large_slices,
    "x800_large": x800_large_slices,
    "accuitomo_large": accuitomo_large_slices,
}
total_list = []
for k, v in dict_of_slices.items():
    total_list = total_list + v


client = lakefs_own.CustomBoto3Client(f"{cfg.LAKEFS_DATA_REPOSITORY}")


new_list = []
for slice in total_list:
    id = [int(slice.split("_")[0])]
    local_path = client.get_file(f"processed_data/frames/256x256/{slice}")

    np_array = single_nifti_to_numpy(local_path)
    mean = np.mean(np_array)
    if mean < 10 or np.any(np_array == 0):
        print(f"Slice {slice} is bad. Searching new slice.")

        while mean < 10 or np.any(np_array == 0):
            new_slice = csvcreator.get_slice_ids(data_csv, id, shuffle=True)[0]
            while new_slice in new_list:
                new_slice = csvcreator.get_slice_ids(data_csv, id, shuffle=True)[0]
            local_path = client.get_file(f"processed_data/frames/256x256/{new_slice}")
            np_array = single_nifti_to_numpy(local_path)
            mean = np.mean(np_array)
        print(f"Slice {new_slice} is good. Adding new slice.")
        new_list.append(new_slice)
    else:
        new_list.append(slice)


print(len(new_list))


def get_sampling_names():
    return new_list


client = lakefs_own.CustomBoto3Client(f"{cfg.LAKEFS_DATA_REPOSITORY}")
inpaintingSliceDataset = dataset.InpaintingSliceDataset(
    client,
    get_sampling_names(),
    "processed_data/frames/256x256",
    random_masks=False,
    augment_data=False,
    log_transform=True,
)

dataloader = DataLoader(inpaintingSliceDataset, batch_size=8, shuffle=False)

data = iter(dataloader)
