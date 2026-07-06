import os

import cbct_artifact_reduction.config as cfg
import cbct_artifact_reduction.lakefs_own as lakefs_own
import cbct_artifact_reduction.pigjawdataset as dataset
from torch.utils.data import DataLoader

client = lakefs_own.CustomBoto3Client(f"{cfg.LAKEFS_DATA_REPOSITORY}")
inpaintingSliceDataset = dataset.InpaintingSliceDataset(
    client,
    os.path.join(cfg.ROOT_DIR, "training_data.csv"),
    "processed_data/frames/256x256",
    "processed_data/masks/256x256",
)

dataloader = DataLoader(inpaintingSliceDataset, batch_size=10, shuffle=True)
data = iter(dataloader)
print(next(data)[1].shape)
