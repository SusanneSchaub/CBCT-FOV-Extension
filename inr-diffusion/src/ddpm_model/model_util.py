import subprocess
from io import BytesIO

import pandas as pd
import torch as th
from torch.utils.data import DataLoader

import cbct_artifact_reduction.config as cfg
import cbct_artifact_reduction.pigjawdataset as dataset
from cbct_artifact_reduction import lakefs_own
from cbct_artifact_reduction.guided_diffusion import dist_util
from cbct_artifact_reduction.guided_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def get_free_gpu():
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]
    )
    gpu_df = pd.read_csv(
        BytesIO(gpu_stats), names=["memory.used", "memory.free"], skiprows=1
    )
    print("GPU usage:\n{}".format(gpu_df))
    gpu_df["memory.free"] = gpu_df["memory.free"].map(lambda x: x.rstrip(" [MiB]"))
    gpu_df["memory.free"] = pd.to_numeric(gpu_df["memory.free"])
    idx = gpu_df["memory.free"].idxmax()
    print(
        "Returning GPU{} with {} free MiB".format(idx, gpu_df.iloc[idx]["memory.free"])
    )
    return idx


def load_model(model_path, device=None, sample=False, **kwargs):
    """Load diffusion model. By default, load to GPU with most available virtual memory.

    Args:
        model_path (str): path to model
        device (str): device to load model on
        sample (bool): whether to sample from model
        kwargs: arguments to pass to create_model_and_diffusion

    Returns:
        model (nn.Module): The loaded model.
        diffusion (nn.Module): The loaded diffusion model.
    """
    if device is None:
        if th.cuda.is_available():
            device = th.device(f"cuda:{get_free_gpu()}")
        else:
            device = th.device("cpu")
    # dist_util.setup_dist()
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(kwargs, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(model_path, map_location=device))
    if sample:
        model.eval()

    return model, diffusion


def create_dataloader(
    filenames,
    random_masks: bool,
    lakefs_folder: str,
    batch_size: int,
    shuffle: bool,
    augment_data: bool,
):
    client = lakefs_own.CustomBoto3Client(f"{cfg.LAKEFS_DATA_REPOSITORY}")
    inpaintingSliceDataset = dataset.InpaintingSliceDataset(
        client,
        filenames,
        lakefs_folder,
        random_masks=random_masks,
        augment_data=augment_data,
    )

    dataloader = DataLoader(
        inpaintingSliceDataset, batch_size=batch_size, shuffle=shuffle
    )

    data = iter(dataloader)

    return data
