"""
Helpers for distributed training.
"""

import io
import os
import socket
import subprocess
from io import BytesIO

import blobfile as bf
import pandas as pd
import torch as th
import torch.distributed as dist
#from mpi4py import MPI

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


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


if th.cuda.is_available():
    free_gpu_id = get_free_gpu()

#
# def setup_dist():
#     """
#     Setup a distributed process group.
#     """
#     if dist.is_initialized():
#         return
#     # Comment this because it lead to error on training resume.
#     # os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"
#
#     comm = MPI.COMM_WORLD
#     backend = "gloo" if not th.cuda.is_available() else "nccl"
#
#     if backend == "gloo":
#         hostname = "localhost"
#     else:
#         hostname = socket.gethostbyname(socket.getfqdn())
#     os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
#     os.environ["RANK"] = str(comm.rank)
#     os.environ["WORLD_SIZE"] = str(comm.size)
#
#     port = comm.bcast(_find_free_port(), root=0)
#     os.environ["MASTER_PORT"] = str(port)
#     dist.init_process_group(backend=backend, init_method="env://")

def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = '127.0.1.1'#comm.bcast(hostname, root=0)
    os.environ["RANK"] = '0'#str(comm.rank)
    os.environ["WORLD_SIZE"] = '1'#str(comm.size)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


# def dev():
#     """
#     Get the device to use for torch.distributed.
#     """
#     if th.cuda.is_available():
#         return th.device(f"cuda:{free_gpu_id}")
#     return th.device("cpu")

def dev():
    if th.cuda.is_available():
        th.cuda.set_device(free_gpu_id)
        return th.device(f"cuda:{free_gpu_id}")
    return th.device("cpu")


# def load_state_dict(path, **kwargs):
#     """
#     Load a PyTorch file without redundant fetches across MPI ranks.
#     """
#     chunk_size = 2**30  # MPI has a relatively small size limit
#     if MPI.COMM_WORLD.Get_rank() == 0:
#         with bf.BlobFile(path, "rb") as f:
#             data = f.read()
#         num_chunks = len(data) // chunk_size
#         if len(data) % chunk_size:
#             num_chunks += 1
#         MPI.COMM_WORLD.bcast(num_chunks)
#         for i in range(0, len(data), chunk_size):
#             MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
#     else:
#         num_chunks = MPI.COMM_WORLD.bcast(None)
#         data = bytes()
#         for _ in range(num_chunks):
#             data += MPI.COMM_WORLD.bcast(None)
#
#     return th.load(io.BytesIO(data), **kwargs)

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    mpigetrank=0
    if mpigetrank==0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """

    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
