import os

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from cbct_artifact_reduction.dataprocessing import single_nifti_to_numpy
from cbct_artifact_reduction.utils import ROOT_DIR

samples_dir = os.path.join(ROOT_DIR, "samples")
# Get all names of the samples in sample directory
sample_names = [f for f in os.listdir(samples_dir) if f.endswith(".nii.gz")]
# Get the first sample

sample_name = sample_names[2]
image = single_nifti_to_numpy(os.path.join(samples_dir, sample_name))


num_timesteps = 1000
# Linear schedule from Ho et al, extended to work for any number of
# diffusion steps.
scale = 1000 / num_timesteps
beta_start = scale * 0.0001
beta_end = scale * 0.02
betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
num_timesteps = int(betas.shape[0])

alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas, axis=0)
alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.0)
assert alphas_cumprod_prev.shape == (num_timesteps,)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
log_one_minus_alphas_cumprod = np.log(1.0 - alphas_cumprod)
sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def q_sample(x_start, t, noise=None):
    """
    Diffuse the data for a given number of diffusion steps.

    In other words, sample from q(x_t | x_0).

    :param x_start: the initial data batch.
    :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
    :param noise: if specified, the split-out normal noise.
    :return: A noisy version of x_start.
    """
    if noise is None:
        noise = th.randn_like(x_start)
    assert noise.shape == x_start.shape
    return (
        _extract_into_tensor(sqrt_alphas_cumprod, t, x_start.shape) * x_start
        + _extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    )


image = th.from_numpy(single_nifti_to_numpy(os.path.join(samples_dir, sample_name)))
print(image[None, 1:2].shape)


for i in range(1000):
    t = th.tensor([i] * 1)

    if i % 50 == 0:
        noisy_sample = q_sample(image[None, 1:2, ...], t)
        plt.imshow(noisy_sample[0, 0, :, :], cmap="bone")
        plt.show()
