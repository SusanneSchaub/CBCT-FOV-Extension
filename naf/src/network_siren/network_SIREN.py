import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Sigmoid
import torch.nn.functional as F
import random
import math
import numpy as np
from src.encoder.hashencoder import functional_hash_encoder

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()


def squareplus(x, b):
    return torch.mul(0.5, torch.add(x, torch.sqrt(torch.add(torch.square(x), b))))


class SquarePlus(nn.Module):
    def __init__(self, b=1.52382103):
        super().__init__()
        self.b = b

    def forward(self, x):
        return squareplus(x, self.b)


class mean(nn.Module):
    def __init__(self, out_size, encoder, hidden_dim):
        super(mean, self).__init__()
        self.encoder = encoder
        self.ln = nn.LayerNorm((20))

    def forward(self, x,i=None):
        encoding_result = self.encoder(x).float()
        if i != None:
            output = torch.mean(encoding_result[:,i].unsqueeze(-1),dim=-1)
        else:
            output = encoding_result
            output = self.ln(output)
        return output


def siren_init(layer, omega_0=30.0, is_first=False):
    with torch.no_grad():
        in_dim = layer.weight.size(-1)
        if is_first:
            layer.weight.uniform_(-1 / in_dim, 1 / in_dim)
        else:
            bound = math.sqrt(6 / in_dim) / omega_0
            layer.weight.uniform_(-bound, bound)


class SineLayer(nn.Module):
    '''
        See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for
        discussion of omega_0.

        If is_first=True, omega_0 is a frequency factor which simply multiplies
        the activations before the nonlinearity. Different signals may require
        different omega_0 in the first layer - this is a hyperparameter.

        If is_first=False, then the weights will be divided by omega_0 so as to
        keep the magnitude of activations constant, but boost gradients to the
        weight matrix (see supplement Sec. 1.5)
    '''

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, scale=10.0, init_weights=True):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if init_weights:
            self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))



class DensityNetworkSIREN(nn.Module):
    def __init__(
        self,
        in_features=3,
        hidden_dim=512,
        hidden_layers=3,
        out_dim=1,
        first_omega_0=30.0,
        hidden_omega_0=30.0,
        outermost_linear=True,
        bound=0.3,

    ):
        super().__init__()
        self.net_w_layers = []

        # ---------- First SIREN layer ----------
        self.net_w_layers.append(
            SineLayer(
                in_features,
                hidden_dim,
                is_first=True,
                omega_0=first_omega_0,
            )
        )

        # ---------- Hidden SIREN layers ----------
        for _ in range(hidden_layers):
            self.net_w_layers.append(
                SineLayer(
                    hidden_dim,
                    hidden_dim,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        # ---------- Output layer ----------
        if outermost_linear:
            final_linear = nn.Linear(hidden_dim, out_dim)
            with torch.no_grad():
                const = np.sqrt(6 / hidden_dim) / max(hidden_omega_0, 1e-12)
                final_linear.weight.uniform_(-const, const)
            self.net_w_layers.append(final_linear)
        else:
            self.net_w_layers.append(
                SineLayer(
                    hidden_dim,
                    out_dim,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*self.net_w_layers)

    def forward(self, coords):
        """
        coords: (..., 3) raw xyz
        """
        breakpoint()
        x = self.net(coords)
        x = torch.sigmoid(x)
        return x

    def forward_w_features(self, coords, bound):
        """
        Returns:
            output: final layer output
            features: list of activations after each layer
        """
        features = []
        x = coords.clone()
        x=x/bound

        for layer in self.net_w_layers:
            x = layer(x)
            features.append(x)

        #output = x.clone()
        output = torch.sigmoid(x.clone())
        return output, features

    def load_weights(self, weights):
        self.load_state_dict(weights)


def functional_forward_density_siren(
    model,
    params,
    coords,
    mask,
    bound,
):
    """
    Functional forward pass for DensityNetworkSIREN.

    Args:
        model: DensityNetworkSIREN instance (structure only)
        params: state_dict-like dict of parameters
        coords: (..., 3) raw xyz coordinates

    Returns:
        output tensor
    """

    x = coords / bound #S: divided by bound to normalise to [-1,1]

    layer_idx = 0
    param_idx = 0

    for layer in model.net_w_layers:

        # ---------- SineLayer ----------
        if isinstance(layer, SineLayer):
            w = params[f"net_w_layers.{param_idx}.linear.weight"]
            b = params[f"net_w_layers.{param_idx}.linear.bias"]

            x = F.linear(x, w, b)
            x = torch.sin(layer.omega_0 * x)

        # ---------- Final Linear ----------
        elif isinstance(layer, torch.nn.Linear):
            w = params[f"net_w_layers.{param_idx}.weight"]
            b = params[f"net_w_layers.{param_idx}.bias"]
            x = F.linear(x, w, b)

        else:
            raise NotImplementedError(f"Unsupported layer type {type(layer)}")

        param_idx += 1

    #S: add sigmoid:
    x = torch.sigmoid(x)
    return x

class SirenINR(nn.Module):
    def __init__(self, inr_type, inr_config):
        super().__init__()

        self.bound = inr_config["bound"]
        self.inr = DensityNetworkSIREN(**inr_config)
        self.normalize_features = False

        #print(self.inr)

    def forward(self, coords):

        inr_output, inr_features = self.inr.forward_w_features(coords, self.bound)

        return inr_output


    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_weights(self):
        return self.state_dict()