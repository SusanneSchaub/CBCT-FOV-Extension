import pdb

import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Sigmoid
import torch.nn.functional as F
import random
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


class DensityNetwork(nn.Module):
    def __init__(self, encoder, bound=0.2, num_layers=8, hidden_dim=256, skips=[4], out_dim=1, last_activation="sigmoid", first_omega_0=0, hidden_omega_0=0, use_siren=False, idx_epoch=100):
        super().__init__()

        self.nunm_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.encoder = encoder
        ##======================== NAF+ ========================##
        #self.in_dim = encoder.output_dim + 16
        ##======================== NAF+ ========================##
        self.in_dim = encoder.output_dim
        self.bound = bound
        self.idx_epoch = idx_epoch

        ##======================== NAF+ ========================##
        #self.ln = nn.LayerNorm((self.in_dim-16))
        ##======================== NAF+ ========================##

        self.ln = nn.LayerNorm((self.in_dim))

        self.embed = nn.Linear(1, 16)

        # Linear layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) if i not in skips
                                                    else nn.Linear(hidden_dim + self.in_dim, hidden_dim) for i in
                                                    range(1, num_layers - 1, 1)])
        self.layers.append(nn.Linear(hidden_dim, out_dim))

        # Activations
        #self.activations = nn.ModuleList([nn.LeakyReLU() for i in range(0, num_layers - 1, 1)])
        self.activations = nn.ModuleList([nn.SiLU() for i in range(0, num_layers - 1, 1)]) #change from relu to swish
        if last_activation == "sigmoid":
            self.activations.append(nn.Sigmoid())
        elif last_activation == "relu":
            self.activations.append(nn.LeakyReLU())
        else:
            raise NotImplementedError("Unknown last activation")



    def forward(self, idx_epoch,x, mask=1, feature_vector=None):

        ##======================== NAF+ ========================##
        #pos = x[..., :3]
        #rho = x[..., -1].unsqueeze(-1)
        #rho = self.embed(rho)
        ##======================== NAF+ ========================##
        pos = x

        if feature_vector != None:
            encoding_result = feature_vector
        else:
            encoding_result = self.encoder(pos, self.bound)

        if self.encoder.encode_type == 'hash':
            x = self.ln(encoding_result) * mask
        else:
            x= encoding_result

        ##======================== NAF+ ========================##
        #x = torch.cat((x, rho), dim=-1)
        ##======================== NAF+ ========================##


        regularization = True

        if regularization == True:
            x = x.view(x.shape[0], 20, 4)
            T = 5
            s = 1
            for i in range(0,x.shape[1]):

                r = np.abs(idx_epoch/T) + s
                if i <= r:
                    m = 1
                else:
                    m = 0

                x[:,i,:] = x[:,i,:]*m

            x = x.view(x.shape[0], x.shape[1]*x.shape[2])

        input_pts = x[..., :self.in_dim]

        for i in range(len(self.layers)):

            linear = self.layers[i]
            activation = self.activations[i]

            if i in self.skips:
                x = torch.cat([input_pts, x], -1)

            x = linear(x)
            x = activation(x)

        return x



def functional_forward_density(
    model,
    params,
    encoder,
    pts,
    conf,
    mask=1,
    ):
    """
    Functional forward pass for DensityNetwork.

    Args:
        model: DensityNetwork instance (structure only)
        idx_epoch: epoch index (kept for signature compatibility)
        x: input tensor
        params: dict of parameters (state_dict-like)
        mask: mask multiplier
        feature_vector: optional precomputed encoder output

    Returns:
        Output tensor
    """

    # ----- LayerNorm -----

    cnf = conf["encoder"]["encoding"]
    if cnf == 'hashgrid':
        encoding_result = functional_hash_encoder(
            model.encoder,
            pts,
            params,
            size=model.bound)
    elif cnf == 'frequency':
        encoding_result = encoder(pts, model.bound)

    ln_weight = params["ln.weight"]
    ln_bias = params["ln.bias"]

    x = F.layer_norm(
        encoding_result,
        normalized_shape=(model.in_dim,),
        weight=ln_weight,
        bias=ln_bias,
    )
    x = x * mask

    input_pts = x[..., :model.in_dim]

    # ----- Main MLP -----
    for i in range(len(model.layers)):
        if i in model.skips:
            x = torch.cat([input_pts, x], dim=-1)

        # Linear
        w = params[f"layers.{i}.weight"]
        b = params[f"layers.{i}.bias"]
        x = F.linear(x, w, b)

        # Activation
        act = model.activations[i]
        if isinstance(act, torch.nn.LeakyReLU):
            x = F.leaky_relu(x, negative_slope=act.negative_slope)
        elif isinstance(act, torch.nn.Sigmoid):
            x = torch.sigmoid(x)
        else:
            raise NotImplementedError(f"Unsupported activation {type(act)}")

    return x
