#from .network_lipschitz import get_lipschitz_constants
from .network_SIREN import DensityNetworkSIREN
from .network_SIREN import functional_forward_density_siren
from .network_SIREN import SirenINR
#from .network_SIREN import SirenMLP #julian


# def get_network(): #julian
#
#     return SirenMLP(input_dim=3, hidden_dim=256, output_dim=1, num_layers=5, omega=30) #SirenINR


def get_network(type):
    if type == "mlp":
        return SirenINR
    else:
        raise NotImplementedError("Unknown network typeß!")
