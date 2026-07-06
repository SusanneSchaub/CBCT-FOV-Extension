from .network import DensityNetwork
from .network_lipschitz import get_lipschitz_constants
from .network import functional_forward_density

def get_network(type):
    if type == "mlp":
        return DensityNetwork
    else:
        raise NotImplementedError("Unknown network typeß!")

