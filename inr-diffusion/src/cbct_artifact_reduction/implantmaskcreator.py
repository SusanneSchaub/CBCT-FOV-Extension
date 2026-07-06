import hashlib
from math import floor

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import scipy.stats as stats


def generateRandomHeight(h_loc, h_scale, random_state=None):
    h = stats.norm.rvs(loc=h_loc, scale=h_scale, random_state=random_state)
    height = abs(floor(h))
    if height == 0:
        return 1
    else:
        return height


def generateRandomWidth(w_loc, w_scale, random_state=None):
    w = stats.norm.rvs(loc=w_loc, scale=w_scale, random_state=random_state)
    w = abs(floor(w))

    if w == 0:
        return 1
    else:
        return w


def generateRotationAngle(r_loc, r_scale, random_state=None):
    r = stats.norm.rvs(loc=r_loc, scale=r_scale, random_state=random_state)
    r = floor(r)
    if r == 0:
        return 0
    else:
        return r


def generateCoordinates(x_loc, x_scale, y_loc, y_scale, random_state=None):
    x = stats.uniform.rvs(loc=x_loc, scale=x_scale, random_state=random_state)
    #y = stats.norm.rvs(loc=y_loc, scale=y_scale, random_state=random_state)
    y = stats.uniform.rvs(loc=x_loc, scale=x_scale, random_state=random_state)
    #x=0
    #y=0
    return abs(floor(x)), abs(floor(y))


class ImplantMaskCreator:
    """Create random masks for implant regions."""

    def __init__(self, resolution: tuple[int, int]) -> None:
        self.resolution = resolution

    def generate_mask(self) -> np.ndarray:

        mask = np.ones(self.resolution, dtype=np.uint8)

        mask[:,100:self.resolution[0]-100] = 0

        return mask

    def generate_mask_with_n_implants(self, n: int, random_state=None) -> np.ndarray:
        mask = np.zeros(self.resolution, dtype=int)
        for _ in range(n):
            if random_state is not None:
                random_state = int.from_bytes(
                    hashlib.sha256(str(random_state).encode("utf-8")).digest()[:4],
                    "little",
                )

            implant = self.generate_mask(random_state=random_state)
            mask = mask + implant

        mask = np.clip(mask, 0, 1)
        return mask

    def generate_mask_with_random_amount_of_implants(
        self, lower, upper, random_state=None
    ) -> np.ndarray:
        """Generate a mask with a random amount of implants.

        Args:
            lower (int): The lower bound for the number of implants.
            upper (int): The upper bound for the number of implants.

        Returns:
            np.ndarray: The generated mask.
        """
        np.random.seed(random_state)
        n = np.random.randint(lower, upper)
        #n=1
        return self.generate_mask_with_n_implants(n, random_state=random_state)


if __name__ == "__main__":
    creator = ImplantMaskCreator((256, 256))
    mask = creator.generate_mask_with_n_implants(2)
    plt.imshow(mask)
    plt.show()
