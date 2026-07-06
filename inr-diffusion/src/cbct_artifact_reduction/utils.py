import os
import random

import numpy as np
import pandas as pd
import torch
from typing import List, Optional

# Save project root directory
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
DATA_DIR = os.path.join(ROOT_DIR, "data")
CODE_DIR = os.path.join(ROOT_DIR, "code")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
FRAME_DIR = os.path.join(ROOT_DIR, "output", "frames")


def lookup_num_in_datatable(lookup_id: int):
    """Looks up a row in the data.csv file and returns it as a dictionary.

    Args:
        num (int): The id of the row to look up.

    Returns:
        dict: The row as a dictionary.
    """
    df = pd.read_csv(os.path.join(ROOT_DIR, "data.csv"))
    row = df[df["id"] == lookup_id]
    if row.empty:
        return None

    row_dict = row.iloc[0].to_dict()

    for key, value in row_dict.items():
        if isinstance(value, torch.Tensor):
            row_dict[key] = value.item()
        elif isinstance(value, tuple) and len(value) == 1:
            row_dict[key] = value[0]
        elif isinstance(value, float) and np.isnan(value):
            row_dict[key] = None

    return row_dict


def get_scanner_from_num(num: int):
    orig_num = num
    scanner = ""
    material = ""
    implants = ""
    fov = ""

    if ((num - 1) // 20) % 4 == 0:
        scanner = "axeos"
    elif ((num - 1) // 20) % 4 == 1:
        scanner = "accuitomo"
    elif ((num - 1) // 20) % 4 == 2:
        scanner = "planmeca"
    elif ((num - 1) // 20) % 4 == 3:
        scanner = "x800"

    if num % 10 == 0:
        material = ""
        implants = 0
    elif (num % 10) % 3 == 1:
        implants = 3
    elif (num % 10) % 3 == 2:
        implants = 2
    elif (num % 10) % 3 == 0:
        implants = 1

    if ((num - 1) % 10) // 3 == 0:
        material = "ti"
    elif ((num - 1) % 10) // 3 == 1:
        material = "tizr"
    elif ((num - 1) % 10) // 3 == 2:
        material = "zr"

    if ((num - 1) // 10) % 2 == 0:
        fov = "small"
    elif ((num - 1) // 10) % 2 == 1:
        fov = "large"

    return orig_num, scanner, material, implants, fov


def generate_random_integers(N, A, B):
    """Generates N random integers between A and B (inclusive).

    Args:
        N (int): Number of random integers.
        A (int): Lower bound.
        B (int): Upper bound.

    Returns:
        list: List of N random integers between A and B.
    """
    return [random.randint(A, B) for _ in range(N)]


def getAllControlIDs(exludeIDs: Optional[List[int]] = [41, 208]) -> List[str]:
    """Find all scan ID's that correspond to control images without implants in the CBCT pig jaw data.

    The CBCT pig jaw dataset consists of 398 scans. The scans with the ID's 41 and 208 are missing.

    Args:
        exludeIDs (list[int], optional): List of scan ID's to exclude. Defaults to missing ID's of CBCT pig jaw data.
    Returns:
        list[str]: List of scan ID's that correspond to control images without implants.
    """

    if exludeIDs is None:
        possibleIDs = [f"{f}" for f in range(1, 401)]
    else:
        possibleIDs = [f"{f}" for f in range(1, 401) if f not in exludeIDs]

    controlIDs: list[str] = []
    for id in possibleIDs:
        scanner = get_scanner_from_num(int(id))
        if scanner[3] == 0:
            controlIDs.append(id)

    return controlIDs


def getAllAxeosIDs(exludeIDs: Optional[List[int]] = [41, 208]) -> List[str]:
    """Find all scan ID's that correspond to Axeos images in the CBCT pig jaw data.

    The CBCT pig jaw dataset consists of 398 scans. The scans with the ID's 41 and 208 are missing.

    Args:
        exludeIDs (list[int], optional): List of scan ID's to exclude. Defaults to missing ID's of CBCT pig jaw data.
    Returns:
        list[str]: List of scan ID's that correspond to Axeos images.
    """

    if exludeIDs is None:
        possibleIDs = [f"{f}" for f in range(1, 401)]
    else:
        possibleIDs = [f"{f}" for f in range(1, 401) if f not in exludeIDs]

    axeosIDs: list[str] = []
    for id in possibleIDs:
        scanner = get_scanner_from_num(int(id))
        if scanner[1] == "axeos":
            axeosIDs.append(id)

    return axeosIDs


def getAllAccuitomoIDs(exludeIDs: Optional[List[int]] = [41, 208]) -> List[str]:
    """Find all scan ID's that correspond to Accuitomo images in the CBCT pig jaw data.

    The CBCT pig jaw dataset consists of 398 scans. The scans with the ID's 41 and 208 are missing.

    Args:
        exludeIDs (list[int], optional): List of scan ID's to exclude. Defaults to missing ID's of CBCT pig jaw data.
    Returns:
        list[str]: List of scan ID's that correspond to Accuitomo images.
    """

    if exludeIDs is None:
        possibleIDs = [f"{f}" for f in range(1, 401)]
    else:
        possibleIDs = [f"{f}" for f in range(1, 401) if f not in exludeIDs]

    accuitomoIDs: list[str] = []
    for id in possibleIDs:
        scanner = get_scanner_from_num(int(id))
        if scanner[1] == "accuitomo":
            accuitomoIDs.append(id)

    return accuitomoIDs


def getAllplanmecaIDs(exludeIDs: Optional[List[int]] = [41, 208]) -> List[str]:
    """Find all scan ID's that correspond to Planmeca images in the CBCT pig jaw data.

    The CBCT pig jaw dataset consists of 398 scans. The scans with the ID's 41 and 208 are missing.

    Args:
        exludeIDs (list[int], optional): List of scan ID's to exclude. Defaults to missing ID's of CBCT pig jaw data.
    Returns:
        list[str]: List of scan ID's that correspond to Planmeca images.
    """

    if exludeIDs is None:
        possibleIDs = [f"{f}" for f in range(1, 401)]
    else:
        possibleIDs = [f"{f}" for f in range(1, 401) if f not in exludeIDs]

    planmecaIDs: list[str] = []
    for id in possibleIDs:
        scanner = get_scanner_from_num(int(id))
        if scanner[1] == "planmeca":
            planmecaIDs.append(id)

    return planmecaIDs


def getAllx800IDs(exludeIDs: Optional[List[int]] = [41, 208]) -> List[str]:
    """Find all scan ID's that correspond to x800 images in the CBCT pig jaw data.

    The CBCT pig jaw dataset consists of 398 scans. The scans with the ID's 41 and 208 are missing.

    Args:
        exludeIDs (list[int], optional): List of scan ID's to exclude. Defaults to missing ID's of CBCT pig jaw data.
    Returns:
        list[str]: List of scan ID's that correspond to x800 images.
    """

    if exludeIDs is None:
        possibleIDs = [f"{f}" for f in range(1, 401)]
    else:
        possibleIDs = [f"{f}" for f in range(1, 401) if f not in exludeIDs]

    x800IDs: list[str] = []
    for id in possibleIDs:
        scanner = get_scanner_from_num(int(id))
        if scanner[1] == "x800":
            x800IDs.append(id)

    return x800IDs
