import os
import re

import cbct_artifact_reduction.utils as utils

DATA_DIR = utils.DATA_DIR
ROOT_DIR = utils.ROOT_DIR


def sort_by_first_number_in_name(name: str):
    match = re.search(r"\d+", name)
    if match:
        return int(match.group(0))
    else:
        return 0


def sort_by_scanner(name: str):
    name = name.lower()
    default_order = {"axeos": 0, "accuitomo": 1, "planmeca": 2, "x800": 3}

    for item in default_order:
        print(item)
        if item in name:
            return default_order[item]
    return float("inf")


def sort_by_fov_implants_material(name: str):
    name = name.lower()
    fov_order = {"small": 0, "large": 1}
    implant_order = {"4": 0, "2": 1, "0": 2}
    element_order = {"ti": 0, "zro2": 1, "zr": 2}

    match


def get_latest_id_from_csv(csv_path: str):
    """Get the latest id from a csv file."""
    with open(csv_path, "r") as f:
        lines = f.readlines()
        latest_id = lines[-1].split(",")[0]
        return latest_id


def extract_info_from_subfolder(subfolder: str):
    subfolder = subfolder.lower()
    fov = None
    material = None
    implants = None
    if subfolder.__contains__("large"):
        fov = "large"
    elif subfolder.__contains__("sll"):
        fov = "sll"

    if subfolder.__contains__(" ti "):
        material = "ti"
    elif (
        subfolder.__contains__("zr02")
        or subfolder.__contains__("zro2")
        or subfolder.__contains__("zrO2")
    ):
        material = "zro2"
    elif subfolder.__contains__("tizr"):
        material = "tizr"
    else:
        material = ""

    if subfolder.__contains__("0"):
        implants = 0
    elif subfolder.__contains__("2"):
        implants = 2
    elif subfolder.__contains__("4"):
        implants = 4

    return fov, material, implants


def process_axeos_subfolder(folder_path: str):
    """Process a folder containing multiple Axeos scan subfolders. The subfolders have relevant information in their names like
    large, sll, ti, zr02, etc.
    """

    subfolders = [
        f
        for f in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, f))
    ]

    # Get folder called axeos projection output
    axeos_projection_output_folder = None
    for subfolder in subfolders:
        if subfolder.__contains__("output"):
            axeos_projection_output_folder = subfolder

    if axeos_projection_output_folder is None:
        raise ValueError("No output folder found in Axeos subfolder")

    # Get list of subfolders in axeos projection output folder
    axeos_projection_subfolders = [
        f
        for f in os.listdir(os.path.join(folder_path, axeos_projection_output_folder))
        if os.path.isdir(os.path.join(folder_path, axeos_projection_output_folder, f))
    ]

    # axeos_projection_subfolder should contain only one folder
    if len(axeos_projection_subfolders) != 1:
        raise ValueError("More than one folder found in Axeos projection output folder")

    axeos_projection_subfolder = axeos_projection_subfolders[0]

    # Get list of subfolders in axeos projection subfolder
    axeos_projection_subfolder_subfolders = [
        f for f in os.listdir(axeos_projection_subfolder) if os.path.isdir(f)
    ]

    for subfolder in axeos_projection_subfolder_subfolders:
        fov, material, implants = extract_info_from_subfolder(subfolder)


def process_pigjaw_folder(folder_path: str):
    """Process a folder containing scans from multiple scanners of a single pigjaw mandible."""
    subfolders = [
        f
        for f in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, f))
    ]

    for subfolder in subfolders:
        if subfolder.__contains__("axeos"):
            pass


if __name__ == "__main__":
    # Get list of folders in DATA_DIR
    folders = sorted(
        [
            os.path.join(DATA_DIR, f)
            for f in os.listdir(DATA_DIR)
            if os.path.isdir(os.path.join(DATA_DIR, f))
        ]
    )

    pigjaw_folder = folders[0]

    DATA_CSV = os.path.join(ROOT_DIR, "data.csv")
    latest_id = get_latest_id_from_csv(DATA_CSV)
    sort_by_scanner("x800")
