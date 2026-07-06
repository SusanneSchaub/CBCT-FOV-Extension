import csv
import os
import re

import cbct_artifact_reduction.utils as utils
import numpy as np
import tifffile
from cbct_artifact_reduction.dataprocessing import numpy_to_nifti
from cbct_artifact_reduction.scanner import planmeca_folder_to_numpy

DATA_DIR = utils.DATA_DIR


def get_latest_id_from_csv(csv_path: str):
    """Get the latest id from a csv file."""
    with open(csv_path, "r") as f:
        lines = f.readlines()
        latest_id = lines[-1].split(",")[0]
        return int(latest_id)


def get_data_filepaths(data_dir: str):
    data_filepaths = []
    for dirpath, _, filenames in os.walk(data_dir):
        original_dirpath = dirpath
        dirpath = dirpath.lower()

        if dirpath.__contains__("dicom"):
            continue

        for filename in filenames:
            if filename.__contains__(".tif") or filename.__contains__(".tar.gz"):
                if filename.__contains__(".Xed"):
                    continue
                file_path = os.path.join(original_dirpath, filename)
                data_filepaths.append(file_path)

    return data_filepaths


def process_filepaths(data_filepaths: list[str]):
    filepaths_info = []
    for filepath in data_filepaths:
        filepaths_info.append(process_filepath(filepath))
    return filepaths_info


def process_filepath(filepath: str):
    scanner = None
    mandible = None
    material = None
    implants = None
    fov = None
    radiation = ""
    original_filepath = filepath
    filepath = filepath.lower()

    mandible_match = re.search(r"pigjaw.?(\d+)", filepath)
    if mandible_match:
        mandible = int(mandible_match.group(1))

    scanner_match = re.search(r"(axeos|aquitomo|accuitomo|planmeca|x800)", filepath)
    if scanner_match:
        scanner = scanner_match.group(1)
        if scanner == "aquitomo":
            scanner = "accuitomo"

    material_match = re.search(r"(\btizr\b|\bti\b|\bzro2\b|\bzr\b|\bzr02\b)", filepath)
    if material_match:
        material = material_match.group(0)
        if material == "zr02":
            material = "zro2"

    fov_match = re.search(r"(\bsll\b|\blarge\b)", filepath)
    if fov_match:
        fov = fov_match.group(1)
        if fov == "sll":
            fov = "small"

    implants_match = re.search(
        r"(?:\bsll\b|\blarge\b|\btizr\b|\bti\b|\bzro2\b|zr02\b|\bzr\b|\bfov\b) (\d+)",
        filepath,
    )
    if implants_match:
        implants = int(implants_match.group(1))

    if implants == 0:
        material = ""

    radiation_match = re.search(r"(\bstandard\b|\bultralow\b)", filepath)
    if radiation_match:
        radiation = radiation_match.group(1)

    if any(
        variable is None
        for variable in [scanner, mandible, material, implants, fov, radiation]
    ):
        print("Error: Could not parse all information from filepath.")
        print(f"filepath in question: {filepath}")

    info_dict = {
        "filepath": original_filepath,
        "scanner": scanner,
        "mandible": mandible,
        "material": material,
        "implants": implants,
        "fov": fov,
        "height": None,
        "width": None,
        "frames": None,
        "radiation": radiation,
    }
    return info_dict


def tif_to_numpy(tif_path: str):
    tif_data = tifffile.imread(tif_path)
    numpy_array = np.array(tif_data)
    return numpy_array


if __name__ == "__main__":
    data_output_dir = os.path.join(utils.OUTPUT_DIR, "raw_nifti_data")
    data_filepaths = get_data_filepaths(DATA_DIR)

    data_filepaths_info = process_filepaths(data_filepaths)

    for idx, item in enumerate(data_filepaths_info):
        data_csv_path = os.path.join(utils.ROOT_DIR, "data.csv")
        id = get_latest_id_from_csv(data_csv_path) + 1
        if idx <= 146:
            continue

        if os.path.exists(os.path.join(data_output_dir, f"{id}.nii.gz")):
            print(f"Skipping {id}.nii.gz as it already exists.")
            continue

        numpy_array = None
        if item["scanner"] == "planmeca":
            numpy_array = planmeca_folder_to_numpy(item["filepath"])
        else:
            numpy_array = tif_to_numpy(item["filepath"])
            numpy_array = np.swapaxes(numpy_array, 0, 2)
            numpy_array = np.swapaxes(numpy_array, 0, 1)

        if numpy_array is not None:
            item["height"] = numpy_array.shape[0]
            item["width"] = numpy_array.shape[1]
            item["frames"] = numpy_array.shape[2]
        else:
            print(f"Could not process {item['filepath']}")
            continue

        with open(data_csv_path, "a") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    id,
                    item["scanner"],
                    item["mandible"],
                    item["material"],
                    item["implants"],
                    item["fov"],
                    item["height"],
                    item["width"],
                    item["frames"],
                    item["radiation"],
                ]
            )

        numpy_to_nifti(numpy_array, os.path.join(data_output_dir, f"{id}.nii.gz"))
        print(f"saved {id}.nii.gz. {idx+1}/{len(data_filepaths_info)}.")
