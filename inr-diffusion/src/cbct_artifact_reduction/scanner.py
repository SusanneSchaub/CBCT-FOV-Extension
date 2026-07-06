import os
import shutil
import tempfile

import numpy as np

from cbct_artifact_reduction.dataprocessing import (
    extract_tar_gz,
    get_filename,
    tif_to_nifti,
)


def planmeca_folder_to_numpy(file_path: str):
    """Convert a planmeca folder to a numpy array.

    Args:
        file_path (str): The path to the folder containing the data. Can be a tar.gz file or a folder.

    Returns:
        numpy_array (np.ndarray): The numpy array containing the data. The shape of the array is (height, width, num_frames).
    """
    filename = get_filename(file_path)
    temp_dir = tempfile.mkdtemp()
    if filename.endswith(".tar.gz"):
        extract_tar_gz(file_path, temp_dir)
    else:
        shutil.copy(file_path, temp_dir)

    corrected_raw_dir = None
    fdk_3DII_conf_path = None
    for dirpath, _, filenames in os.walk(temp_dir):
        for filename in filenames:
            if filename == "fdk_3DII.conf":
                fdk_3DII_conf_path = os.path.join(dirpath, filename)

        if "corrected" in dirpath:
            corrected_raw_dir = dirpath

    if corrected_raw_dir is None:
        raise ValueError("No corrected raw directory found in Planmeca file")
    if fdk_3DII_conf_path is None:
        raise ValueError("No fdk_3DII.conf file found in Planmeca file")

    numbers = []
    with open(fdk_3DII_conf_path, "r") as f:
        fdk_3DII_conf = f.read()
        fdk_3DII_conf = fdk_3DII_conf.split("\n")
        for line in fdk_3DII_conf:
            if "nRec" in line:
                values = line.split("=")[1]
                numbers = [int(x.strip()) for x in values.split(",")]

    if len(numbers) != 2:
        raise ValueError("Could not find scan resolution in fdk_3DII.conf file")

    width, height = numbers

    extracted_frames = sorted(
        [f for f in os.listdir(corrected_raw_dir) if f.endswith(".raw")]
    )
    num_frames = len(extracted_frames)

    vol = np.zeros((height, width, num_frames))

    for index, frame in enumerate(extracted_frames):
        data = np.fromfile(
            os.path.join(corrected_raw_dir, frame),
            dtype=np.uint16,
        )
        data = np.reshape(data, (height, width))
        data = np.flip(data, axis=0)
        vol[..., index] = data

    return vol


def accuitomo_folder_to_nifti(folder_path: str, output_path: str):
    tif_path = None
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename == "Capture.tif":
                tif_path = os.path.join(dirpath, filename)

    if tif_path is None:
        raise ValueError("No Capture.tif file found in Accuitomo folder")

    tif_to_nifti(tif_path, os.path.join(output_path))


def x800_folder_to_nifti(folder_path: str, output_path: str):
    tif_path = None
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename == "Capture.tif":
                tif_path = os.path.join(dirpath, filename)

    if tif_path is None:
        raise ValueError("No Capture.tif file found in x800 folder")

    tif_to_nifti(tif_path, os.path.join(output_path))


def axeos_folder_to_nifti(folder_path: str, output_path: str):
    tif_path = None
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename == "correctedRawData.tif":
                tif_path = os.path.join(dirpath, filename)

    if tif_path is None:
        raise ValueError("No Capture.tif file found in axeos folder")

    tif_to_nifti(tif_path, os.path.join(output_path))
