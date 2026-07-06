# Description: This script will preprocess the projection data. There is data in tif and raw format from 3 different scanners. Output will be in nii.gz format.

import os

import nibabel as nib
import numpy as np
from brainglobe_utils.IO.image import save

from cbct_artifact_reduction.dataprocessing import extract_tar_gz, tif_to_nifti
from cbct_artifact_reduction.utils import DATA_DIR, OUTPUT_DIR, get_scanner_from_num

if __name__ == "__main__":

    # Define data and output folder paths
    script_dir = os.path.dirname(__file__)
    output_dir = OUTPUT_DIR
    data_dir = DATA_DIR

    # get list of folders in data_dir
    folders = [
        f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))
    ]

    # filter folder for accuitomo data
    accuitomo_folders = [f for f in folders if "Accuitomo" in f]
    # filter folder for axeo data
    axeo_folders = [f for f in folders if "Axeo" in f]
    # filter folder for planmeca data
    planmeca_folders = [f for f in folders if "Planmeca" in f]

    # Define which data to process
    accuitomo = True
    axeo = True
    planmeca = True

    for folder in folders:
        # Get list of files in folder, these should be foulders named with numbers
        files = sorted(os.listdir(os.path.join(data_dir, folder)))

        for f in files:
            print(f)
            # if f is .DS_Store, skip
            if f == ".DS_Store":
                continue
            # if f already exists in output_dir, skip
            elif os.path.exists(os.path.join(output_dir, f"{f}.nii.gz")):
                print(f"Skipping {f}, already exists in output folder")
                continue
            else:
                # Get scanner from number
                _, scanner, _, _, _ = get_scanner_from_num(int(f))

                if scanner == "accuitomo" or scanner == "x800":
                    if not accuitomo:
                        print("Skipping Accuitomo data")
                        break
                    else:
                        print("Processing Accuitomo data")
                    # Join the data path with the file name from files and Capture.tif
                    tif_path = os.path.join(data_dir, folder, f, "Capture.tif")

                    # Save all tif files as nifti, each file will have the name of its scan number
                    tif_to_nifti(tif_path, os.path.join(output_dir, f"{f}.nii.gz"))

                elif scanner == "axeos":
                    if not axeo:
                        print("Skipping Axeos data")
                        break
                    else:
                        print("Processing Axeos data")
                    # Join the data path with the file name from files and Capture.tif
                    tif_path = os.path.join(data_dir, folder, f, "correctedRawData.tif")

                    # Save all tif files as nifti
                    tif_to_nifti(tif_path, os.path.join(output_dir, f"{f}.nii.gz"))

                elif scanner == "planmeca":
                    if not planmeca:
                        print("Skipping Planmeca data")
                        break
                    else:
                        print("Processing Planmeca data")

                    # Get list of files in folder
                    contents = os.listdir(os.path.join(data_dir, folder, f))
                    for item in contents:
                        if item.endswith(".tar.gz"):
                            # Extract tar.gz files
                            extract_tar_gz(
                                os.path.join(data_dir, folder, f, item),
                                os.path.join(data_dir, folder, f),
                            )
                            extracted_path = os.path.join(
                                data_dir, folder, f, "pm3DData", item[0:-7], "corrected"
                            )

                            # Get all directories in extracted path
                            extracted_frames = sorted(os.listdir(extracted_path))
                            num_frames = len(extracted_frames)

                            _, _, _, _, fov = get_scanner_from_num(int(f))
                            if fov == "small":
                                n, m = 840, 732
                            else:
                                n, m = 420, 736

                            # Create tensor for all frames
                            vol = np.zeros((m, n, num_frames))

                            # Put all frames in tensor
                            for index, frame in enumerate(extracted_frames):
                                data = np.fromfile(
                                    os.path.join(extracted_path, frame),
                                    dtype=np.uint16,
                                )
                                data = np.rot90(np.reshape(data, (n, m)))
                                vol[..., index] = data

                            # Save tensor as nifti
                            save.to_nii(vol, os.path.join(output_dir, f"{f}.nii.gz"))
                            nifti_file = nib.nifti1.load(
                                os.path.join(output_dir, f"{f}.nii.gz")
                            )

                    # Save all tif files as nifti
                    # tif_to_nifti(tif_path, os.path.join(output_dir, f"{f}.nii.gz"))
