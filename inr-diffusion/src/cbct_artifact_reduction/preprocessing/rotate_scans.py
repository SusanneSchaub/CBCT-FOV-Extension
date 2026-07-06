from cbct_artifact_reduction.utils import get_scanner_from_num, OUTPUT_DIR
import os
import nibabel as nib
import numpy as np


preprocessed_data = os.path.join(OUTPUT_DIR, "data")
list_scans = os.listdir(preprocessed_data)
rotated_data_path = os.path.join(OUTPUT_DIR, "rotated")


for scan in list_scans:
    if os.path.exists(os.path.join(rotated_data_path, scan)):
        print(f"Skipping {scan} because it already exists")
        continue
    else:
        print(f"Processing {scan}")

    _, scanner, _, _, _ = get_scanner_from_num(int(scan[:-7]))

    if scanner == "planmeca":
        print(f"Skipping planmeca scan {scan} ")
        continue

    scan_path = os.path.join(preprocessed_data, scan)
    img_array = np.array(nib.load(scan_path).get_fdata())
    img_corrected = np.rot90(np.swapaxes(img_array, 0, 2), 3)
    nib_corrected = nib.Nifti1Image(img_corrected, np.eye(4))
    nib.save(nib_corrected, os.path.join(rotated_data_path, scan))
    print(f"Finished processing {scan}")