import csv
import os

import cbct_artifact_reduction.utils as utils

csv_file_path = os.path.join(utils.ROOT_DIR, "data.csv")

excluded_ids = [41, 208]

ids = [i for i in range(1, 401) if i not in excluded_ids]

rows = [utils.get_scanner_from_num(i) for i in ids]
print(rows)
with open(csv_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["id", "scanner", "material", "implants", "fov"])
    writer.writerows(rows)
