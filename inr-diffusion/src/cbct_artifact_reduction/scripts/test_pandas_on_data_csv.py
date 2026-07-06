import os

import cbct_artifact_reduction.utils as utils
import pandas as pd

path_to_data_csv = os.path.join(utils.ROOT_DIR, "data.csv")
path_to_shape_csv = os.path.join(utils.ROOT_DIR, "data_shape.csv")

data_csv = pd.read_csv(path_to_data_csv)
shape_csv = pd.read_csv(path_to_shape_csv)


A = data_csv
B = shape_csv.drop("id", axis=1)

result = pd.concat([A, B], axis=1)
result.to_csv(os.path.join(utils.ROOT_DIR, "data_with_shape.csv"), index=False)
