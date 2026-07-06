import csv
import os
import random

import pandas as pd

from cbct_artifact_reduction import utils


def get_slice_ids(file_path, id, shuffle=False):
    """
    Generates a list of slice filenames in the format 'id_randomFrameNumber.nii.gz'.

    :param file_path: Path to the CSV file containing 'id' and 'frames' columns.
    :param id: ID to generate slice filenames for.
    :return: List of filenames in the format 'id_randomFrameNumber.nii.gz'.
    """
    df = pd.read_csv(file_path)

    id_to_frames = df.set_index("id")["frames"].to_dict()
    frames = id_to_frames[id[0]]

    slices = [f"{id[0]}_{i}.nii.gz" for i in range(frames)]
    if shuffle:
        random.shuffle(slices)

    return slices


def get_random_slice_from_id(file_path, ids):
    """
    Generates a list of slice filenames in the format 'id_randomFrameNumber.nii.gz' and returns a random slice for each id from the list.

    :param file_path: Path to the CSV file containing 'id' and 'frames' columns.
    :param ids: List of IDs to generate slice filenames for.
    :return: List of filenames in the format 'id_randomFrameNumber.nii.gz'.
    """
    df = pd.read_csv(file_path)
    id_to_frames = df.set_index("id")[
        "frames"
    ].to_dict()  # Create a lookup for frames per ID

    slices = [
        f"{id}_{random.randint(0, id_to_frames.get(id, 0) - 1)}.nii.gz" for id in ids
    ]

    return slices


def filter_csv(file_path, output_path=None, exclude_filled_radiation=False, **filters):
    """
    Filters a CSV file based on given column conditions. Allows multiple values per filter key,
    including passing comma-separated strings. Optionally excludes rows with values in the 'radiation' column.

    :param file_path: Path to the CSV file.
    :param output_path: (Optional) Path to save the filtered CSV.
    :param exclude_filled_radiation: If True, removes rows where 'radiation' has a value.
    :param filters: Column-value pairs to filter data (e.g., scanner="axeos,planmeca", material="ti").
    :return: Filtered DataFrame with the same data types as the original.
    """
    df = pd.read_csv(file_path)
    original_dtypes = df.dtypes  # Store original data types

    # Convert all filter values to string for consistency
    for column in filters:
        if isinstance(filters[column], str) and "," in filters[column]:
            filters[column] = filters[column].split(",")

    # Apply filters
    for column, value in filters.items():
        if column in df.columns:
            if isinstance(value, list):  # Allow multiple values per filter key
                df = df[df[column].astype(str).isin(value)]
            else:
                df = df[df[column].astype(str) == str(value)]

    # Exclude rows where 'radiation' has a value
    if exclude_filled_radiation and "radiation" in df.columns:
        df = df[df["radiation"].isna()]

    # Restore original data types
    df = df.astype(original_dtypes)

    # Save if output path is provided
    if output_path:
        df.to_csv(output_path, index=False)

    return df


def get_random_samples(file_path, n=5, **filters):
    """
    Returns n random samples from a CSV file, applying column-value filters if needed.

    :param file_path: Path to the CSV file.
    :param n: Number of random samples to return (default = 5).
    :param filters: Column-value pairs for filtering,
                    e.g., scanner="axeos" or scanner="axeos,planmeca".
    :return: A Pandas DataFrame of the sampled rows.
    """
    # Read the CSV and store original data types
    df = pd.read_csv(file_path)
    original_dtypes = df.dtypes

    # Apply filters
    for column, value in filters.items():
        if column in df.columns:
            # If the value is a comma-separated string, split into a list
            if isinstance(value, str) and "," in value:
                value_list = [v.strip() for v in value.split(",")]
                df = df[df[column].astype(str).isin(value_list)]
            else:
                df = df[df[column].astype(str) == str(value)]

    # Sample rows (ensuring we don't request more than available)
    df = df.sample(n=min(n, len(df)), random_state=42)

    # Restore original data types
    df = df.astype(original_dtypes)

    return df


def get_random_entries(file_path, n, exclude_filled_radiation=False, **filters):
    """
    Returns n random entries from the filtered CSV data.

    :param file_path: Path to the CSV file.
    :param n: Number of random entries to return.
    :param exclude_filled_radiation: If True, removes rows where 'radiation' has a value.
    :param filters: Column-value pairs to filter data before sampling.
    :return: Sampled DataFrame with the same data types as the original.
    """
    filtered_df = filter_csv(
        file_path, exclude_filled_radiation=exclude_filled_radiation, **filters
    )
    return filtered_df.sample(
        n=min(n, len(filtered_df)), random_state=None
    )  # Ensuring reproducibility


def get_random_entries_by_scanner_fov_mandible(
    file_path, n, exclude_filled_radiation=True, exclude_mandibles=None
):
    """
    Returns n random entries for each combination of scanner, fov, and mandible,
    with options to exclude specific mandibles and remove rows with filled radiation values.

    :param file_path: Path to the CSV file.
    :param n: Number of random entries to return per scanner, fov, and mandible combination.
    :param exclude_filled_radiation: If True, removes rows where 'radiation' has a value.
    :param exclude_mandibles: List of mandible values to exclude.
    :return: Sampled DataFrame with entries from each scanner, fov, and mandible group.
    """
    df = pd.read_csv(file_path)
    original_dtypes = df.dtypes  # Store original data types

    if not {"scanner", "fov", "mandible"}.issubset(df.columns):
        raise ValueError(
            "Columns 'scanner', 'fov', and 'mandible' must be present in the dataset."
        )

    # Exclude rows where 'radiation' has a value
    if exclude_filled_radiation and "radiation" in df.columns:
        df = df[df["radiation"].isna()]

    # Exclude specific mandible values
    if exclude_mandibles:
        df = df[~df["mandible"].astype(str).isin(map(str, exclude_mandibles))]

    sampled_df = df.groupby(["scanner", "fov", "mandible"], group_keys=False).apply(
        lambda x: x.sample(n=min(n, len(x)), random_state=42)
    )

    # Restore original data types
    sampled_df = sampled_df.astype(original_dtypes)

    return sampled_df


def filter_with_user_input(file_path):
    """
    Asks the user for input to filter the CSV file interactively and prints available options.

    :param file_path: Path to the CSV file.
    :return: Filtered DataFrame with the same data types as the original.
    """
    df = pd.read_csv(file_path)
    print("Available columns:", list(df.columns))
    filters = {}

    while True:
        column = input(
            "Enter column name to filter by (or press Enter to stop): "
        ).strip()
        if not column:
            break
        if column not in df.columns:
            print(f"Column '{column}' does not exist. Try again.")
            continue

        unique_values = df[column].dropna().unique()
        print(f"Available options for '{column}': {', '.join(map(str, unique_values))}")
        value = input(
            f"Enter values for '{column}' (comma-separated for multiple values): "
        ).strip()
        filters[column] = value

    exclude_filled_radiation = (
        input("Exclude rows with filled radiation values? (yes/no): ").strip().lower()
        == "yes"
    )

    return filter_csv(
        file_path, exclude_filled_radiation=exclude_filled_radiation, **filters
    )


def get_slices_from_ids(csv_path: str, ids: list[str]):
    """Get the slices from the ids.

    Args:
        csv_path (str): Path to the data csv.
        ids (list[str]): List of ids.

    Returns:
        list[str]: List of slice paths.
    """

    data_csv = pd.read_csv(csv_path)
    amount_of_slices = data_csv[data_csv["id"].isin(ids)]["frames"]
    slice_paths = []

    for iter, id in enumerate(ids):
        num_slices = amount_of_slices.values[iter]
        for n in range(num_slices):
            slice_paths.append(f"{id}_{n}.nii.gz")
    return slice_paths


def createSliceCSV(slices: list[str], csv_output_path: str):
    """Populate a csv file paths to the slices.

    Args:
        slices (list[str]): List of paths to the slices.
        csv_output_path (str): Path to the csv file.
    """

    rows = zip(slices)
    with open(csv_output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["slice"])
        writer.writerows(rows)


def createSliceMaskCSV(slices: list[str], masks: list[str], csv_output_path: str):
    """Populate a CSV file with everything necessary to create a dataset for the training of an inpainting model.

    The csv file will contain 2 columns: 'slice' and 'mask'. The 'slice' column will contain the path to the slice image, and the 'mask' column will contain the path to the mask image.

    Args:
        slices (list[str]): List of paths to the slices.
        masks (list[str]): List of paths to the masks.
    """
    assert len(slices) == len(masks), (
        "The number of slices and masks paths must be the same."
    )

    rows = zip(slices, masks)
    with open(csv_output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["slice", "mask"])
        writer.writerows(rows)


if __name__ == "__main__":
    csv_path = os.path.join(utils.ROOT_DIR, "data.csv")
    ids = filter_with_user_input(csv_path).id.tolist()
    print(ids)
    slice_paths = get_slices_from_ids(csv_path, ids)
    args = input(
        "Type slice for csv with slice paths or mask for csv with slice+mask paths:"
    )
    path = input(
        "What name do you want to save the csv file as (default: training_data.csv)?:"
    )
    if path == "":
        path = "training_data.csv"

    if args == "slice":
        createSliceCSV(slice_paths, os.path.join(utils.ROOT_DIR, f"{path}"))
    elif args == "mask":
        print("Not implemented yet")
