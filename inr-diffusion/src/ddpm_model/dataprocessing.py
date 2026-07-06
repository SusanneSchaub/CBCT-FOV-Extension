import mimetypes
import os
import tarfile
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage import transform as skTrans

from cbct_artifact_reduction.utils import ROOT_DIR


def get_filename(filepath: str) -> str:
    """Get the filename from a full filepath."""
    return os.path.basename(filepath)


def create_binary_threshold_mask(np_array, threshold):
    """Function used to threshold a numpy array."""
    return np.where(np_array > threshold, 1, 0)


def remove_outliers(
    img: np.ndarray, lower_quantile: float = 0.001, upper_quantile: float = 0.999
):
    """Clip the given image to the lower and upper quantiles."""

    return np.clip(
        img,
        np.quantile(img, lower_quantile),
        np.quantile(img, upper_quantile),
    )


def min_max_normalize(img: np.ndarray):
    """Function used to normalize image to range [0, 1]."""

    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def min_max_normalize_common(img: np.ndarray):
    """Function used to normalize image to range [0, 1]."""

    _min = img.min()
    _max = 3.5
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def min_max_normalize_common_pig(img: np.ndarray):
    """Function used to normalize image to range [0, 1]."""

    _min = img.min()
    _max = 1.2
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def guess_extensions(filename: str):
    mimetypes.add_type("image/nifti", ".nii")
    mimetypes.add_type("file/archive", ".gz")
    return [s for s in Path(filename).suffixes if s in mimetypes.types_map]


def filename_without_extension(filename: str):
    extensions = guess_extensions(filename)

    for i in extensions:
        filename = filename.replace(i, "")

    return filename


def extract_tar_gz(tar_gz_path: str, output_path: str):
    file = tarfile.open(tar_gz_path)
    file.extractall(output_path)
    file.close()


def numpy_to_nifti(np_array, output_path: str):
    assert not os.path.exists(output_path), (
        f"output path {output_path} does already exist"
    )

    # Load the data
    data = nib.nifti1.Nifti1Image(np_array, np.eye(4), dtype=np_array.dtype)
    # Save the data
    nib.nifti1.Nifti1Image.to_filename(data, output_path)


def tif_to_nifti(input_path: str, output_path: str):
    # TODO: This function does not work properly. Fix it or remove it.
    assert os.path.exists(input_path), f"input path {input_path} does not exist"
    assert not os.path.exists(output_path), (
        f"output path {output_path} does already exist"
    )

    # Load the data
    data = nib.nifti1.Nifti1Image.from_filename(input_path)
    # Save the data
    nib.nifti1.Nifti1Image.to_filename(data, output_path)


def resize_single_file(
    nifti_path: str,
    new_dimensions: tuple[int, int],
    output_file_path: str,
    preserve_range: bool = False,
):
    """Resize a single nifti file to new dimensions."""
    np_array = single_nifti_to_numpy(nifti_path)
    resized_array = skTrans.resize(
        np_array, new_dimensions, preserve_range=preserve_range
    )
    resized_nifti = nib.nifti1.Nifti1Image(resized_array, affine=None)
    nib.nifti1.save(
        resized_nifti,
        output_file_path,
    )


def single_nifti_to_numpy(nifti_path: str):
    """Convert a single nifti file to a numpy array."""
    assert os.path.exists(nifti_path), f"{nifti_path} does not exist"
    nib_object = nib.nifti1.Nifti1Image.from_filename(nifti_path)
    data = nib_object.get_fdata()
    np_array = np.array(data)
    return np_array


def nifti_vol_to_frames(nifti_path: str, output_dir: str, overwrite: bool = False):
    """Extract frames from a 3d nifti volume and save them as individual 2d nifti files.
    Input dimensions would be NxMxT, where T is the number of frames."""

    assert os.path.exists(nifti_path), f"{nifti_path} does not exist"
    assert os.path.exists(output_dir), f"{output_dir} does not exist"
    image = nib.nifti1.Nifti1Image.from_filename(nifti_path)
    np_array = np.array(image.dataobj)

    base_filename = filename_without_extension(os.path.basename(nifti_path))
    for i in range(np_array.shape[2]):
        if not overwrite and os.path.exists(
            os.path.join(output_dir, f"{base_filename}_{i}.nii.gz")
        ):
            print(f"Skipping {base_filename}_{i}.nii.gz as it already exists")
            continue
        frame = np_array[:, :, i]
        nib_frame = nib.nifti1.Nifti1Image(frame, image.affine)
        nib.nifti1.save(
            nib_frame, os.path.join(output_dir, f"{base_filename}_{i}.nii.gz")
        )


class DataFolder:
    """Class that represents a folder containing data for the CBCT artifact reduction project.
    This class defines a number of methods that can be used to interact with the data in the folder.
    """

    def __init__(self, folder_path: str, data_extension: str) -> None:
        self.folder_path: str = folder_path
        self.data_extension: str = data_extension
        self.data_path_list: list[str] = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(data_extension)
        ]

    def list_filenames(self) -> list[str]:
        """List all files in the data folder."""
        data_list = [
            f
            for f in os.listdir(self.folder_path)
            if f.endswith(f"{self.data_extension}")
        ]

        return data_list

    def print_filenames(self):
        """Print all filenames in the data folder."""
        for file in self.list_filenames():
            print(file)
        pass

    def print_filepaths(self):
        """Print all filepaths in the data folder."""
        for file in self.data_path_list:
            print(file)
        pass


class NiftiDataFolder(DataFolder):
    """Class that represents a folder containing nifti volumes or slices data for the CBCT artifact reduction project."""

    def __init__(self, folder_path: str) -> None:
        super().__init__(folder_path, ".nii.gz")

    def resize_all_files(
        self,
        output_folder_path: str,
        new_dimensions: tuple[int, int],
        overwrite_files: bool = False,
        preserve_range: bool = False,
    ):
        """Resize all nifti files in the data folder to new dimensions. Takes in the output folder path, whether to overwrite files, the new dimensions and whether to preserve the range."""

        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        for count, f_path in enumerate(self.data_path_list):
            resized_nifti_path = os.path.join(
                output_folder_path, os.path.basename(f_path)
            )
            if not overwrite_files and os.path.exists(resized_nifti_path):
                print(
                    f"Resized {get_filename(f_path)} already exists at destination. Skipping."
                )
                continue

            # Resized file doesn't exist or overwrite is True
            resize_single_file(
                f_path,
                new_dimensions,
                resized_nifti_path,
                preserve_range=preserve_range,
            )
            print(
                f"Resized file {count + 1}/{len(self.data_path_list)}. Saved at {resized_nifti_path}"
            )

    def split_all_volumes_into_frames(self, output_folder_path: str):
        """Split all nifti volumes in the data folder into frames and save them as individual 2d nifti files."""
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        for count, f_path in enumerate(self.data_path_list):
            nifti_vol_to_frames(f_path, output_folder_path)
            print(
                f"Split volume {count + 1}/{len(self.data_path_list)}. Saved at {output_folder_path}"
            )


if __name__ == "__main__":
    data_folder = NiftiDataFolder(os.path.join(ROOT_DIR, "sample_data"))
    data_folder.print_filenames()
    data_folder.print_filepaths()
