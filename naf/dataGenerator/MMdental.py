import os
import os.path as osp
import tigre
from tigre.utilities.geometry import Geometry
from tigre.utilities import gpu
import numpy as np
import yaml
from numpy import genfromtxt
import nibabel as nib
import tigre.algorithms as algs

import pickle
import scipy.io
import scipy.ndimage.interpolation
import _RandomNumberGenerator as RNG
import random
from tigre.utilities import CTnoise

import matplotlib.pyplot as plt
import visdom

import argparse

cfg = {"server": "localhost", "port": 8823}
viz = visdom.Visdom('http://' + cfg["server"], port = cfg["port"])

def visualize(img):

    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def add(projections, Gaussian=None, Poisson=None):

    if Poisson is not None:
        if not np.isscalar(Poisson):
            raise ValueError(
                "Poisson value should be an scalar, is " + str(type(Poisson)) + " instead."
            )
    else:
        Poisson = np.ceil(np.log2(np.max(np.abs(projections))))  # nextpow2
    if Gaussian is not None:
        if not isinstance(Gaussian, np.ndarray):
            raise ValueError(
                "Gaussian value should be an array, is " + str(type(Gaussian)) + " instead."
            )
        if Gaussian.shape != (2,):
            raise ValueError("Gaussian shape should be 1x2, is " + str(Gaussian.shape) + "instead.")
    else:
        Gaussian = np.array([0, 0.5])
    max_proj = np.max(projections)
    projections = Poisson * np.exp(-projections / max_proj)

    projections = RNG.add_noise(np.float32(projections), Gaussian[0], Gaussian[1])

    projections = -np.log(projections / Poisson) * max_proj
    projections = np.float32(projections)
    return projections


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctName", default="chest", type=str,
                        help="Name of CT")
    parser.add_argument("--outputName", default="chest_50", type=str,
                        help="Name of output data")
    parser.add_argument("--dataFolder", default="raw", type=str,
                        help="folder of raw data")
    parser.add_argument("--outputFolder", default="/home/s.schaub/MMDental/pickle_files", type=str,         #"./data"
                        help="folder of output data")
    return parser


def main():
    parser = config_parser()
    args = parser.parse_args()
    dataType = args.ctName
    dataFolder = args.dataFolder
    outputName = args.outputName
    outputFolder = args.outputFolder
    matPath = f"./dataGenerator/{dataFolder}/{dataType}/img.mat"


    nbr = 537

    configPath = "/home/s.schaub/naf_meta_learned/dataGenerator/raw/jaw/MMdental.yml"

    outputPath = osp.join(outputFolder, f"{str(nbr)}_projections.pickle")
    generator(matPath, configPath, outputPath, nbr,True)

# %% Geometry
class ConeGeometry_special(Geometry):
    """
    Cone beam CT geometry.
    """

    def __init__(self, data):
        Geometry.__init__(self)

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = data["DSD"] / 1000  # Distance Source Detector      (m)
        self.DSO = data["DSO"] / 1000  # Distance Source Origin        (m)
        # Detector parameters
        #self.nDetector = np.array([data["nDetector"][0],data["nDetector"][1]-data["outpaint"]])        #(px)
        self.nDetector = np.array(data["nDetector"])
        self.dDetector = np.array(data["dDetector"]) / 1000  # size of each pixel            (m)
        self.sDetector = self.nDetector * self.dDetector  # total size of the detector    (m)
        # Image parameters
        self.nVoxel = np.array(data["nVoxel"][::-1])  # number of voxels              (vx)
        self.dVoxel = np.array(data["dVoxel"][::-1]) / 1000  # size of each voxel            (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image       (m)

        # Offsets
        self.offOrigin = np.array(data["offOrigin"][::-1]) / 1000  # Offset of image from origin   (m)
        self.offDetector = np.array(
            [data["offDetector"][1], data["offDetector"][0]]) / 1000  # Offset of Detector            (m)

        # Auxiliary
        self.accuracy = data["accuracy"]  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        # Mode
        self.mode = data["mode"]  # parallel, cone                ...
        self.filter = data["filter"]



def convert_to_attenuation(data: np.array, rescale_slope: float, rescale_intercept: float):
    """
    CT scan is measured using Hounsfield units (HU). We need to convert it to attenuation.

    The HU is first computed with rescaling parameters:
        HU = slope * data + intercept

    Then HU is converted to attenuation:
        mu = mu_water + HU/1000x(mu_water-mu_air)
        mu_water = 0.206
        mu_air=0.0004

    Args:
    data (np.array(X, Y, Z)): CT data.
    rescale_slope (float): rescale slope.
    rescale_intercept (float): rescale intercept.

    Returns:
    mu (np.array(X, Y, Z)): attenuation map.

    """
    HU = data * rescale_slope + rescale_intercept
    mu_water = 0.206
    mu_air = 0.0004
    mu = mu_water + (mu_water - mu_air) / 1000 * HU
    # mu = mu * 100
    return mu


def loadImage(dirname, nVoxels, convert, rescale_slope, rescale_intercept, nbr, normalize=True):
    """
    Load CT image.
    """

    pth = "/home/s.schaub/MMDental/537.nii.gz"

    header = nib.load(pth).header
    print(header["pixdim"])
    test_data = nib.load(pth).get_fdata()


    test_data = test_data[:, 170:test_data.shape[1] - 0, 100:test_data.shape[2] - 170]
    #test_data = test_data[30:test_data.shape[0] - 30, 100:test_data.shape[1] - 0, 100:test_data.shape[2] - 150]

    #viz.image(visualize(test_data[..., 100]))


    #viz.image(visualize(test_data[..., 0]), opts=dict(title=str(nbr)))
    #viz.image(visualize(test_data[..., 100]), opts=dict(title=str(nbr)))
    #viz.image(visualize(test_data[..., -1]), opts=dict(title=str(nbr)))
    print(test_data.shape)



    # Loads data in F_CONTIGUOUS MODE (column major), convert to Row major
    image_ori = test_data.astype(np.float32)
    if convert:
        #print("Convert from HU to attenuation")
        image = convert_to_attenuation(image_ori, rescale_slope, rescale_intercept)
    else:
        image = image_ori

    imageDim = image.shape

    zoom_x = nVoxels[0] / imageDim[0]
    zoom_y = nVoxels[1] / imageDim[1]
    zoom_z = nVoxels[2] / imageDim[2]

    c_x = (1/zoom_x)*header["pixdim"][1] #new voxel size
    c_y = (1/zoom_y)*header["pixdim"][2]
    c_z = (1/zoom_z)*header["pixdim"][3]


    if zoom_x != 1.0 or zoom_y != 1.0 or zoom_z != 1.0:
        #print(f"Resize ct image from {imageDim[0]}x{imageDim[1]}x{imageDim[2]} to "
        #      f"{nVoxels[0]}x{nVoxels[1]}x{nVoxels[2]}")
        image = scipy.ndimage.zoom(
            image, (zoom_x, zoom_y, zoom_z), order=3, prefilter=False
        )

    image_max = np.max(image)
    image_min = np.min(image)
    image_mean = np.mean(image)
    #print("Range of CT image is [%f, %f], mean: %f" % (image_min, image_max, image_mean))
    if normalize and image_min !=0 and image_max != 1:
        #print("Normalize range to [0, 1]")
        image = (image - image_min) / (image_max - image_min)

    return image


def generator(matPath, configPath, outputPath, nbr, show=False):
    """
    Generate projections given CT image and configuration.

    """

    # Load configuration
    with open(configPath, "r") as handle:
        data = yaml.safe_load(handle)

    # Load CT image
    geo = ConeGeometry_special(data)
    img = loadImage(matPath, data["nVoxel"], data["convert"],data["rescale_slope"], data["rescale_intercept"], nbr, data["normalize"])
    data["image"] = img.copy()

    num_views = data["numTrain"] + data["numVal"]

    angle_z = np.linspace(0, 2 * np.pi, num_views)

    off_det=np.zeros((num_views,1))
    geo_SID = np.zeros((num_views,1))+data["DSD"]
    geo_SOD = np.zeros((num_views,1))+data["DSO"]

    projections_np = tigre.Ax(np.transpose(img, (2, 1, 0)).copy(), geo, angle_z)[:, ::-1, :]

    imgFDK = algs.fdk(projections_np, geo, angle_z)

    projs_crop = projections_np[:, :, 50:(projections_np.shape[2]-50)]


    geo.nDetector = np.array([projs_crop.shape[1], projs_crop.shape[2]])
    geo.sDetector = geo.nDetector * geo.dDetector

    # offset correction
    #geo.offDetector[1] += (150/2) * geo.dDetector[1] #/ 1000

    imgFDK_crop = algs.fdk(projs_crop, geo, angle_z)

    viz.image(visualize(imgFDK[100,]))
    viz.image(visualize(imgFDK_crop[100,]))

    viz.image(visualize(imgFDK_crop[0,]))
    viz.image(visualize(imgFDK_crop[50,]))
    viz.image(visualize(imgFDK_crop[100,]))
    viz.image(visualize(imgFDK_crop[150,]))
    viz.image(visualize(imgFDK_crop[200,]))
    viz.image(visualize(imgFDK_crop[-1,]))
    new_image = nib.Nifti1Image(imgFDK_crop, affine=np.eye(4))
    nib.save(new_image, '/home/s.schaub/img_Off.nii.gz')

    breakpoint()

    viz.image(visualize(imgFDK[0,]))
    viz.image(visualize(imgFDK[50,]))
    viz.image(visualize(imgFDK[100,]))
    viz.image(visualize(imgFDK[120,]))
    viz.image(visualize(imgFDK[-1,]))

    viz.image(visualize(imgFDK_crop[0,]))
    viz.image(visualize(imgFDK_crop[50,]))
    viz.image(visualize(imgFDK_crop[100,]))
    viz.image(visualize(imgFDK_crop[150,]))
    viz.image(visualize(imgFDK_crop[-1,]))


    number_of_projections = projections_np.shape[0]


    if data["randomAngle"] is False:
        train_indices = np.linspace(0, data["numTrain"] - 1, data["numTrain"])
        train_indices = np.around(train_indices)
        train_indices = train_indices.astype(int)
        projections_train = projections_np[train_indices, ...]
        train_angles = np.around(np.linspace(0, number_of_projections - 1, data["numTrain"]))

        #data["train"] = {"angles": np.linspace(0, data["totalAngle"] / 180 * np.pi, data["numTrain"]+1)[:-1] + data["startAngle"]/ 180 * np.pi}
        data["train"] = {"angles": angle_z[0:data["numTrain"]]}
        data["train"]["SID"] = {"SID": geo_SID[0:data["numTrain"]]}
        data["train"]["SOD"] = {"SOD": geo_SOD[0:data["numTrain"]]}
        data["train"]["off_det"] = {"off_det": off_det[0:data["numTrain"]]}

        val_indices = np.around(np.linspace(data["numTrain"], number_of_projections - 1, data["numVal"]))
        data["val"] = {"angles": angle_z[data["numTrain"]:]}
        data["val"]["SID"] = {"SID": geo_SID[data["numTrain"]:]}
        data["val"]["SOD"] = {"SOD": geo_SOD[data["numTrain"]:]}
        data["val"]["off_det"] = {"off_det": off_det[data["numTrain"]:]}


        val_indices = val_indices.astype(int)
        projections_val = projections_np[val_indices, ...]

    else:
        ind = np.sort(random.sample(range(number_of_projections), data["numTrain"]))
        data["train"] = {"angles": angle_z[ind]}
        data["train"]["SID"] = {"SID": geo_SID[ind]}
        data["train"]["SOD"] = {"SOD": geo_SOD[ind]}
        data["train"]["off_det"] = {"off_det": off_det[ind]}

        unique_numbers_set = set(ind)
        full_set = set(range(num_views))
        missing_numbers = list(full_set - unique_numbers_set)
        data["val"] = {"angles": angle_z[missing_numbers]}
        data["val"]["SID"] = {"SID": geo_SID[missing_numbers]}
        data["val"]["SOD"] = {"SOD": geo_SOD[missing_numbers]}
        data["val"]["off_det"] = {"off_det": off_det[missing_numbers]}

        projections_train = projections_np[ind, ...]
        projections_val = projections_np[missing_numbers, ...]

    if 1:#data["noise"] != 0 and data["normalize"]:
        #print("Add noise to projections")
        noise_projections_train = add(projections_train, Poisson=1e5, Gaussian=np.array([0, data["noise"]]))
        noise_projections_val = add(projections_val, Poisson=1e5, Gaussian=np.array([0, data["noise"]]))
        noise_projections_train[noise_projections_train < 0.0] = 0.0
        noise_projections_val[noise_projections_val < 0.0] = 0.0
        data["train"]["projections"] = noise_projections_train #/ 1e3
        data["val"]["projections"] = noise_projections_val #/ 1e3
    else:
        projections_train[projections_train < 0.0] = 0.0
        projections_val[projections_val < 0.0] = 0.0
        data["train"]["projections"] = projections_train / 1e1
        data["val"]["projections"] = projections_val / 1e1

    viz.image(visualize(data["train"]["projections"][0,]), opts=dict(title=str(nbr)))
    #viz.image(visualize(data["train"]["projections"][100,]))
    #viz.image(visualize(data["train"]["projections"][200,]))
    #viz.image(visualize(data["train"]["projections"][300,]))
    #viz.image(visualize(data["train"]["projections"][400,]))
    #viz.image(visualize(data["train"]["projections"][498,]))
    #breakpoint()

    print(data["train"]["projections"].mean(), data["train"]["projections"].shape)

    breakpoint()

    if show:
        print("Display ct image")

    # Save data
    os.makedirs(osp.dirname(outputPath), exist_ok=True)
    with open(outputPath, "wb") as handle:
        pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)

    print(f"Save files in {outputPath}")

if __name__ == "__main__":
    main()
