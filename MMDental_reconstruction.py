from __future__ import absolute_import
import sys
import os
import numpy as np
import tigre
from tigre.utilities import sample_loader
from tigre.utilities import CTnoise
import tigre.algorithms as algs
from tigre.utilities.common_geometry import staticDetectorGeo
from tigre.utilities.geometry import Geometry
from tigre.utilities.geometry_default import ConeGeometryDefault
import tigre.algorithms.single_pass_algorithms
import torch
import pickle

import visdom
import nibabel as nib
from numpy import genfromtxt
from skimage.transform import resize
from ssim import ssim
from pathlib import Path

cfg = {"server": "localhost", "port": 8823}
viz = visdom.Visdom('http://' + cfg["server"], port = cfg["port"])

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


import numpy as np

outpaint_nbr = 75

def get_geometry() -> ConeGeometryDefault:
    geo = tigre.geometry_default(high_resolution=False)
    pixel_mm, slice_mm = 3.0 / 1000, 3.0 / 1000
    row_pixels, col_pixels = 256, 256

    geo.DSD = 1500.0 / 1000
    geo.DSO = 700 / 1000

    geo.nDetector = np.array(([row_pixels, col_pixels]))
    geo.dDetector = np.array([pixel_mm, pixel_mm])
    geo.sDetector = geo.nDetector * geo.dDetector

    num_voxels_z, num_voxels_y, num_voxels_x = 256, 256, 256
    geo.nVoxel = np.array([num_voxels_z, num_voxels_y, num_voxels_x])
    geo.sVoxel = np.array([num_voxels_z * 0.9 / 1000, num_voxels_y *0.9 / 1000, num_voxels_x *0.9 / 1000])
    geo.dVoxel = geo.sVoxel / geo.nVoxel

    geo.offOrigin = np.array((0, 0, 40)) / 1000

    geo.accuracy=0.5

    center_of_rotation_y_displacement_mm = 0.
    geo.COR = center_of_rotation_y_displacement_mm  # type: ignore

    geo.mode= "cone"  # Can be "parallel" as well
    return geo


def get_geometry2() -> ConeGeometryDefault:
    geo = tigre.geometry_default(high_resolution=False)
    pixel_mm, slice_mm = 3.0 / 1000, 3.0 / 1000
    row_pixels, col_pixels = 256, 256-2*outpaint_nbr


    geo.DSD = 1500.0 / 1000
    geo.DSO = 700.0 / 1000

    geo.nDetector = np.array(([row_pixels, col_pixels]))
    geo.dDetector = np.array([pixel_mm, pixel_mm])
    geo.sDetector = geo.nDetector * geo.dDetector

    num_voxels_z, num_voxels_y, num_voxels_x = 256, 256, 256
    geo.nVoxel = np.array([num_voxels_z, num_voxels_y, num_voxels_x])
    geo.sVoxel = np.array([num_voxels_z * 0.9 / 1000, num_voxels_y * 0.9 / 1000, num_voxels_x * 0.9 / 1000])
    geo.dVoxel = geo.sVoxel / geo.nVoxel

    geo.offOrigin = np.array((0, 0, 40)) / 1000

    geo.accuracy=0.5

    # Can also be defined per angle
    center_of_rotation_y_displacement_mm = 0.
    geo.COR = center_of_rotation_y_displacement_mm  # type: ignore

    geo.mode= "cone"  # Can be "parallel" as well
    return geo

#%% Load data and generate projections
num_views = 300

angle_z = np.linspace(0, 2 * np.pi, num_views)

geo = get_geometry()

import csv

global_min = np.inf
global_max = -np.inf

numbers = []

for nbrs in numbers:

    print(nbrs)

    with open("" + str(nbrs) + "_projections.pickle", "rb") as handle:
        data = pickle.load(handle)

    projections_train = data["train"]["projections"]
    projections_val= data["val"]["projections"]
    projections = np.concatenate((projections_train, projections_val), axis=0)


    imgFDK_ori = algs.ossart(projections, geo, angle_z, niter=100)

    projs_not_crop = projections.copy()


    projections = projections[...,outpaint_nbr:256-outpaint_nbr]

    imgFDK = algs.fdk(projections, get_geometry2(), angle_z, niter=100)


    outpaint=nib.load("" + str(nbrs) + "").get_fdata()

    outpaint = outpaint.astype(np.float32)


    outpaint[...,outpaint_nbr:256-outpaint_nbr] = projs_not_crop[...,outpaint_nbr:256-outpaint_nbr]


    imgFDK_outpaint = algs.ossart(outpaint, geo, angle_z, niter=100)

    mini = -0.2

    for i in range(0,256):
        x = 128
        y = 128
        r = 120
        Y, X = np.ogrid[:256, :256]
        dist_from_center = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
        _mask = dist_from_center <= r
        imgFDK[i,_mask==0]=mini

    for i in range(0,256):
        x = 128
        y = 128
        r = 120
        Y, X = np.ogrid[:256, :256]
        dist_from_center = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
        _mask = dist_from_center <= r
        imgFDK_outpaint[i,_mask==0]=mini

    for i in range(0,256):
        x = 128
        y = 128
        r = 120
        Y, X = np.ogrid[:256, :256]
        dist_from_center = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
        _mask = dist_from_center <= r
        imgFDK_ori[i,_mask==0]=mini

    breakpoint()


    global_min = min(global_min, imgFDK_ori.min(), imgFDK_outpaint.min())
    global_max = max(global_max, imgFDK_ori.max(), imgFDK_outpaint.max())


    vol_gt_ni = nib.Nifti1Image(imgFDK_ori, affine=np.eye(4))
    vol_inr_ni = nib.Nifti1Image(imgFDK_outpaint, affine=np.eye(4))
    vol_ossart_ni = nib.Nifti1Image(imgFDK, affine=np.eye(4))

    nib.save(vol_gt_ni,"" + str(nbrs) + "_gt" + "_vol" + ".nii.gz")
    nib.save(vol_inr_ni,"" + str(nbrs) + "_inr" + "_vol" + ".nii.gz")
    nib.save(vol_ossart_ni,"" + str(nbrs) + "_ossart" + "_vol" + ".nii.gz")



    for j in range(0,imgFDK_ori.shape[0]):

        slice_gt = imgFDK_ori[j,...]
        slice_inr = imgFDK_outpaint[j,...]
        slice_ossart = imgFDK[j, ...]

        slice_gt_ni = nib.Nifti1Image(slice_gt, affine=np.eye(4))
        slice_inr_ni = nib.Nifti1Image(slice_inr, affine=np.eye(4))
        slice_ossart_ni = nib.Nifti1Image(slice_ossart, affine=np.eye(4))

        nib.save(slice_gt_ni, "" + str(nbrs) + "_gt_" + str(j) + ".nii.gz")
        nib.save(slice_inr_ni, "" + str(nbrs) + "_inr_" + str(j) + ".nii.gz")
        nib.save(slice_ossart_ni, "" + str(nbrs) + "_ossart_" + str(j) + ".nii.gz")



print("Global minimum:", global_min)
print("Global maximum:", global_max)



