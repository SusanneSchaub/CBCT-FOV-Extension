import pdb

import torch
import pickle
import os
import sys
import numpy as np
import yaml
from numpy import genfromtxt
import copy
from torch.utils.data import DataLoader, Dataset
import torch
import visdom


def visualize(img):

    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


class ConeGeometry(object):
    """
    Cone beam CT geometry. Note that we convert to meter from millimeter.
    """
    def __init__(self, data, prefix):

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        if prefix=="train":
            tmp1=np.array(list(data["train"]["SID"].values()))
            tmp2=np.array(list(data["train"]["SOD"].values()))
            tmp1 = torch.from_numpy(tmp1)
            tmp2 = torch.from_numpy(tmp2)

            self.DSD = tmp1 / 1000   # Distance Source Detector      (m)
            self.DSO = tmp2 / 1000     # Distance Source Origin        (m)

        else:
            tmp1=np.array(list(data["val"]["SID"].values()))
            #tmp1=np.ones_like(tmp1)*data["DSD"]
            tmp2=np.array(list(data["val"]["SOD"].values()))
            #tmp2=np.ones_like(tmp2)*data["DSO"]
            tmp1 = torch.from_numpy(tmp1)
            tmp2 = torch.from_numpy(tmp2)

            self.DSD = tmp1  /1000 # Distance Source Detector      (m)
            self.DSO = tmp2  /1000  # Distance Source Origin        (m)
            self.DSD = self.DSD.to(torch.float32)
            self.DSO = self.DSO.to(torch.float32)

        # Detector parameters

        self.nDetector = np.array(data["nDetector"])
        self.nDetector  = torch.from_numpy(self.nDetector)# number of pixels              (px)
        self.dDetector = np.array(data["dDetector"]) / 1000
        self.dDetector = torch.from_numpy(self.dDetector)# size of each pixel            (m)
        self.sDetector = self.nDetector * self.dDetector  # total size of the detector    (m)
        # Image parameters
        self.nVoxel = np.array(data["nVoxel"][::-1])  # number of voxels              (vx)
        self.nVoxel = torch.from_numpy(self.nVoxel)
        self.dVoxel = np.array(data["dVoxel"][::-1]) / 1000  # size of each voxel            (m)
        self.dVoxel  = torch.from_numpy(self.dVoxel)
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image       (m)

        # OffsetsData
        self.offOrigin = np.array(data["offOrigin"][::-1]) / 1000 # Offset of image from origin   (m)
        self.offOrigin = torch.from_numpy(self.offOrigin)


        if prefix == "train":

            tmp1 = np.zeros_like(tmp1)
            tmp1 = np.swapaxes(tmp1,0,1)
            tmp1 = np.zeros_like(tmp1)
            tmp1 = np.repeat(tmp1, 2, axis=1)
            tmp1[:,0] = 0
            tmp1[:,1] = 0
            self.offDetector = tmp1

        else:
            if 0:
                tmp1 = np.array(list(data["val"]["off_det"].values()))
            else:
                tmp1 = np.zeros_like(tmp1)
                tmp1 = np.swapaxes(tmp1, 0, 1)
            tmp1 = np.repeat(tmp1, 2, axis=1)
            tmp1[:,0] = 0
            tmp1[:,1] = 0
            self.offDetector = tmp1


        # Auxiliary
        self.accuracy = data["accuracy"]  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        # Mode
        self.mode = data["mode"]  # parallel, cone                ...
        self.filter = data["filter"]



class TIGREDataset(Dataset):
    """
    TIGRE dataset.
    """
    def __init__(self, path, n_rays=501, type="train", device="cuda", inner_steps=5, adapt_state=False, outpaint=50):
        super().__init__()

        with open(path, "rb") as handle:
            data = pickle.load(handle)

        self.outpaint = outpaint
        self.adapt_state = adapt_state
        self.inner_stp = inner_steps
        self.geo_train = ConeGeometry(data, prefix="train")
        self.geo_val = ConeGeometry(data, prefix="val")
        self.type = type
        self.n_rays = n_rays
        #self.near_train, self.far_train = self.get_near_far(self.geo_train) #??? use only max of both?
        #self.near_val, self.far_val = self.get_near_far(self.geo_val)
        self.near, self.far = self.get_near_far(self.geo_train, self.geo_val)
        self.projs = torch.tensor(data[type]["projections"], dtype=torch.float32, device=device)

        self.angles = data[type]["angles"]
        #pdb.set_trace()

        if type == "train":

            jj=None
            rays = self.get_rays(self.angles, self.geo_train, jj, device, mode='normal')
            self.rays = torch.cat([rays, torch.ones_like(rays[...,:1])*self.near, torch.ones_like(rays[...,:1])*self.far], dim=-1)
            self.n_samples = data["numTrain"]
            coords = torch.stack(torch.meshgrid(torch.linspace(0, self.geo_train.nDetector[0] - 1, self.geo_train.nDetector[0], device=device),
                                                torch.linspace(0, self.geo_train.nDetector[1] - 1, self.geo_train.nDetector[1], device=device), indexing="ij"),-1)

            self.coords = torch.reshape(coords, [-1, 2])
            self.voxels = torch.tensor(self.get_voxels(self.geo_train), dtype=torch.float32, device=device)
            self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)
        elif type == "val":
            jj=None
            rays = self.get_rays(self.angles, self.geo_val, jj, device, mode='normal')
            #rays = self.get_rays(self.angles, self.geo_val, device, mode='normal')
            self.rays = torch.cat([rays, torch.ones_like(rays[...,:1])*self.near, torch.ones_like(rays[...,:1])*self.far], dim=-1)
            self.n_samples = data["numVal"]
            self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)
            self.voxels = torch.tensor(self.get_voxels(self.geo_val), dtype=torch.float32, device=device)

        
    def __len__(self):
        return self.n_samples


    def __getitem__(self, index):
        if self.type == "train":

            list_proj = []
            list_rays = []

            if self.adapt_state == True:
                sample_only_inner_part = False
            else:
                sample_only_inner_part = True

            for stp in range(self.inner_stp): #add +1 for meta learning!

                otp = self.outpaint  # 100 for 512 images

                if sample_only_inner_part == True:

                    if stp != self.inner_stp:
                        projs_valid = (self.projs[index] > 0)
                        H, W = projs_valid.shape
                        border_mask = torch.zeros((H, W), dtype=torch.bool, device=projs_valid.device)
                        border_mask[:, otp:W-otp] = True
                        final_mask = projs_valid & border_mask
                        coords_valid = self.coords[final_mask.flatten()]

                    if stp == self.inner_stp:

                        projs_valid = (self.projs[index] > 0).flatten()
                        coords_valid = self.coords[projs_valid]

                else:
                    projs_valid = (self.projs[index] > 0)
                    H, W = projs_valid.shape
                    border_mask = torch.zeros((H, W), dtype=torch.bool, device=projs_valid.device)
                    border_mask[:, otp:W - otp] = True
                    final_mask = projs_valid & border_mask
                    coords_valid = self.coords[final_mask.flatten()]


                    #for tv calculation of outpainted region
                    border_mask = ~border_mask
                    final_mask_tv_out = projs_valid & border_mask
                    coords_valid_tv_out = self.coords[final_mask_tv_out.flatten()]



                select_inds = np.random.choice(coords_valid.shape[0], size=[self.n_rays], replace=False)
                select_coords = coords_valid[select_inds].long()

                rays = self.rays[index, select_coords[:, 0], select_coords[:, 1]]
                projs = self.projs[index, select_coords[:, 0], select_coords[:, 1]]       #dont change!!
                list_rays.append(rays)
                list_proj.append(projs)

            out = {
                "projs":torch.stack(list_proj),
                "rays":torch.stack(list_rays),
                #"rays_out": torch.stack(list_rays_tv_out),
            }

        elif self.type == "val":
            rays = self.rays[index]
            projs = self.projs[index]
            out = {
                "projs":projs,
                "rays":rays,
            }
        return out

    def get_voxels(self, geo: ConeGeometry):
        """
        Get the voxels.
        """
        n1, n2, n3 = geo.nVoxel 
        s1, s2, s3 = geo.sVoxel / 2 - geo.dVoxel / 2

        xyz = np.meshgrid(np.linspace(-s1, s1, n1),
                        np.linspace(-s2, s2, n2),
                        np.linspace(-s3, s3, n3), indexing="ij")
        voxel = np.asarray(xyz).transpose([1, 2, 3, 0])

        return voxel


    def get_outpaint_rays(self, geo, j, device='cuda'):

        geoo = copy.copy(geo)
        geoo.DSD = copy.copy([geo.DSD[0]])
        geoo.DSO = copy.copy([geo.DSO[0]])
        rays = self.get_rays([self.angles[j]], geoo, j, device, mode='outpaint',save=True)
        return torch.cat([rays, torch.ones_like(rays[..., :1]) * self.near, torch.ones_like(rays[..., :1]) * self.far], dim=-1)


    def get_generated_rays(self,geo, j, angle_more, offset_nbr, device='cuda'):

        geoo = copy.copy(geo)
        geoo.DSD = copy.copy([geo.DSD[0]])
        geoo.DSO = copy.copy([geo.DSO[0]])
        tmp = copy.copy(geo.offDetector)
        if offset_nbr != None:
            tmp[:,0] = offset_nbr[0] / 1000
            tmp[:,1] = offset_nbr[1] / 1000
            geoo.offDetector = tmp

        if angle_more == False:
            rays = self.get_rays([self.angles[j]], geoo, j, device, mode='normal',save=True) #angle_zz
        else:
            rays = self.get_rays([self.angles_more[j]], geoo, j, device, mode='normal', save=True)
        return torch.cat([rays, torch.ones_like(rays[..., :1]) * self.near, torch.ones_like(rays[..., :1]) * self.far], dim=-1)


    def get_rays(self, angles, geo: ConeGeometry, jj, device, mode, save=False):
        """
        Get rays given one angle and x-ray machine geometry.
        """

        H, W = geo.nDetector
        rays = []


        indx=0
        for angle in angles:
            if jj!=None:
                DSD = geo.DSD[0][jj]
                DSO = geo.DSO[0][jj]

                goD0 = geo.offDetector[jj, 0]
                goD1 = geo.offDetector[jj, 1]

            else:
                DSD = geo.DSD[0][indx]
                DSO = geo.DSO[0][indx]

                goD0 = geo.offDetector[indx, 0]
                goD1 = geo.offDetector[indx, 1]

            pose = torch.Tensor(self.angle2pose(DSO, angle)).to(device)
            rays_o, rays_d = None, None
            if geo.mode == "cone":

                i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                    torch.linspace(0, H - 1, H, device=device), indexing="ij")  # pytorch"s meshgrid has indexing="ij"

                uu = (i.t() + 0.5 - W / 2) * (geo.dDetector[0]).item() + goD0.item()
                vv = (j.t() + 0.5 - H / 2) * (geo.dDetector[1]).item() + goD1.item()
                DSD=(DSD.to(torch.float32)).to(device)

                dirs = torch.stack([uu / DSD, vv / DSD, torch.ones_like(uu)], -1)
                rays_d = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1) # pose[:3, :3] *
                rays_o = pose[:3, -1].expand(rays_d.shape)
            elif geo.mode == "parallel":

                    i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                                torch.linspace(0, H - 1 , H, device=device), indexing="ij")  # pytorch"s meshgrid has indexing="ij"

                    uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + goD0
                    vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + goD1
                    dirs = torch.stack([torch.zeros_like(uu), torch.zeros_like(uu), torch.ones_like(uu)], -1)
                    rays_d = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1) # pose[:3, :3] *
                    rays_o = torch.sum(torch.matmul(pose[:3,:3], torch.stack([uu,vv,torch.zeros_like(uu)],-1)[..., None]).to(device), -1) + pose[:3, -1].expand(rays_d.shape)


            rays.append(torch.concat([rays_o, rays_d], dim=-1))
            indx = indx + 1

        return torch.stack(rays, dim=0)

    def angle2pose(self, DSO, angle):
        phi1 = -np.pi / 2
        R1 = np.array([[1.0, 0.0, 0.0],
                    [0.0, np.cos(phi1), -np.sin(phi1)],
                    [0.0, np.sin(phi1), np.cos(phi1)]])
        phi2 = np.pi / 2
        R2 = np.array([[np.cos(phi2), -np.sin(phi2), 0.0],
                    [np.sin(phi2), np.cos(phi2), 0.0],
                    [0.0, 0.0, 1.0]])
        R3 = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0]])
        rot = np.dot(np.dot(R3, R2), R1)


        trans = np.array([(DSO * np.cos(angle)).item(), (DSO * np.sin(angle)).item(), 0])
        T = np.eye(4)
        T[:-1, :-1] = rot
        T[:-1, -1] = trans
        return T

    def get_near_far(self, geo1, geo2, tolerance=0.005):
        """
        Compute the near and far threshold.
        """
        stack_DSO=np.hstack((geo1.DSO,geo2.DSO))

        dist1 = np.linalg.norm([geo1.offOrigin[0] - geo1.sVoxel[0] / 2, geo1.offOrigin[1] - geo1.sVoxel[1] / 2])
        dist2 = np.linalg.norm([geo1.offOrigin[0] - geo1.sVoxel[0] / 2, geo1.offOrigin[1] + geo1.sVoxel[1] / 2])
        dist3 = np.linalg.norm([geo1.offOrigin[0] + geo1.sVoxel[0] / 2, geo1.offOrigin[1] - geo1.sVoxel[1] / 2])
        dist4 = np.linalg.norm([geo1.offOrigin[0] + geo1.sVoxel[0] / 2, geo1.offOrigin[1] + geo1.sVoxel[1] / 2])
        dist_max = np.max([dist1, dist2, dist3, dist4])
        zer=np.zeros_like((stack_DSO))
        near = np.max([zer, stack_DSO- dist_max - tolerance])
        far = np.min([stack_DSO * 2, stack_DSO + dist_max + tolerance])
        return near, far
