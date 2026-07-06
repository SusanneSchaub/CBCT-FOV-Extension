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

#cfg = {"server": "localhost", "port": 8824}
#viz = visdom.Visdom('http://' + cfg["server"], port = cfg["port"])

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
            #tmp1=np.ones_like(tmp1)*data["DSD"]
            tmp2=np.array(list(data["train"]["SOD"].values()))
            #tmp2=np.ones_like(tmp2)*data["DSO"]
            tmp1 = torch.from_numpy(tmp1)
            tmp2 = torch.from_numpy(tmp2)

            self.DSD = tmp1 / 1000  #np.array(list(data["train"]["SID"].values())) /1000 #geo_SID / 1000 #data["DSD"] / 1000#geo_SID[0] / 1000  #data["DSD"] / 1000  # Distance Source Detector      (m)
            self.DSO = tmp2 / 1000   #np.array(list(data["train"]["SOD"].values())) /1000 #geo_SOD / 1000 #data["DSO"] / 1000#geo_SOD[0] / 1000 #data["DSO"] / 1000  # Distance Source Origin        (m)
            #for fid dataset just use self.DSD = tmp1, self.DSO = tmp2

        else:
            tmp1=np.array(list(data["val"]["SID"].values()))
            #tmp1=np.ones_like(tmp1)*data["DSD"]
            tmp2=np.array(list(data["val"]["SOD"].values()))
            #tmp2=np.ones_like(tmp2)*data["DSO"]
            tmp1 = torch.from_numpy(tmp1)
            tmp2 = torch.from_numpy(tmp2)

            self.DSD = tmp1  /1000 #[602/ 1000] #np.array(list(data["val"]["SID"].values())) / 1000 #geo_SID / 1000 #data["DSD"] / 1000#geo_SID[0] / 1000  #data["DSD"] / 1000  # Distance Source Detector      (m)
            self.DSO = tmp2  /1000 #[332/ 1000] #np.array(list(data["val"]["SOD"].values())) / 1000 #geo_SOD / 1000 #data["DSO"] / 1000#geo_SOD[0] / 1000 #data["DSO"] / 1000  # Distance Source Origin        (m)
            #for fid dataset just use self.DSD = tmp1, self.DSO = tmp2

        # Detector parameters

        self.nDetector = np.array(data["nDetector"])  # number of pixels              (px)
        self.dDetector = np.array(data["dDetector"]) / 1000  # size of each pixel            (m)
        self.sDetector = self.nDetector * self.dDetector  # total size of the detector    (m)
        # Image parameters
        self.nVoxel = np.array(data["nVoxel"][::-1])  # number of voxels              (vx)
        self.dVoxel = np.array(data["dVoxel"][::-1]) / 1000  # size of each voxel            (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image       (m)

        # self.nDetector = np.array(([560,160])) # 560,160
        # self.sDetector = np.array(([5.6000000000000005, 5.6000000000000005]))   #(m)
        # self.dDetector = self.sDetector / self.nDetector
        # num_voxels_z, num_voxels_y, num_voxels_x = 256, 256, 256  # 256, 256, 256 #761, 761, 761 #761
        # self.nVoxel = np.array([num_voxels_z, num_voxels_y, num_voxels_x])
        # self.sVoxel = np.array([2.0, 2.0, 2.0])   #(m)
        # self.dVoxel = self.sVoxel / self.nVoxel


        # OffsetsData
        self.offOrigin = np.array([0,0,0])  #np.array(data["offOrigin"][::-1]) / 1000  # Offset of image from origin   (m)
        #self.offDetector = np.array([data["offDetector"][0], data["offDetector"][1], 0]) / 1000  # Offset of Detector   #         (m) #changed!!!

        file_list = False
        regi = False

        if prefix == "train":
            if file_list != True:
                if 0:
                    tmp1 = np.array(list(data["train"]["off_det"].values()))
                else:
                    tmp1 = np.zeros_like(tmp1)
                    tmp1 = np.swapaxes(tmp1,0,1)
                tmp1 = np.zeros_like(tmp1)
                tmp1 = np.repeat(tmp1, 2, axis=1)
                tmp1[:,0] = 0#0.016/1000
                tmp1[:,1] = 0#-0.211/1000
                self.offDetector = tmp1 #np.array(list(data["train"]["off_det"].values())) / 1000 #off_det / 1000
            else:
                if regi != True:
                    max_array1 = np.load('/raid/cian/user/susanne.schaub/registration/offsets_regi1.npy',
                                         allow_pickle=True)
                    max_array1[:, 0] = -max_array1[:, 0]
                    max_array1[:, 1] = -max_array1[:, 1]
                    # max_array2 = np.load('/raid/cian/user/susanne.schaub/naf/results_phantom_vis_2iter/jaw_50/eval/epoch_00300/offsets.npy',allow_pickle=True)

                    self.offDetector = max_array1 / 1000
                    #self.offDetector = np.array(
                    #    [[float(part.strip()) for part in entry.split(',')] for entry in max_array1]) / 1000  ## ???
                else:
                    self.offDetector = max_array1 / 1000
        else:
            if 0:
                tmp1 = np.array(list(data["val"]["off_det"].values()))
            else:
                tmp1 = np.zeros_like(tmp1)
                tmp1 = np.swapaxes(tmp1, 0, 1)
            tmp1 = np.repeat(tmp1, 2, axis=1)
            tmp1[:,0] = 0 #0.223 / 1000 #-0.205/1000   #regi: -0.224, -0.946 #create again for iter 1 (without generating all projections)
            tmp1[:,1] = 0 #-0.946 / 1000 #0.249/1000
            self.offDetector = tmp1 #np.array(list(data["val"]["off_det"].values()))  / 1000  # off_det / 1000Data


        # Auxiliary
        self.accuracy = data["accuracy"]  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        # Mode
        self.mode = data["mode"]  # parallel, cone                ...
        self.filter = data["filter"]



class TIGREDataset(Dataset):
    """
    TIGRE dataset.
    """
    def __init__(self, paths, n_rays=501, type="train", device="cuda", inner_steps=5, adapt_state=False, outpaint=50):
        super().__init__()

        self.adapt_state = adapt_state
        self.paths = paths
        self.n_proj = 499

        self.outpaint = outpaint

        self.nvolumes= len(paths)

        with open(paths[0], "rb") as handle:
            data = pickle.load(handle)

        self.inner_stp = inner_steps
        self.geo_train = ConeGeometry(data, prefix="train")
        self.geo_val = ConeGeometry(data, prefix="val")
        self.type = type
        self.n_rays = n_rays
        #self.near_train, self.far_train = self.get_near_far(self.geo_train) #??? use only max of both?
        #self.near_val, self.far_val = self.get_near_far(self.geo_val)
        self.near, self.far = self.get_near_far(self.geo_train, self.geo_val)

        self.device = "cuda"
        self.angles = data[type]["angles"]
        #pdb.set_trace()

        if type == "train":

            jj=None
            rays = self.get_rays(self.angles, self.geo_train, jj, device, mode='normal')
            #rays = self.get_rays(self.angles, self.geo_train, device, mode='normal')
            self.rays = torch.cat([rays, torch.ones_like(rays[...,:1])*self.near, torch.ones_like(rays[...,:1])*self.far], dim=-1)
            self.n_samples = data["numTrain"]
            coords = torch.stack(torch.meshgrid(torch.linspace(0, self.geo_train.nDetector[0] - 1, self.geo_train.nDetector[0], device=device),
                                                torch.linspace(0, self.geo_train.nDetector[1] - 1, self.geo_train.nDetector[1], device=device), indexing="ij"),-1)

            self.coords = torch.reshape(coords, [-1, 2])
            #self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)
            self.voxels = torch.tensor(self.get_voxels(self.geo_train), dtype=torch.float32, device=device)
            self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)
        elif type == "val":

            self.projs = torch.tensor(data[self.type]["projections"], dtype=torch.float32, device=self.device)
            jj=None
            rays = self.get_rays(self.angles, self.geo_val, jj, device, mode='normal')
            #rays = self.get_rays(self.angles, self.geo_val, device, mode='normal')
            self.rays = torch.cat([rays, torch.ones_like(rays[...,:1])*self.near, torch.ones_like(rays[...,:1])*self.far], dim=-1)
            self.n_samples = data["numVal"]
            self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)
            self.voxels = torch.tensor(self.get_voxels(self.geo_val), dtype=torch.float32, device=device)

        
    def __len__(self):
        return self.n_volumes*self.n_samples


    def __getitem__(self, index):

        volume_idx = index // self.n_proj
        proj_idx = index % self.n_proj

        with open(self.paths[volume_idx], "rb") as handle:
            data = pickle.load(handle)

        data = torch.tensor(data[self.type]["projections"], dtype=torch.float32, device=self.device)


        if self.type == "train":

            if self.adapt_state == True:
                sample_only_inner_part = False
            else:
                sample_only_inner_part = True

            list_proj = []
            list_rays = []

            for stp in range(self.inner_stp+1):

                otp = self.outpaint

                if sample_only_inner_part == True:

                    if stp != self.inner_stp:
                        projs_valid = (data[proj_idx] > 0)
                        H, W = projs_valid.shape
                        border_mask = torch.zeros((H, W), dtype=torch.bool, device=projs_valid.device)
                        border_mask[:, otp:W-otp] = True
                        final_mask = projs_valid & border_mask
                        coords_valid = self.coords[final_mask.flatten()]

                    if stp == self.inner_stp:
                        # projs_valid = (data[index] > 0)
                        # H, W = projs_valid.shape
                        # border_mask = torch.zeros((H, W), dtype=torch.bool, device=projs_valid.device)
                        # border_mask[:,0:otp] = True
                        # border_mask[:,(W-otp):W] = True
                        # final_mask = projs_valid & border_mask
                        # coords_valid = self.coords[final_mask.flatten()]

                        projs_valid = (data[proj_idx] > 0).flatten()
                        coords_valid = self.coords[projs_valid]

                else:
                    ## coordinates over whole image
                    ##projs_valid = (self.projs[index] > 0).flatten()
                    ##coords_valid = self.coords[projs_valid]
                    projs_valid = (self.projs[index] > 0)
                    H, W = projs_valid.shape
                    border_mask = torch.zeros((H, W), dtype=torch.bool, device=projs_valid.device)
                    border_mask[:, otp:W - otp] = True
                    final_mask = projs_valid & border_mask
                    coords_valid = self.coords[final_mask.flatten()]

                select_inds = np.random.choice(coords_valid.shape[0], size=[self.n_rays], replace=False)
                select_coords = coords_valid[select_inds].long()

                #here stack several rays and projs together (keep same index)
                rays = self.rays[proj_idx, select_coords[:, 0], select_coords[:, 1]]
                projs = data[proj_idx, select_coords[:, 0], select_coords[:, 1]]       #dont change!!
                list_rays.append(rays)
                list_proj.append(projs)

            out = {
                "projs":torch.stack(list_proj),
                "rays":torch.stack(list_rays),
            }


        # elif self.type == "val":
        #     breakpoint()
        #     rays = self.rays[proj_idx] #incorrect
        #     projs = data[proj_idx]
        #     out = {
        #         "projs":projs,
        #         "rays":rays,
        #     }
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
        #geoo.offDetector = tmp.reshape(1,1,2)
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
            #geoo.offDetector = np.zeros_like((tmp.reshape(tmp.shape[0],2))) + offset_nbr/1000

        if angle_more == False:
            rays = self.get_rays([self.angles[j]], geoo, j, device, mode='normal',save=True) #angle_zz
        else:
            rays = self.get_rays([self.angles_more[j]], geoo, j, device, mode='normal', save=True)
        return torch.cat([rays, torch.ones_like(rays[..., :1]) * self.near, torch.ones_like(rays[..., :1]) * self.far], dim=-1)


    def get_rays(self, angles, geo: ConeGeometry, jj, device, mode, save=False):
        """
        Get rays given one angle and x-ray machine geometry.
        """

        if mode == 'normal':
            #W, H = geo.nDetector
            #W, H = 512, 512  # subbatch
            H, W = geo.nDetector
        else:
            H,W = 512, 512 #145,230 #250,800 #560,560 #outpaint
        rays = []

        list_subbatches = []
        list_subbatches = []
        #angles1= angles[0:500]
        #angles2=angles[500:]

        indx=0
        for angle in angles:
            if jj!=None:
                DSD = geo.DSD[0][jj]
                DSO = geo.DSO[0][jj]

                goD0 = geo.offDetector[jj, 0]
                goD1 = geo.offDetector[jj, 1]
                #print(goD0,goD1, jj)
            else:
                DSD = geo.DSD[0][indx]
                DSO = geo.DSO[0][indx]

                goD0 = geo.offDetector[indx, 0]
                goD1 = geo.offDetector[indx, 1]
                #print(goD0,goD1, indx)

            pose = torch.Tensor(self.angle2pose(DSO, angle)).to(device)
            rays_o, rays_d = None, None
            if geo.mode == "cone":

                i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                    torch.linspace(0, H - 1, H, device=device), indexing="ij")  # pytorch"s meshgrid has indexing="ij"
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + goD0
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + goD1
                dirs = torch.stack([uu / DSD, vv / DSD, torch.ones_like(uu)], -1)
                rays_d = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1) # pose[:3, :3] *
                rays_o = pose[:3, -1].expand(rays_d.shape)
            elif geo.mode == "parallel":

                # for t in range(0,4):
                #     list_subbatches.append(t)


                    i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                                torch.linspace(0, H - 1 , H, device=device), indexing="ij")  # pytorch"s meshgrid has indexing="ij"

                    uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + goD0
                    vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + goD1
                    dirs = torch.stack([torch.zeros_like(uu), torch.zeros_like(uu), torch.ones_like(uu)], -1)
                    rays_d = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1) # pose[:3, :3] *
                    rays_o = torch.sum(torch.matmul(pose[:3,:3], torch.stack([uu,vv,torch.zeros_like(uu)],-1)[..., None]).to(device), -1) + pose[:3, -1].expand(rays_d.shape)


            rays.append(torch.concat([rays_o, rays_d], dim=-1))
            indx = indx + 1

        #return np.swapaxes(torch.stack(rays, dim=0),1,2)

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

        trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])
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
