# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from .utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 feat_path=None, feat_chw=None, dino_feat_path=None, dino_feat_chw=None,
                 mask=None,
                 part_feat_chw=None, thinned=None, mask_path=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.feat_path = feat_path
        self.feat_chw = feat_chw
        self.mask = mask
        self.mask_path = mask_path

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        
        if self.mask is not None and self.mask.sum() > 0:
            self.mask_valid_flag = True
        else:
            self.mask_valid_flag = False
        
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        if gt_alpha_mask is not None:
            self.foreground_mask = gt_alpha_mask.to(self.data_device)

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.thinned = None
        if thinned is not None:
            mask = thinned[0] != 0    
            ys, xs = torch.where(mask) 
            self.thinned = torch.stack([xs, ys], dim=1).unsqueeze(0).to(self.data_device)    # (1, px, 2)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.intrinsic = get_intrinsic(self.image_width, self.image_height, self.FoVx, self.FoVy)
        self.extrinsic = create_transformation_matrix(self.R, self.T)

    
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


def get_intrinsic(W, H, FoVx, FoVy):
    # Compute focal lengths
    focal_x = W / (2 * np.tan(FoVx / 2))
    focal_y = H / (2 * np.tan(FoVy / 2))
    
    return torch.tensor(
            [[focal_x,   0.0000, 0.5*W],
            [  0.0000, focal_y, 0.5*H],
            [  0.0000,   0.0000,   1.0000]], dtype=torch.float32)


def create_transformation_matrix(R, T):
    """Build a 4x4 camera-to-world extrinsic matrix from COLMAP-style R, T."""
    matrix_R = np.transpose(-R)
    matrix_T = -T 
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = matrix_R
    extrinsic[:3, 3] = matrix_T

    extrinsic = np.linalg.inv(extrinsic)

    return torch.from_numpy(extrinsic).to('cuda').float()