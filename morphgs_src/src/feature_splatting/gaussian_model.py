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
import os

import sys
knn_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../extlibs/simple-knn"))
if knn_path not in sys.path:
    sys.path.insert(0, knn_path)
from pytorch3d.ops import knn_points  # replaces Inria simple-knn distCUDA2 (non-commercial license)

from .utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from .utils.sh_utils import RGB2SH
from .utils.graphics_utils import BasicPointCloud
from .utils.general_utils import strip_symmetric, build_scaling_rotation

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling,
                                                   scaling_modifier,
                                                   rotation):
            """
            Build covariance matrix from scaling and rotation.

            scaling: (N, 3); vanilla scaling array from GD
            rotation: (N, 4); vanilla rotation array from GD
            rotation_idx: (M, ); optional indices of points to be rotated from vanilla view
            rotation_mat: (3, 3); optional rotation matrices of points to be rotated from vanilla view
            """
            L = build_scaling_rotation(scaling_modifier * scaling,
                                       rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, distill_feature_dim : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self.distill_feature_dim = distill_feature_dim
        self._xyz = torch.empty(0)  # (N, 3)
        # self.get_features contains both features_dc and features_rest
        self._features_dc = torch.empty(0)  # (N, 8, 3)
        self._features_rest = torch.empty(0)  # (N, 8 3)
        self._scaling = torch.empty(0)  # (N, 3); pass through sigmoid before return
        self._rotation = torch.empty(0)  # (N, 4); quaternion; pass through L2 normalization before return
        self._opacity = torch.empty(0)  # (N, 1)
        self._distill_features = torch.empty(0)  # (N, distill_feature_dim)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._distill_features,
            self._skinning_weights,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.spatial_lr_scale,
            self.ivtx,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._distill_features,
        self._skinning_weights,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        self.spatial_lr_scale,
        self.ivtx,
        ) = model_args

        self.training_setup(training_args)

        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    def get_rotation_bias(self, rotation_bias=None):
        rotation_bias = rotation_bias if rotation_bias is not None else 0.
        return self.rotation_activation(self._rotation + rotation_bias)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_distill_features(self):
        """
        Get view-invariant features for each point.
        """
        return self._distill_features
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_skinning_weights(self):
        return self._skinning_weights
    
    def get_covariance(self, scaling_modifier = 1,  d_rotation=None, gs_rot_bias=None):
        if d_rotation is not None:
            rotation = quaternion_multiply(self._rotation, d_rotation)
        else:
            rotation = self._rotation
        if gs_rot_bias is not None:
            rotation = rotation / rotation.norm(dim=-1, keepdim=True)
            rotation = quaternion_multiply(gs_rot_bias, rotation)
        return self.covariance_activation(self.get_scaling, scaling_modifier, rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, skinning_weights, distill_features=None, opacity_init=0.1):
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        pts_cuda = torch.from_numpy(np.asarray(pcd.points)).float().cuda().unsqueeze(0)
        knn_result = knn_points(pts_cuda, pts_cuda, K=4)  # K=4: nearest neighbor is the point itself (dist 0)
        dist2 = torch.clamp_min(knn_result.dists[0, :, 1:4].mean(dim=1), 0.0000001)  # mean sq-dist to 3 nearest neighbors, matching distCUDA2 semantics
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        init_opacity = float(opacity_init)
        init_opacity = min(max(init_opacity, 1e-6), 1.0 - 1e-6)
        opacities = inverse_sigmoid(init_opacity * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        if distill_features is not None:
            self._distill_features = nn.Parameter(distill_features.to("cuda"), requires_grad=False)
        else:
            self._distill_features = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], self.distill_feature_dim), device="cuda").requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self._skinning_weights = nn.Parameter(torch.tensor(skinning_weights, device="cuda").requires_grad_(True))

        # hold indices for dense keypoint mapping
        self.ivtx = torch.arange(0, fused_point_cloud.shape[0], device="cuda", dtype=torch.long).unsqueeze(-1)


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self._base_lrs = {
            'xyz': training_args.position_lr_init * self.spatial_lr_scale,
            'f_dc': training_args.feature_lr,
            'f_rest': training_args.feature_lr / 20.0,
            'opacity': training_args.opacity_lr,
            'scaling': training_args.scaling_lr,
            'rotation': training_args.rotation_lr,
            'distill_features': training_args.distill_lr,
            'skinning_weights': training_args.shape_sw_lr,
        }
        self._start_iters = {
            'xyz': getattr(training_args, 'xyz_start_iteration', 0),
            'f_dc': getattr(training_args, 'features_dc_start_iteration', 0),
            'f_rest': getattr(training_args, 'features_rest_start_iteration', 0),
            'opacity': getattr(training_args, 'opacity_start_iteration', 0),
            'scaling': getattr(training_args, 'scaling_start_iteration', 0),
            'rotation': getattr(training_args, 'rotation_start_iteration', 0),
            'distill_features': getattr(training_args, 'distill_features_start_iteration', 0),
            'skinning_weights': getattr(training_args, 'skinning_weights_start_iteration', 0),
        }

        l = [
            {'params': [self._xyz], 'lr': 0, "name": "xyz"},
            {'params': [self._features_dc], 'lr': 0, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': 0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': 0, "name": "opacity"},
            {'params': [self._scaling], 'lr': 0, "name": "scaling"},
            {'params': [self._rotation], 'lr': 0, "name": "rotation"},
            {'params': [self._distill_features], 'lr': 0, "name": "distill_features"},
            {'params': [self._skinning_weights], 'lr': 0, "name": "skinning_weights"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self._freeze_flags = {
            'xyz': getattr(training_args, 'freeze_xyz', False),
            'opacity': getattr(training_args, 'freeze_opacity', False),
            'scaling': getattr(training_args, 'freeze_scaling', False),
            'rotation': getattr(training_args, 'freeze_rotation', False),
            'f_dc': getattr(training_args, 'freeze_features_dc', False),
            'f_rest': getattr(training_args, 'freeze_features_rest', False),
            'distill_features': getattr(training_args, 'freeze_distill_features', False),
            'skinning_weights': True,
        }
        frozen_params = [k for k, v in self._freeze_flags.items() if v]
        if frozen_params:
            print(f"Frozen parameters: {', '.join(frozen_params)}")

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.feature_scheduler_args = get_expon_lr_func(lr_init=0.0025,
                                                        lr_final=5e-4,
                                                        max_steps=10000)
        
        self.update_features_until_iter = training_args.update_features_until_iter
    
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        xyz_lr = 0
        for param_group in self.optimizer.param_groups:
            name = param_group["name"]

            if self._freeze_flags.get(name, False):
                param_group['lr'] = 0
                continue

            start_iter = self._start_iters.get(name, 0)
            if iteration < start_iter:
                param_group['lr'] = 0
                continue

            if name == "xyz":
                effective_iter = max(0, iteration - start_iter)
                lr = self.xyz_scheduler_args(effective_iter)
                param_group['lr'] = lr
                xyz_lr = lr
            else:
                param_group['lr'] = self._base_lrs.get(name, 0)

        return xyz_lr

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                if stored_state is not None:
                    self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._distill_features = optimizable_tensors["distill_features"]
        self._skinning_weights = optimizable_tensors["skinning_weights"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.ivtx = self.ivtx[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_distill_features, new_skinning_weights, new_ivtx):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling" : new_scaling,
            "rotation" : new_rotation,
            "distill_features" : new_distill_features,
            "skinning_weights" : new_skinning_weights,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._distill_features = optimizable_tensors["distill_features"]
        self._skinning_weights = optimizable_tensors["skinning_weights"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.ivtx = torch.cat((self.ivtx, new_ivtx), dim=0)


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_distill_features = self._distill_features[selected_pts_mask].repeat(N,1)
        new_skinning_weights = self._skinning_weights[selected_pts_mask].repeat(N,1)
        new_ivtx = self.ivtx[selected_pts_mask].repeat(N,1)

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_distill_features, new_skinning_weights, new_ivtx)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_distill_features = self._distill_features[selected_pts_mask]
        new_skinning_weights = self._skinning_weights[selected_pts_mask]
        new_ivtx = self.ivtx[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_distill_features, new_skinning_weights, new_ivtx)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
