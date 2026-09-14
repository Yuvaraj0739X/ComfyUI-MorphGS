# Copyright (c) 2026 MorphGS Authors.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import os
from pykeops.torch import LazyTensor
import numpy as np
import copy

from model.RigModel import Rig

from feature_splatting.gaussian_model import GaussianModel
from feature_splatting.utils.graphics_utils import BasicPointCloud
from feature_splatting.utils.sh_utils import SH2RGB

from utils.articulation_utils import calc_skinning_weights
from utils.mesh_utils import fps_pointcloud

class ParametricModel():
    def __init__(self, rig: Rig, gaussian_params, lbs_parts, dominant_joints, model_cfg, device='cuda', init_gaussian=None):
        self.device = device
        self.eps = 1e-9
        self.create_from_ellipsoid = lbs_parts is None
        self.opt_cfg = model_cfg.opt
        self.skinning_apply_softmax = False

        # Initialize Gaussian model.
        # mu_offset is set as offset to the blended joint position.
        xyz, skinning_weights, features = gaussian_params[0], gaussian_params[1], gaussian_params[2]
        if len(gaussian_params) > 3:
            pcd_colors = gaussian_params[3]

        sw_processed = calc_skinning_weights(rig.joints_pos, skinning_weights, device=device)
        blended_joint_pos = torch.sum(sw_processed.unsqueeze(-1) * rig.joints_pos, dim=1)
        offset_xyz = xyz - blended_joint_pos
        self.initial_offset_xyz = offset_xyz.detach().clone()
        
        # Clamp offset_xyz to [0.5x, 2x] of its initial value.
        bound1 = offset_xyz.clone() * 0.5
        bound2 = offset_xyz.clone() * 2.0
        self.min_xyz = torch.minimum(bound1, bound2)
        self.max_xyz = torch.maximum(bound1, bound2)
    
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        
        gs_colors = pcd_colors.cpu().numpy() if len(gaussian_params) > 3 else SH2RGB(shs)
        
        points = BasicPointCloud(points=offset_xyz.cpu().numpy(), colors=gs_colors, normals=np.zeros((num_pts, 3)))

        if init_gaussian is None:
            self.gaussians = GaussianModel(sh_degree=3, distill_feature_dim=model_cfg.feature_dim)
            self.gaussians.create_from_pcd(
                points,
                1,
                skinning_weights=skinning_weights,
                distill_features=features,
                opacity_init=1.0,
            )
        else:
            self.gaussians = init_gaussian
            self.gaussians._skinning_weights = nn.Parameter(torch.tensor(skinning_weights, device="cuda").requires_grad_(True))
            self.gaussians._distill_features = nn.Parameter(features.to("cuda"), requires_grad=False)

        # for keypoint loss
        self.fps_points, self.fps_idx = fps_pointcloud(points.points, self.opt_cfg.fps_num)
        self.prev_gs_num = self.gaussians.get_xyz.shape[0]

        # Initialize bone lengths
        bone_directions = torch.zeros((len(rig.bones), 3), device=device)
        initial_bone_length = torch.zeros(len(rig.bones), device=device)
        for i, (parent, child) in enumerate(rig.bones):
            initial_bone_length[i] = torch.linalg.norm(rig.joints_pos[parent] - rig.joints_pos[child])
            bone_directions[i] = (rig.joints_pos[child] - rig.joints_pos[parent]) / initial_bone_length[i]

        self.bone_directions = bone_directions
        self.rig = copy.deepcopy(rig)
        self.rest_bone_length = initial_bone_length.detach().clone()
        self.delta_log_bone_length = nn.Parameter(torch.zeros_like(initial_bone_length), requires_grad=True)
        self._bone_length_parameter = self.delta_log_bone_length

        self.merging_mat = None
        self.theta_weight = torch.nn.Parameter(torch.tensor([0.1], device=self.device), requires_grad=False)
        self.xyz_nn_dist = None
        self.xyz_nn_i = None
        self.delta_log_scale = nn.Parameter(torch.zeros(1, device=device), requires_grad=True)
        self._scale_parameter = self.delta_log_scale

        # Joint tree for recursive kinematics (must be ready before get_xyz is called)
        self.children = [[] for _ in range(len(self.rig.joints_pos))]
        self.parent_of = [-1 for _ in range(len(self.rig.joints_pos))]
        for parent, child in self.rig.bones:
            self.children[parent].append(child)
            self.parent_of[child] = parent

        # Initialize relationship with the Gaussian model
        if not self.create_from_ellipsoid:
            self.joint_indicies = dominant_joints
            
            xyz = self.get_xyz.clone().detach()

            # Initialize ARAP nearest neighbours
            neighbor_num = self.opt_cfg.arap_nn_num
            xyz1 = LazyTensor(xyz[:, None, :], )
            xyz2 = LazyTensor(xyz[None, :, :])
            D_ij = ((xyz1 - xyz2) ** 2).sum(-1)
            self.xyz_nn_i = D_ij.argKmin(dim=1, K=neighbor_num).to(self.device)
            self.xyz_nn_dist = torch.sqrt(((xyz[:,None,:] - xyz[self.xyz_nn_i,:])**2).sum(-1) + self.eps).to(self.device)

            # ARAP's K-nearest-neighbours are chosen by pure rest-pose 3D distance, with no
            # awareness of which bone/joint each point belongs to. For spatially-packed but
            # kinematically-independent parts (fingers being the clearest case: adjacent
            # fingers sit millimeters apart in the rest pose despite moving independently),
            # this makes ARAP penalize a point for moving differently from a neighbor that is
            # actually a different rigid part -- directly suppressing the independent
            # per-finger motion the correspondence signal is trying to teach. Build a mask
            # that keeps a neighbor pair only if the two points share the same dominant joint,
            # or their dominant joints are directly parent-child in the skeleton (a legitimate
            # smooth transition across a joint, e.g. forearm-to-hand) -- and drops pairs
            # between unrelated/sibling parts (e.g. index finger vs. middle finger, both
            # children of the hand) that pure spatial proximity would otherwise link.
            parent_of_t = torch.tensor(self.parent_of, device=self.device, dtype=torch.long)
            dom = dominant_joints.to(self.device)
            dom_i = dom.unsqueeze(1).expand(-1, neighbor_num)          # (N, K)
            dom_j = dom[self.xyz_nn_i]                                  # (N, K)
            same_part = dom_i == dom_j
            parent_child = (parent_of_t[dom_i] == dom_j) | (parent_of_t[dom_j] == dom_i)
            self.arap_valid_mask = (same_part | parent_child).to(self.device)

        self.optimizer = torch.optim.Adam([
            {'params': self._bone_length_parameter, 'lr': self.opt_cfg.shape_bl_lr, 'name': 'bone_length'},
            {'params': self._scale_parameter, 'lr': self.opt_cfg.shape_scale_lr, 'name': 'scale'},
        ])
    

    @property
    def get_joints_pos(self):
        """
            return updated joint positions according to the bone lengths
        """
        joints_pos = torch.zeros((len(self.rig.joints_pos), 3), device=self.device)
        joints_pos[self.rig.root_idx] = self.rig.joints_pos[self.rig.root_idx] # root
        bone_length = self.bone_length
        edge_lookup = {(parent, child): i for i, (parent, child) in enumerate(self.rig.bones)}
        queue = [self.rig.root_idx]
        while queue:
            parent = queue.pop(0)
            for child in self.children[parent]:
                edge_idx = edge_lookup[(parent, child)]
                joints_pos[child] = joints_pos[parent] + self.bone_directions[edge_idx] * bone_length[edge_idx]
                queue.append(child)
        return joints_pos
    
    @property
    def get_skinning_weights(self):
        """
            return skinning weights updated according to the theta weight
        """
        return calc_skinning_weights(
            self.rig.joints_pos,
            self.gaussians.get_skinning_weights,
            self.theta_weight,
            self.merging_mat,
            self.eps,
            self.device,
            apply_softmax=self.skinning_apply_softmax,
        )
    
    @property
    def get_xyz(self):
        """
            return ALG position according to the updated bone length
        """
        offset_xyz = self.gaussians.get_xyz
        joint_pos = self.get_joints_pos  # apply the changed bone length
        
        # Clamp with dynamic index mapping after densify/prune.
        # min/max bounds are stored on the initial point set; ivtx maps current gaussians to that set.
        if offset_xyz.shape[0] == self.min_xyz.shape[0]:
            min_xyz = self.min_xyz
            max_xyz = self.max_xyz
        elif hasattr(self.gaussians, "ivtx") and self.gaussians.ivtx is not None:
            map_idx = self.gaussians.ivtx.squeeze(-1).long()
            min_xyz = self.min_xyz[map_idx]
            max_xyz = self.max_xyz[map_idx]
        else:
            # Fallback to avoid crashing if mapping is unavailable.
            min_xyz = self.min_xyz[:offset_xyz.shape[0]]
            max_xyz = self.max_xyz[:offset_xyz.shape[0]]
        offset_xyz = torch.clamp(offset_xyz, min_xyz, max_xyz)

        _skinning_weights = calc_skinning_weights(
            self.rig.joints_pos,
            self.gaussians.get_skinning_weights,
            self.theta_weight,
            self.merging_mat,
            self.eps,
            self.device,
            apply_softmax=False,
        ).detach()
        blended_joint_pos = torch.sum(_skinning_weights.unsqueeze(-1) * joint_pos, dim=1)
        xyz = blended_joint_pos + offset_xyz
        
        # Update ARAP nearest neighbours
        _xyz = xyz.clone().detach()
        if self.prev_gs_num != self.gaussians.get_xyz.shape[0]: # gaussian count changed; xyz_nn_i must be rebuilt
            neighbor_num = self.opt_cfg.arap_nn_num
            xyz1 = LazyTensor(_xyz[:, None, :], )
            xyz2 = LazyTensor(_xyz[None, :, :])
            D_ij = ((xyz1 - xyz2) ** 2).sum(-1)
            self.xyz_nn_i = D_ij.argKmin(dim=1, K=neighbor_num).to(self.device)

            self.prev_gs_num = self.gaussians.get_xyz.shape[0]

        # Refresh rest-pose NN distances; ARAP compares these against the posed NN distances.
        if self.xyz_nn_dist is not None:
            self.prev_xyz_nn_dist = self.xyz_nn_dist.clone().detach()
            self.xyz_nn_dist = torch.sqrt(((_xyz[:,None,:] - _xyz[self.xyz_nn_i,:])**2).sum(-1) + self.eps).to(self.device) 

        return xyz

    @property
    def bone_length(self):
        return self.rest_bone_length * torch.exp(self.delta_log_bone_length)

    @property
    def scale(self):
        return torch.exp(self.delta_log_scale)
    
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if bool(getattr(self.opt_cfg, "use_bone_adapt_lr_schedule", False)):
            bone_length_start_iter = int(getattr(self.opt_cfg, "bone_length_start_iter", 0))
            
            for param_group in self.optimizer.param_groups:
                name = param_group["name"]
                
                if name == "bone_length":
                    if iteration < bone_length_start_iter:
                        param_group["lr"] = 0.0
                    elif iteration < 500:
                        # Warm-up to a low peak (5e-4) to avoid overshooting
                        progress = iteration / 500.0
                        param_group["lr"] = 5e-4 * progress
                    elif iteration < 2000:
                        # Hold at peak
                        param_group["lr"] = 5e-4
                    elif iteration < 3500:
                        # First decay
                        param_group["lr"] = 1e-4
                    else:
                        # Final fine-tuning
                        param_group["lr"] = 5e-5

                elif name == "scale":
                    # Fit scale quickly early on, then freeze it so it does not
                    # interfere with bone-length optimization.
                    if iteration < 1000:
                        param_group["lr"] = 5e-3
                    elif iteration < 2000:
                        param_group["lr"] = 1e-3
                    else:
                        # Freeze scale after iteration 2000
                        param_group["lr"] = 0.0
            return

        bone_length_start_iter = int(getattr(self.opt_cfg, "bone_length_start_iter", 0))
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "bone_length":
                if iteration < bone_length_start_iter:
                    param_group['lr'] = 0.0
                elif iteration > self.opt_cfg.update_bl_decay_iter:
                    iter = iteration - self.opt_cfg.update_bl_decay_iter
                    lr = self.opt_cfg.shape_bl_lr * 0.1 ** (iter // self.opt_cfg.update_bl_decay_step)
                    param_group['lr'] = lr
                else:
                    param_group['lr'] = self.opt_cfg.shape_bl_lr
            if param_group["name"] == "scale":
                if iteration == self.opt_cfg.update_ts_until_iter:
                    param_group['lr'] = 0  

    def save_params(self, save_path):
        checkpoint = {
            'scale': self.scale.detach(),
            'bone_length': self.bone_length.detach(),
            'delta_log_scale': self.delta_log_scale.detach(),
            'delta_log_bone_length': self.delta_log_bone_length.detach(),
            'theta_weight': self.theta_weight,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, save_path)
        print(f"Model parameters saved to {save_path}") 
    
    def load_params(self, load_path):
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path)
            if 'delta_log_scale' in checkpoint and checkpoint['delta_log_scale'] is not None:
                self.delta_log_scale.data.copy_(checkpoint['delta_log_scale'].to(self.delta_log_scale.device))
            elif 'scale' in checkpoint:
                loaded_scale = torch.clamp(checkpoint['scale'].to(self.delta_log_scale.device), min=self.eps)
                self.delta_log_scale.data.copy_(torch.log(loaded_scale))

            if 'delta_log_bone_length' in checkpoint and checkpoint['delta_log_bone_length'] is not None:
                self.delta_log_bone_length.data.copy_(checkpoint['delta_log_bone_length'].to(self.delta_log_bone_length.device))
            elif 'bone_length' in checkpoint:
                loaded_bl = torch.clamp(checkpoint['bone_length'].to(self.delta_log_bone_length.device), min=self.eps)
                ratio = loaded_bl / self.rest_bone_length.clamp(min=self.eps)
                self.delta_log_bone_length.data.copy_(torch.log(ratio))

            if 'theta_weight' in checkpoint:
                self.theta_weight.data.copy_(checkpoint['theta_weight'].to(self.theta_weight.device))

            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"Optimizer state is not fully compatible. Skipping optimizer load: {e}")
            print(f"Model parameters loaded from {load_path}")
            return
        else:
            print(f"Checkpoint {load_path} not found.")
        
    def get_arap_loss(self, warped_pcd):
        """
        Computes the As-Rigid-As-Possible (ARAP) loss.

        Only averages over neighbor pairs that pass arap_valid_mask (same dominant joint, or
        directly parent-child in the skeleton) -- see the mask's construction in __init__ for
        why: unfiltered spatial-only neighbors wrongly link kinematically-independent but
        spatially-close parts (e.g. adjacent fingers), suppressing their independent motion.
        """
        self.get_xyz # by calling this, self.xyz_nn_i and self.xyz_nn_dist will be updated
        warped_nn_distance = torch.sqrt((warped_pcd[:,None,:] - warped_pcd[self.xyz_nn_i,:]).pow(2).sum(-1) + self.eps)
        per_pair_loss = torch.log1p((self.xyz_nn_dist - warped_nn_distance).abs())
        mask = self.arap_valid_mask
        return (per_pair_loss * mask).sum() / mask.sum().clamp_min(1)

    def sampling_skeleton_points(self, joints, num_sample=512):
        points_c = joints[1:] # Exclude root. N, 3
        parents = self.rig.parents
        points_p = joints[parents[1:]] # N, 3
        
        distance = (points_c - points_p).norm(dim=-1).detach()
        each_distance = distance.sum()/num_sample 
        max_distance = distance.max() 
        
        t = torch.linspace(0,1,int(max_distance/each_distance), device=points_c.device)[:,None,None]
        new_ps = t * joints[1:] + (1-t)*joints[parents[1:]]
        sampling_points = new_ps.reshape(-1,3)
        
        return sampling_points
