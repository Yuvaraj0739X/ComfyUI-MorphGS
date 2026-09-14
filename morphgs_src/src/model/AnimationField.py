# Copyright (c) 2026 MorphGS Authors.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils.articulation_utils import calc_rec_abs_T_fast, matrix_to_quaternion, Rodrigues


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

class SimpleDeformNet(nn.Module):
    def __init__(
        self,
        bone_num,
        time_input_ch=13,
        D=8,
        W=256,
        multires=8,
        device='cuda',
    ):
        super(SimpleDeformNet, self).__init__()
        self.D = D
        self.W = W
        self.multires = multires
        self.skips = [D // 2]

        # Time embedding network.
        self.time_out = 30
        self.timenet = nn.Sequential(
            nn.Linear(time_input_ch, W), nn.ReLU(inplace=True),
            nn.Linear(W, W), nn.ReLU(inplace=True),
            nn.Linear(W, self.time_out)
        ).to(device)

        self.out_dim = bone_num * 3
        self.joint_num = bone_num  
        
        # MLP trunk of depth D with skip connections.
        self.net = nn.ModuleList(
            [nn.Linear(self.time_out, W)] + 
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.time_out, W) for i in range(D-1)]
            ).to(device)

        self.global_translation = nn.Linear(W, 3).to(device)
        self.joint_rotation = nn.Linear(W, self.out_dim).to(device)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.uniform_(self.joint_rotation.weight, 1e-6, 1e-6)
        torch.nn.init.constant_(self.joint_rotation.bias, 0)
    
    def forward(self, t_emb):
        B = t_emb.shape[0]
        t_emb = self.timenet(t_emb)

        # Joint Estimation
        h = t_emb
        for i, l in enumerate(self.net):
            h = self.net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([t_emb, h], -1)
        
        d_global_translation = self.global_translation(h).view(B, -1, 3)  # [B, 1, 3]
        rot_params = self.joint_rotation(h).view(B, -1, 3)  # [B, J, 3]

        # rot_params are axis-angle vectors; do not normalize them.
        return d_global_translation, rot_params
    
class Animation():
    def __init__(self, joint_num, opt_cfg):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.skinning_method = str(getattr(opt_cfg, 'skinning_method', 'lbs')).lower()
        if self.skinning_method not in ('lbs', 'dqs'):
            print(f"[Warning] Unknown skinning_method={self.skinning_method}. Falling back to 'lbs'.")
            self.skinning_method = 'lbs'
        self.joint_num = joint_num

        self.embed_time_fn, time_input_ch = get_embedder(6, 1)
        self.deform_net = SimpleDeformNet(
            joint_num,
            time_input_ch,
        )

        head_parameters = list(self.deform_net.global_translation.parameters()) + \
                        list(self.deform_net.joint_rotation.parameters())
        head_param_ids = {id(p) for p in head_parameters}  # Convert to a set of IDs

        # Optimizer setup
        self.optimizer_by_video = torch.optim.Adam([
            {"params": [p for p in self.deform_net.parameters() if id(p) not in head_param_ids], "lr": opt_cfg.motion_lr}
        ])
        frame_groups = [
            {"params": self.deform_net.global_translation.parameters(), "lr": opt_cfg.global_t_lr},
            {"params": self.deform_net.joint_rotation.parameters(), "lr": opt_cfg.joint_rot_lr},
        ]
        self.optimizer_by_frame = torch.optim.Adam(frame_groups)
        
        self.initial_lrs_video = [pg['lr'] for pg in self.optimizer_by_video.param_groups]
        self.initial_lrs_frame = [pg['lr'] for pg in self.optimizer_by_frame.param_groups]
        self.total_iters = opt_cfg.iterations
        self._iter = 0
        lr_final_ratio = getattr(opt_cfg, "lr_final_ratio", 0.1)
        gamma = lr_final_ratio ** (1.0 / max(self.total_iters, 1))
        self.sched_video = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_by_video, gamma=gamma)
        self.sched_frame = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_by_frame, gamma=gamma)

    def save_params(self, save_path):
        """
        Saves the model's parameters and optimizer state.
        
        Args:
            filename (str): The name of the file to save the parameters.
        """
        checkpoint = {
            'deform_net_state_dict': self.deform_net.state_dict(),
            'optimizer_by_video_state_dict': self.optimizer_by_video.state_dict(),
            'optimizer_by_frame_state_dict': self.optimizer_by_frame.state_dict()
        }
        torch.save(checkpoint, save_path)
        print(f"Model parameters saved to {save_path}")

    def load_params(self, load_path):
        """
        Loads the model's parameters and optimizer state from a checkpoint.
        
        Args:
            filename (str): The name of the file from which to load the parameters.
        """
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path)
            self.deform_net.load_state_dict(checkpoint['deform_net_state_dict'])
            try:
                self.optimizer_by_video.load_state_dict(checkpoint['optimizer_by_video_state_dict'])
                self.optimizer_by_frame.load_state_dict(checkpoint['optimizer_by_frame_state_dict'])
            except Exception as e:
                print(f"Optimizer state is not fully compatible. Skipping optimizer load: {e}")
            print(f"Model parameters loaded from {load_path}")
        else:
            print(f"Checkpoint {load_path} not found.")

    def step_lr(self):
        """Call once per iteration after optimizer.step()."""
        self._iter += 1
        if self._iter <= self.total_iters:
            self.sched_video.step()
            self.sched_frame.step()

    @staticmethod
    def _quat_mul(a, b):
        """Hamilton product for quaternions in [w, x, y, z] convention."""
        aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return torch.stack([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ], dim=-1)

    @classmethod
    def _dqs_deform(cls, xyz_orig, bone_Ts, weights, return_rotation_quat=False):
        """
        DQS deformation ported from extlibs/pynocchio-0.0.4/auto_rig_skinning.py.
        """
        if bone_Ts.dim() == 3:  # (J, 4, 4)
            bone_Ts = bone_Ts.unsqueeze(0).expand(xyz_orig.shape[0], -1, -1, -1)

        V, J = weights.shape
        device = xyz_orig.device
        dtype = xyz_orig.dtype

        R = bone_Ts[..., :3, :3]  # (V, J, 3, 3)
        t = bone_Ts[..., :3, 3]   # (V, J, 3)

        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

        qw = torch.zeros(V, J, device=device, dtype=dtype)
        qx = torch.zeros(V, J, device=device, dtype=dtype)
        qy = torch.zeros(V, J, device=device, dtype=dtype)
        qz = torch.zeros(V, J, device=device, dtype=dtype)

        m1 = trace > 0
        s1 = torch.sqrt((trace + 1.0).clamp(min=1e-10)) * 2.0
        qw = torch.where(m1, 0.25 * s1, qw)
        qx = torch.where(m1, (R[..., 2, 1] - R[..., 1, 2]) / s1.clamp(min=1e-10), qx)
        qy = torch.where(m1, (R[..., 0, 2] - R[..., 2, 0]) / s1.clamp(min=1e-10), qy)
        qz = torch.where(m1, (R[..., 1, 0] - R[..., 0, 1]) / s1.clamp(min=1e-10), qz)

        m2 = (~m1) & (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2])
        s2 = torch.sqrt((1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2]).clamp(min=1e-10)) * 2.0
        qw = torch.where(m2, (R[..., 2, 1] - R[..., 1, 2]) / s2.clamp(min=1e-10), qw)
        qx = torch.where(m2, 0.25 * s2, qx)
        qy = torch.where(m2, (R[..., 0, 1] + R[..., 1, 0]) / s2.clamp(min=1e-10), qy)
        qz = torch.where(m2, (R[..., 0, 2] + R[..., 2, 0]) / s2.clamp(min=1e-10), qz)

        m3 = (~m1) & (~m2) & (R[..., 1, 1] > R[..., 2, 2])
        s3 = torch.sqrt((1.0 + R[..., 1, 1] - R[..., 0, 0] - R[..., 2, 2]).clamp(min=1e-10)) * 2.0
        qw = torch.where(m3, (R[..., 0, 2] - R[..., 2, 0]) / s3.clamp(min=1e-10), qw)
        qx = torch.where(m3, (R[..., 0, 1] + R[..., 1, 0]) / s3.clamp(min=1e-10), qx)
        qy = torch.where(m3, 0.25 * s3, qy)
        qz = torch.where(m3, (R[..., 1, 2] + R[..., 2, 1]) / s3.clamp(min=1e-10), qz)

        m4 = (~m1) & (~m2) & (~m3)
        s4 = torch.sqrt((1.0 + R[..., 2, 2] - R[..., 0, 0] - R[..., 1, 1]).clamp(min=1e-10)) * 2.0
        qw = torch.where(m4, (R[..., 1, 0] - R[..., 0, 1]) / s4.clamp(min=1e-10), qw)
        qx = torch.where(m4, (R[..., 0, 2] + R[..., 2, 0]) / s4.clamp(min=1e-10), qx)
        qy = torch.where(m4, (R[..., 1, 2] + R[..., 2, 1]) / s4.clamp(min=1e-10), qy)
        qz = torch.where(m4, 0.25 * s4, qz)

        q_r = torch.stack([qw, qx, qy, qz], dim=-1)  # (V, J, 4)
        t_q = torch.cat([torch.zeros(V, J, 1, device=device, dtype=dtype), t], dim=-1)
        q_d = 0.5 * cls._quat_mul(t_q, q_r)

        q_r0 = q_r[:, 0:1, :]
        signs = torch.sign((q_r * q_r0).sum(dim=-1, keepdim=True))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        q_r = q_r * signs
        q_d = q_d * signs

        w = weights.unsqueeze(-1)
        blended_r = (w * q_r).sum(dim=1)
        blended_d = (w * q_d).sum(dim=1)

        norm_r = blended_r.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        blended_r = blended_r / norm_r
        blended_d = blended_d / norm_r

        q_r_conj = blended_r * torch.tensor([1, -1, -1, -1], device=device, dtype=dtype)
        t_out = 2.0 * cls._quat_mul(blended_d, q_r_conj)[..., 1:]

        p_q = torch.cat([torch.zeros(V, 1, device=device, dtype=dtype), xyz_orig], dim=-1)
        rotated = cls._quat_mul(cls._quat_mul(blended_r, p_q), q_r_conj)[..., 1:]
        xyz_deformed = rotated + t_out

        if return_rotation_quat:
            return xyz_deformed, blended_r
        return xyz_deformed

    def step(self, xyz, joints, weights, rig, t, rot_params=None, global_transl_only=False,
             global_translation=None, precomp_joints=None, scale_factor=1.0, fixed_joints=None, iteration=None):
            
        # 1. Given time, estimate the bone rotation field.
        self.t = t
        t_embed = self.embed_time_fn(t.unsqueeze(0))  # [1, 13]
        if rot_params is None:  # main case: infer motion from time
            global_t, rot_params = self.deform_net(t_embed)
            global_t = global_t[0]  # batch size is 1
            rot_params = rot_params[0]  # [Nj, 3] axis-angle rotation parameters

            R_t, self.prev_thetas = Rodrigues(rot_params)
            self.prev_global_t = global_t
            self.prev_rot_params = rot_params

        else:
            if rot_params.dim() == 2:
                if rot_params.shape[-1] != 3:
                    raise ValueError(f"Expected axis-angle rotation params with shape [J, 3], got {tuple(rot_params.shape)}")
                R_t, self.prev_thetas = Rodrigues(rot_params)
            elif rot_params.dim() == 3 and rot_params.shape[-2:] == (3, 3):
                # already rotation matrices [Nj, 3, 3]
                R_t = rot_params
                self.prev_thetas = None
            else:
                raise ValueError(f"Expected rotation params [J, 3] or rotation matrices [J, 3, 3], got {tuple(rot_params.shape)}")
            global_t = global_translation
            self.prev_global_t = global_t

        # Rig-authored fixed joints keep identity rotation while descendants remain free.
        if fixed_joints is not None:
            identity = torch.eye(3, device=R_t.device, dtype=R_t.dtype)
            for j_idx in fixed_joints:
                R_t[j_idx] = identity

        if global_transl_only:
            # Keep only root rotation active before local pose stage.
            _R_t = torch.eye(3, device=R_t.device, dtype=R_t.dtype).unsqueeze(0).repeat(R_t.shape[0], 1, 1)
            _R_t[0] = R_t[0]
        else:
            _R_t = R_t

        bone_Ts = calc_rec_abs_T_fast(
            _R_t,
            joints,
            rig.parent_joint_ex,
            rig.parent_indices,
            pivot_mode=getattr(rig, "fk_pivot_mode", "parent"),
        ) # [J, 4, 4]
        jointsh = torch.concat([joints, torch.ones((len(joints), 1), device=joints.device)], axis=-1)
        jointsh = torch.bmm(bone_Ts, jointsh.unsqueeze(-1)).squeeze(-1)
        if self.skinning_method == 'dqs':
            xyz_skinned, d_rotations = self._dqs_deform(xyz, bone_Ts, weights, return_rotation_quat=True)
            xyzh = torch.concat([xyz_skinned, torch.ones((len(xyz_skinned), 1), device=xyz_skinned.device)], axis=-1)
        else:
            weighted_G_tw = (bone_Ts * weights[:, :, None, None]).sum(dim=1)
            xyzh = torch.concat([xyz, torch.ones((len(xyz), 1), device=xyz.device)], axis=-1)
            xyzh = torch.bmm(weighted_G_tw, xyzh.unsqueeze(-1)).squeeze(-1)
            d_rotations = matrix_to_quaternion(weighted_G_tw[:, :3, :3])

        if global_t is None:
            global_t = torch.zeros(3, dtype=torch.float32, device=_R_t.device)
            self.prev_global_t = global_t

        _xyz = xyzh[:,:3] + global_t * scale_factor

        joints_warped_rel = jointsh[:,:3]
        joints_warped = joints_warped_rel + global_t

        ret_value = {'xyz': _xyz}
        ret_value['joints_warped'] = joints_warped
        ret_value['global_t'] = global_t
        ret_value['rot_params'] = rot_params
        ret_value['d_rotation'] = d_rotations
        ret_value['bone_rotation'] = _R_t

        return ret_value
    
    def get_twist_regularisation_loss(self, rig):
        """
        Penalize the component of each joint's axis-angle vector that lies along
        its parent bone direction. This is a lightweight twist suppressor that
        leaves bend freedom intact while discouraging screw-like motion.
        """
        if self.prev_rot_params is None:
            return torch.tensor(0.0, device=self.device)

        rot_params = self.prev_rot_params
        if rot_params.shape[-1] != 3:
            return torch.tensor(0.0, device=rot_params.device, dtype=rot_params.dtype)

        joint_count = min(rot_params.shape[0], len(rig.joints_pos))
        if joint_count <= 1:
            return torch.tensor(0.0, device=rot_params.device, dtype=rot_params.dtype)

        bone_axes = []
        rot_subset = []
        parent_of = {child: parent for parent, child in rig.bones}
        for child_idx in range(1, joint_count):
            parent_idx = parent_of.get(child_idx, -1)
            if parent_idx < 0:
                continue
            bone_vec = rig.joints_pos[child_idx] - rig.joints_pos[parent_idx]
            bone_len = torch.linalg.norm(bone_vec)
            if float(bone_len.item()) < 1e-8:
                continue
            bone_axes.append(bone_vec / bone_len)
            rot_subset.append(rot_params[child_idx])

        if not bone_axes:
            return torch.tensor(0.0, device=rot_params.device, dtype=rot_params.dtype)

        bone_axes = torch.stack(bone_axes, dim=0).to(device=rot_params.device, dtype=rot_params.dtype)
        rot_subset = torch.stack(rot_subset, dim=0)
        twist_component = (rot_subset * bone_axes).sum(dim=-1)
        return (twist_component ** 2).mean()

    def get_transformation_regularisation_loss(self):
        """
            Suppress the model from learning too much rotation.
        """
        t = self.prev_global_t.abs()
        thetas = self.prev_thetas[1:].abs()
        return (torch.abs(t).sum() + thetas.sum()) / len(thetas + 1)

    def get_smoothness_loss(self, NF):
        """
            Suppress the model from learning too much rotation.
        """
        if self.t == 0:
            return 0
        prev_t = (self.t*NF - 1) / NF
        prev_t_embed = self.embed_time_fn(prev_t.unsqueeze(0))
        prev_global_t, prev_rot_params = self.deform_net(prev_t_embed)

        prev_global_t = prev_global_t[0]
        prev_rot_params = prev_rot_params[0]
        prev_global_t = prev_global_t.detach()
        prev_rot_params = prev_rot_params.detach()

        # Calculate the difference between the two rotations
        _, _theta = Rodrigues(prev_rot_params)
        theta_diff = torch.abs(self.prev_thetas - _theta)

        # Calculate the difference between the two translations
        translation_diff = torch.abs(self.prev_global_t - prev_global_t)
        translation_diff = translation_diff.sum(dim=-1)
        
        rot_param_diff = torch.abs(self.prev_rot_params - prev_rot_params)
        rot_param_diff = rot_param_diff.sum(dim=-1)

        translation_diff = translation_diff.mean()
        theta_diff = theta_diff.mean()
        rot_param_diff = rot_param_diff.mean()
        
        return translation_diff + theta_diff + rot_param_diff

    def update_learning_rate(self, iteration):
        # Backward compatibility for old training loops.
        return
