# Copyright (c) 2026 MorphGS Authors.
# Licensed under the MIT License.

import torch
from torch import nn
import os
import numpy as np
from utils import camera_utils, mesh_utils, general_utils, render_utils

from model.RigModel import Rig
from model.AnimationField import Animation
from model.ParametricModel import ParametricModel
import torch.optim as optim 
from tqdm import tqdm
from PIL import Image


def _parse_debug_frame_indices(spec: str):
    frame_indices = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            frame_indices.update(range(int(a), int(b) + 1))
        else:
            frame_indices.add(int(token))
    return sorted(frame_indices)

class Autoencoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=64):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class MorphGS():
    def __init__(self, proj_path, loaded_iter=0, device='cuda', model_cfg=None, orig_proj_path=None,
                 use_color=False, multi_stage=None,
                 render_mode=False,
                 load_deform_only=False,
        ):
        if orig_proj_path is None:
            orig_proj_path = proj_path
        self.device = device
        self.proj_path = proj_path
        tgt_name = os.path.basename(proj_path.tgt_dir)
        is_amass_like_target = tgt_name.startswith("smplh_")
        use_amass_rig_behavior = is_amass_like_target

        # Load cameras (source rendered cameras)
        # in_ndc True for full MorphGS, temporary patch for multiview experiment
        in_ndc = True if os.path.basename(os.path.dirname(proj_path.src_dir)) == "processed_videos" else False
        cameras_original = camera_utils.load_camera_params(proj_path, device=device, in_ndc=in_ndc)
        self.cameras, nerf_normalization = camera_utils.get_raster_cameras(
            cameras_original, proj_path,
            data_device=device,
            in_ndc=in_ndc
        )
        self.cameras_extent = nerf_normalization["radius"]
        self.cameras_p3d = cameras_original
        pretrain_inds = self.load_easiest_set()
        self.pretrain_cameras = [self.cameras[i] for i in pretrain_inds]
        print(f"✅ Loaded Cameras: {len(self.cameras)} cameras")

        # Load target mesh and initialize rig
        self.mesh = mesh_utils.load_mesh(proj_path.tgt_obj_path)
        vertices = torch.tensor(self.mesh.vertices, device=device, dtype=torch.float32)

        calculate_skinning_w = bool(getattr(model_cfg, "calculate_skinning_w", False))
        hybrid_skinning_w = bool(getattr(model_cfg, "hybrid_skinning_w", False))
        if calculate_skinning_w:
            print("✅ Recomputing target skinning weights from rig geometry")
        if hybrid_skinning_w:
            print("✅ Using hybrid target skinning weights")

        self.rig = Rig(
            self.mesh,
            proj_path.tgt_rig_path,
            calculate_skinning_w=calculate_skinning_w,
            hybrid_skinning_w=hybrid_skinning_w,
            hybrid_skinning_calculated_joints=getattr(model_cfg, "hybrid_skinning_calculated_joints", []),
            smooth_w=model_cfg.smooth_w,
            device=device,
            skinning_method=getattr(model_cfg.opt, "skinning_method", "lbs"),
            fk_pivot_mode="joint" if use_amass_rig_behavior else "parent",
        )

        print(f"✅ Loaded Target: {len(self.rig.joints_pos)} joints, {len(self.rig.bones)} bones")

        # Joints whose local rotation is forced to identity. Character-specific
        # constraints come from the rig file via fixed_joint lines.
        self.fixed_joint_indices = list(getattr(self.rig, "fixed_joint_indices", []))
        if self.fixed_joint_indices:
            ignored = [
                self.rig.joints_name[i] if 0 <= i < len(self.rig.joints_name) else f"<out-of-range:{i}>"
                for i in self.fixed_joint_indices
            ]
            print(f"✅ Fixed joint rotations: {self.fixed_joint_indices} {ignored}")

        # Initialize parametric model
        # 1. Sample 3D points
        _xyz = vertices.clone()
        _indicies = torch.arange(vertices.shape[0],device=device)
        
        # 2. Sample 3D features
        proj_feat_path = proj_path.tgt_feat_path
        if not render_mode:
            demo_feat_path = getattr(proj_path, "demo_tgt_feat_path", None)
            cached_feat_path = (
                demo_feat_path
                if demo_feat_path is not None and os.path.exists(demo_feat_path)
                else None
            )
            if not os.path.exists(proj_feat_path) and cached_feat_path is None:
                _features = self.sample_3d_features(_xyz.detach().clone(), _indicies.detach().clone(),)
                torch.save({'xyz': _xyz, 'feats_3d': _features, 'indices': _indicies}, proj_feat_path)
            else:
                load_feat_path = proj_feat_path if os.path.exists(proj_feat_path) else cached_feat_path
                if load_feat_path == cached_feat_path:
                    print(f"✅ Loaded cached 3D features: {cached_feat_path}")
                proj_feat_dict = torch.load(load_feat_path, weights_only=False)
                _xyz = proj_feat_dict['xyz'].to(device)
                _features = proj_feat_dict['feats_3d'].to(device)
                _indicies = proj_feat_dict['indices'].to(device)

        self.sampled_xyz = _xyz
        self.sampled_indices = _indicies
        
        if not render_mode:
            self.sampled_features = self.downsample_features(_features, model_cfg.downsample, proj_path, model_cfg.feature_dim)
            print(f"✅ Sampled 3D Points and Features: {self.sampled_xyz.shape[0]} points, {self.sampled_features.shape[1]}-dim features")
        else:
            self.sampled_features = None
            print(f"✅ Sampled 3D Points: {self.sampled_xyz.shape[0]} points (render mode, no features)")
              
        # 3. Sample skinning weights
        self.skinning_weights = self.rig.skinning_weights.clone() if self.sampled_indices is None else self.rig.skinning_weights[self.sampled_indices].clone()
        # Cached target features may contain a sampled subset of the original mesh. ARAP's
        # dominant-joint labels must describe that same subset, not every source vertex.
        lbs_parts, dominant_joints = self.rig.get_lbs_parts(self.sampled_indices)
        
        gaussian_params = [self.sampled_xyz, self.skinning_weights, self.sampled_features]
        if use_color:
            vertex_colors = render_utils.query_from_uv(self.mesh.visual.uv, np.array(self.mesh.visual.material.image))
            pcd_color = torch.tensor(vertex_colors, dtype=torch.float32, device=device) # (N, 3)
            if self.sampled_indices is not None:
                pcd_color = pcd_color[self.sampled_indices]
            gaussian_params.append(pcd_color)

        self.model = ParametricModel(self.rig, gaussian_params, lbs_parts, dominant_joints, model_cfg, device=device)

        # Initialize animation field
        self.animation_field = Animation(joint_num=len(self.rig.joints_pos), opt_cfg=model_cfg.opt)
        print(f"✅ Initialized Models: {self.model.__class__.__name__}, {self.animation_field.__class__.__name__}")
        
        # Load previous iteration if resuming
        if loaded_iter != 0:
            if multi_stage is None:
                gs_base = orig_proj_path.gaussians_dir
                am_base = orig_proj_path.deform_dir
                pm_base = orig_proj_path.pm_dir
            else:
                gs_base = os.path.join(orig_proj_path.gaussians_dir, f"{multi_stage}")
                am_base = os.path.join(orig_proj_path.deform_dir, f"{multi_stage}")
                pm_base = os.path.join(orig_proj_path.pm_dir, f"{multi_stage}")
            self.animation_field.load_params(os.path.join(am_base, f"iteration_{loaded_iter}.pth"))  
            if load_deform_only:
                print("✅ Loaded deform checkpoint only; using original target mesh and skeleton.")
            else:
                (model_params, first_iter) = torch.load(os.path.join(gs_base, f"iteration_{loaded_iter}.pth"), weights_only=False)
                self.model.load_params(os.path.join(pm_base, f"iteration_{loaded_iter}.pth"))
                self.model.gaussians.restore(model_params, model_cfg.opt)
            print(f"✅ Loaded Iteration: {loaded_iter}")

    def downsample_features(self, features, mode, proj_path, feature_dim):
        if mode == "autoencoder":
            self.feature_encoder = Autoencoder(input_dim=features.shape[1], latent_dim=feature_dim)
            self.feature_encoder.to(self.device)
            if os.path.exists(proj_path.encoder_path):
                self.feature_encoder.load_state_dict(torch.load(proj_path.encoder_path, weights_only=False))
            else:
                src_features = torch.cat([
                    torch.load(cam.feat_path, weights_only=False)
                    for cam in self.cameras
                    if hasattr(cam, 'feat_path') and cam.feat_path is not None
                ])
                src_features = src_features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).to(self.device)
                optimizer = optim.Adam(self.feature_encoder.parameters(), lr=0.0001)
                criterion = nn.MSELoss()
                num_epochs = 50
                batch_size = 256
                for epoch in tqdm(range(num_epochs), desc="Training Autoencoder"):
                    for i in range(0, features.shape[0], batch_size):
                        optimizer.zero_grad()
                        batch_features = features[i:i+batch_size]
                        _, decoded = self.feature_encoder(batch_features.detach())
                        loss = criterion(decoded, batch_features.detach())
                        loss.backward()
                        optimizer.step()
                torch.save(self.feature_encoder.state_dict(), proj_path.encoder_path)

            return self.feature_encoder.encoder(features.detach()).to(self.device)

        elif mode == "uniform":
            # randomly sample 64 channels from _features (768 channels)
            interval = features.shape[1] // feature_dim
            if interval <= 0:
                raise ValueError(f"Invalid uniform downsample setup: in_dim={features.shape[1]}, feature_dim={feature_dim}")
            self.feature_encoder = lambda x: x[:,::interval].clone().to(self.device)
            
        elif mode == "averaging":
            # do channel-wise averaging within intervals
            interval = features.shape[1] // feature_dim
            if interval <= 0:
                raise ValueError(f"Invalid averaging downsample setup: in_dim={features.shape[1]}, feature_dim={feature_dim}")
            
            def feature_encoder(x):
                encoded_features = torch.zeros((x.shape[0], feature_dim), device=self.device)
                for i in range(feature_dim):
                    start = i * interval
                    end = (i+1) * interval
                    encoded_features[:, i] = x[:, start:end].mean(dim=1)
                return encoded_features
            self.feature_encoder = feature_encoder
        elif mode in {"original", "none", "identity"}:
            # Keep the original feature dimension.
            self.feature_encoder = lambda x: x.to(self.device)
        else:
            raise ValueError(f"Unknown downsample mode: {mode}")

        return self.feature_encoder(features).to(self.device)


    def sample_3d_features(self, sampled_xyz, sampled_indices):
        print(self.proj_path.tgt_2d_feat_dir)
        feature_dtype = torch.float32
        with torch.no_grad():   
            # Load 2D features
            tgt_feat_paths = [
                os.path.join(self.proj_path.tgt_2d_feat_dir, f)
                for f in sorted(os.listdir(self.proj_path.tgt_2d_feat_dir))
                if f.endswith(".pt")
            ]
            if len(tgt_feat_paths) == 0:
                raise RuntimeError("No target feature paths found for sample_3d_features().")
            tgt_feat_first = torch.load(tgt_feat_paths[0], map_location="cpu", weights_only=False).to(dtype=feature_dtype)
            
            src_feat_paths = [
                cam.feat_path
                for cam in self.cameras
                if hasattr(cam, 'feat_path') and cam.feat_path is not None
            ]
            if len(src_feat_paths) == 0:
                raise RuntimeError("No source feature paths found for sample_3d_features().")
            src_feat_first = torch.load(src_feat_paths[0], map_location="cpu", weights_only=False).to(dtype=feature_dtype)
            src_dim = int(src_feat_first.shape[1])

            tgt_dim = int(tgt_feat_first.shape[1])
            src_feat_paths_resolved = src_feat_paths
            if src_dim != tgt_dim:
                raise RuntimeError(
                    f"Source/Target feature dim mismatch: src={src_dim}, tgt={tgt_dim}. "
                    "Check the precomputed sd_dino features under source and target feature/ dirs."
                )
            
        tgt_mask_dir = self.proj_path.tgt_render_dir.replace("color", "mask")
        if os.path.isdir(tgt_mask_dir) and os.listdir(tgt_mask_dir):
            tgt_mask_imgs = [
                Image.open(os.path.join(tgt_mask_dir, f)).convert("L")
                for f in sorted(os.listdir(tgt_mask_dir))
            ]
            resolution = tgt_mask_imgs[0].height, tgt_mask_imgs[0].width
            tgt_masks = torch.cat([general_utils.PILtoTorch(img, resolution) for img in tgt_mask_imgs]).cuda()
            tgt_fg_masks = tgt_masks > 0
        else:
            tgt_depth_dir = self.proj_path.tgt_render_dir.replace("color", "depth")
            tgt_depth_imgs = [Image.open(os.path.join(tgt_depth_dir, f)).convert("L") for f in sorted(os.listdir(tgt_depth_dir))]
            resolution = tgt_depth_imgs[0].height, tgt_depth_imgs[0].width
            tgt_depth = torch.cat([general_utils.PILtoTorch(img, resolution) for img in tgt_depth_imgs]).cuda()
            tgt_fg_masks = tgt_depth > 0

        src_mask_paths = [
            cam.mask_path
            for cam in self.cameras
            if hasattr(cam, 'mask_path') and cam.mask_path is not None
        ]
        if len(src_mask_paths) == 0:
            raise RuntimeError("No source mask paths found for sample_3d_features().")

        # Aggregate 3D features
        camera_params_path = os.path.join(self.proj_path.tgt_dir, "cameras")
        camera_params = [torch.load(os.path.join(camera_params_path, f), weights_only=False) for f in sorted(os.listdir(camera_params_path))]
        
        # Load visibility masks
        sampled_visibility_mask = None
        if os.path.exists(os.path.join(self.proj_path.tgt_dir, "visibility_masks.npy")):
            visibility_mask_path = os.path.join(self.proj_path.tgt_dir, "visibility_masks.npy")
            visibility_mask_batch =torch.tensor(np.load(visibility_mask_path)).cuda() # (B, N)
            sampled_visibility_mask = visibility_mask_batch[:, sampled_indices] # (B, N)
        mapping_debug_dir = None
        mapping_debug_frames = None
        if os.environ.get("MORPHGS_DEBUG_MAPPING", "").strip():
            mapping_debug_dir = os.path.join(self.proj_path.mapping_dir, "_debug")
            frame_spec = os.environ.get("MORPHGS_DEBUG_MAPPING_FRAMES", "").strip()
            mapping_debug_frames = _parse_debug_frame_indices(frame_spec) if frame_spec else None
            print(f"[Debug] Mapping debug enabled -> {mapping_debug_dir}")

        with torch.no_grad():
            from utils.feature_utils import aggregate_features_3d

            feats_3d_mean = aggregate_features_3d(
                sampled_xyz,
                tgt_feat_paths,
                src_feat_paths_resolved,
                camera_params,
                sampled_visibility_mask,
                tgt_fg_masks,
                src_mask_paths,
                mapping_dir=self.proj_path.mapping_dir,
                visualize_path=None,
                resolution=resolution,
                mapping_debug_dir=mapping_debug_dir,
                mapping_debug_frame_indices=mapping_debug_frames,
            )

        return feats_3d_mean.float()


    @torch.no_grad()
    def load_easiest_set(self):
        # Use the first 10 cameras for pretraining.
        easycam_inds = np.array([i for i in range(10)])
        return easycam_inds
