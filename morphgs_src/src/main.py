# Copyright (c) 2026 MorphGS Authors.
# Licensed under the MIT License.

import os
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
from collections import deque
import random
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from pytorch3d.loss import chamfer_distance

from model.MorphGS import MorphGS
from model.ParametricModel import ParametricModel
from model.AnimationField import Animation

from feature_splatting.arguments import PipelineParams
from feature_splatting.cameras import Camera
from feature_splatting.gaussian_model import GaussianModel
from feature_splatting.gaussian_renderer import render

from utils.setup_utils import (
    get_data_bases,
    get_demo_config_path,
    infer_experiment_from_config,
    load_config,
    merge_configs,
    parse_override_arg,
    ProjectPath,
    resolve_config_path,
    save_config,
    split_experiment_name,
)
from utils.render_utils import project_points_2d
from utils.general_utils import set_seed
from utils.camera_utils import (
    filter_active_view_names,
    get_aligned_mapping_paths,
    get_all_view_names,
)

from render import render_anim_w_mesh
import time
from skimage import filters
from skimage.morphology import binary_closing, disk, thin


def l1_loss(network_output, gt, mask=None):
    assert network_output.shape == gt.shape
    if mask is not None:
        if gt.shape[-2:] != mask.shape:
            mask = F.interpolate(mask[None, None].half(), size=gt.shape[-2:], mode="nearest")[0, 0].bool()
        gt = gt[..., mask > 0]
        network_output = network_output[..., mask > 0]
    return torch.abs(network_output - gt).mean()


def _count_mapping_files(mapping_dir):
    if mapping_dir is None:
        return 0
    if not os.path.isdir(mapping_dir):
        return 0
    return sum(1 for name in os.listdir(mapping_dir) if name.endswith(".pt"))


def _expected_mapping_count(src_dir):
    expected = 0
    for view_name in filter_active_view_names(get_all_view_names(src_dir)):
        color_dir = os.path.join(src_dir, view_name, "color")
        if not os.path.isdir(color_dir):
            continue
        expected += sum(
            1
            for name in os.listdir(color_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        )
    return expected


def _mapping_dir_with_fallback(proj_path):
    mapping_dir = proj_path.mapping_dir
    expected_count = _expected_mapping_count(proj_path.src_dir)
    output_count = _count_mapping_files(mapping_dir)
    if output_count > 0 and (expected_count == 0 or output_count >= expected_count):
        return mapping_dir
    if output_count > 0:
        print(
            f"[Info] Ignoring incomplete output mapping cache: "
            f"{mapping_dir} ({output_count}/{expected_count})"
        )

    demo_mapping_dir = getattr(proj_path, "demo_mapping_dir", None)
    demo_count = _count_mapping_files(demo_mapping_dir)
    if demo_count > 0 and (expected_count == 0 or demo_count >= expected_count):
        print(f"✅ Loaded cached 2D-3D mappings: {demo_mapping_dir}")
        return demo_mapping_dir

    return mapping_dir


def train_pose(morphgs: MorphGS, proj_path, background, opt_cfg, args, device, cfg, VN=5, gt_views=[0], on_train_start=None, on_iteration_end=None, on_train_end=None):
    gaussians:GaussianModel = morphgs.model.gaussians
    model:ParametricModel = morphgs.model
    train_skinning_apply_softmax = bool(getattr(cfg.model, "skinning_apply_softmax", False))
    model.skinning_apply_softmax = train_skinning_apply_softmax
    anim_field:Animation = morphgs.animation_field
    cameras:Camera = morphgs.cameras
    cam_to_global_idx = {id(cam): i for i, cam in enumerate(cameras)}

    start_iter = args.iteration if args.iteration else 0
    
    gaussians.training_setup(opt_cfg)

    progress_bar = tqdm(range(start_iter+1, opt_cfg.iterations+1), desc="Training progress")
    ema_loss_for_log = 0.0

    mapping_dir = _mapping_dir_with_fallback(proj_path)
    mapping_paths = get_aligned_mapping_paths(mapping_dir, proj_path.src_dir)
    if len(mapping_paths) == 0:
        raise RuntimeError(
            "No dense 2D-3D mapping cache found. Expected mapping .pt files under "
            f"{proj_path.mapping_dir} or {getattr(proj_path, 'demo_mapping_dir', None)}."
        )
    keyp_mapping_cache = {}
    def _get_keyp_mapping(cam_global_idx):
        cached = keyp_mapping_cache.get(cam_global_idx)
        if cached is None:
            mapping_dict = torch.load(mapping_paths[cam_global_idx], map_location="cpu", weights_only=False)
            px2vtx = mapping_dict["pseudo_gt"]
            confs = mapping_dict["conf"]
            fg_pixels = torch.nonzero(px2vtx >= 0)
            cached = (
                fg_pixels,
                px2vtx[fg_pixels[:, 0], fg_pixels[:, 1]],
                confs[fg_pixels[:, 0], fg_pixels[:, 1]],
                int(px2vtx.shape[0]),
                int(px2vtx.shape[1]),
            )
            keyp_mapping_cache[cam_global_idx] = cached
        fg_pixels, vtx_ids, confs, map_h, map_w = cached
        return (
            fg_pixels.to(device=device, non_blocking=True),
            vtx_ids.to(device=device, non_blocking=True),
            confs.to(device=device, non_blocking=True),
            map_h,
            map_w,
        )
    cid_num = gaussians.get_xyz.shape[0]

    # Group cameras by view index (colmap_id) so reference view selection is order-agnostic.
    cams_by_view = {}
    for cam in cameras:
        cams_by_view.setdefault(int(cam.colmap_id), []).append(cam)
    for k in cams_by_view:
        cams_by_view[k] = sorted(cams_by_view[k], key=lambda c: int(c.uid))
    frame_idx_by_cam = {}
    for k in cams_by_view:
        for fi, cam in enumerate(cams_by_view[k]):
            frame_idx_by_cam[id(cam)] = fi

    if len(cams_by_view) == 0:
        raise ValueError("No cameras loaded.")

    # Number of frames from the first selected reference view.
    if len(gt_views) == 0:
        raise ValueError("gt_views must contain at least one camera index.")
    if int(gt_views[0]) not in cams_by_view:
        raise ValueError(f"Reference camera index {gt_views[0]} not found. Available: {sorted(cams_by_view.keys())}")
    NF = len(cams_by_view[int(gt_views[0])])
    
    use_multiview = opt_cfg.use_multiview
    multiview_loss_weight = opt_cfg.multiview_loss_weight
    keyp_weight_decay_start_iter = int(opt_cfg.keyp_weight_decay_start_iter)
    keyp_weight_decay_end_iter = int(opt_cfg.keyp_weight_decay_end_iter)
    keyp_weight_decay_final_ratio = float(opt_cfg.keyp_weight_decay_final_ratio)
    keyp_multiview_end_iter = int(opt_cfg.keyp_multiview_end_iter)
    keyp_weight_decay_final_ratio = max(0.0, min(1.0, keyp_weight_decay_final_ratio))
    if keyp_weight_decay_end_iter < keyp_weight_decay_start_iter:
        keyp_weight_decay_end_iter = keyp_weight_decay_start_iter
    rasterizer_backend = opt_cfg.rasterizer_backend
    enable_opacity_reset = opt_cfg.enable_opacity_reset
    adaptive_chamfer_min_weight = 0.05
    adaptive_chamfer_sigma_div = 2.0

    densify_from_iter = int(opt_cfg.densify_from_iter)
    densify_until_iter = int(opt_cfg.densify_until_iter)
    if densify_from_iter > densify_until_iter:
        print(f"[Warning] densify_from_iter({densify_from_iter}) > densify_until_iter({densify_until_iter}). Swapping values.")
        densify_from_iter, densify_until_iter = densify_until_iter, densify_from_iter

    print(
        f"[Train Config] rasterizer_backend={rasterizer_backend}, "
        f"densify=({densify_from_iter}->{densify_until_iter})"
    )
    motion_lr_val = float(opt_cfg.motion_lr)
    global_t_lr_val = float(opt_cfg.global_t_lr)
    joint_rot_lr_val = float(opt_cfg.joint_rot_lr)
    if motion_lr_val > 0.02:
        print(
            f"[Warning] motion_lr={motion_lr_val:.4g} is high and can destabilize motion "
            "(off-screen drift / empty renders). Recommended <= 0.005."
        )
    if global_t_lr_val > 0.02:
        print(
            f"[Warning] global_t_lr={global_t_lr_val:.4g} is high and can destabilize translation. "
            "Recommended <= 0.01."
        )
    if joint_rot_lr_val > 0.02:
        print(
            f"[Warning] joint_rot_lr={joint_rot_lr_val:.4g} is high and can destabilize rotation. "
            "Recommended <= 0.01."
        )
    all_view_ids = sorted(cams_by_view.keys())
    VN = max(int(VN), 1)
    fixed_multiview_ids = {}
    if use_multiview:
        pair_rng = random.Random(0)
        for ref_vid in all_view_ids:
            other_view_ids = [vid for vid in all_view_ids if vid != ref_vid]
            num_other_to_sample = max(0, min(VN - 1, len(other_view_ids)))
            sampled_other_ids = pair_rng.sample(other_view_ids, k=num_other_to_sample) if num_other_to_sample > 0 else []
            fixed_multiview_ids[int(ref_vid)] = [int(ref_vid)] + sampled_other_ids
    _viewpoint_stack = []
    for gt_cam_i in gt_views:
        gt_cam_i = int(gt_cam_i)
        if gt_cam_i not in cams_by_view:
            raise ValueError(f"gt_view {gt_cam_i} not found. Available: {sorted(cams_by_view.keys())}")
        _viewpoint_stack += cams_by_view[gt_cam_i]
    viewpoint_stack = _viewpoint_stack.copy()

    if len(viewpoint_stack) == 0:
        raise ValueError("No reference-view cameras selected. Check single_view_cam_index / gt_views.")

    def _effective_keyp_weight(cur_iter: int):
        base = float(opt_cfg.weight_keyp_loss)
        if base <= 0:
            return 0.0
        if cur_iter <= keyp_weight_decay_start_iter:
            return base
        if cur_iter >= keyp_weight_decay_end_iter:
            return base * keyp_weight_decay_final_ratio
        span = max(1, keyp_weight_decay_end_iter - keyp_weight_decay_start_iter)
        t = float(cur_iter - keyp_weight_decay_start_iter) / float(span)
        ratio = (1.0 - t) + t * keyp_weight_decay_final_ratio
        return base * ratio
    
    training_start_time = time.time()
    total_loss = []
    
    # Convergence detection: window must be larger than the densification interval to be safe.
    convergence_window_size = 500
    convergence_threshold = 1e-4  # converged when relative loss change falls below 0.01%
    loss_window = deque(maxlen=convergence_window_size)
    convergence_time = None
    
    thinning_cache_path = os.path.join(proj_path.project_dir, "gt_thinning_cache.pt")
    thinning_cache = {}
    if os.path.exists(thinning_cache_path):
        thinning_cache = torch.load(thinning_cache_path, weights_only=False)

    thinning_footprint = disk(2)
    cache_updated = False
    for cam in cameras:
        cache_key = (int(cam.uid), int(cam.colmap_id))
        binary_image = cam.foreground_mask.squeeze(0).cpu().numpy().astype(bool)  # (H, W)
        coarse_binary = binary_closing(binary_image, thinning_footprint)
        thinned_coarse = thin(coarse_binary)
        smoothed = filters.gaussian(binary_image, sigma=3)
        thinned_smoothed_coarse = (
            torch.from_numpy(thinned_coarse.astype(float)) *
            torch.from_numpy(smoothed)
        )
        gt_proj_nodes_i = torch.nonzero(
            thinned_smoothed_coarse > 0
        )[:, [1, 0]].float().unsqueeze(0).cpu()  # (y, x) -> (x, y)
        thinning_cache[cache_key] = gt_proj_nodes_i
        cache_updated = True

    if cache_updated:
        torch.save(thinning_cache, thinning_cache_path)
    thinning_cache_device = {k: v.to(device) for k, v in thinning_cache.items()}

    if on_train_start is not None:
        on_train_start()

    # For adaptive multiview chamfer weighting (RigGS-style reliability weighting).
    mv_chamfer_history = torch.ones(len(cameras), device=device) * 1.0e5
    
    for iteration in range(start_iter + 1, opt_cfg.iterations+1):

        gaussians.update_learning_rate(iteration)
        model.update_learning_rate(iteration)
        oneup_step = int(opt_cfg.oneup_sh_degree_step)
        if oneup_step > 0 and iteration % oneup_step == 0:
            gaussians.oneupSHdegree()

        if len(viewpoint_stack) == 0:
            viewpoint_stack = _viewpoint_stack.copy()
            
        cam_idx = random.randint(0, len(viewpoint_stack)-1)
        view = viewpoint_stack[cam_idx]

        if use_multiview:
            frame_idx = frame_idx_by_cam[id(view)]
            ref_vid = int(view.colmap_id)
            selected_view_ids = [
                vid for vid in fixed_multiview_ids.get(ref_vid, [ref_vid])
                if frame_idx < len(cams_by_view[vid])
            ]
            all_views = [cams_by_view[vid][frame_idx] for vid in selected_view_ids]
        else:
            all_views = [view]
        
        iter_background = background

        gt_images = []
        gt_masks = []
        for vidx, _v in enumerate(all_views):
            _gt_img = _v.original_image
            gt_images.append(_gt_img)
            gt_masks.append(_v.foreground_mask)
        gt_images = torch.stack(gt_images) # (V, H, W, 3)
        gt_masks = torch.stack(gt_masks) # (V, 1, H, W)

        ##################
        ##### Render #####
        ##################
        
        # Animate with Skeleton: Batch=1, one timestep per iteration
        cam_t = frame_idx_by_cam[id(view)] / NF
        canonical_xyz = model.get_xyz
        canonical_joints = model.get_joints_pos
        canonical_weights = model.get_skinning_weights
        deform_params = anim_field.step(
            canonical_xyz, canonical_joints, canonical_weights, model.rig,
            torch.tensor([cam_t], dtype=torch.float, device=device),
            global_transl_only=(iteration <= opt_cfg.joint_rot_start_iterations), # Initially, only optimize global translation to avoid bad local minima. Then optimize rotation after some iterations.
            iteration=iteration,
            fixed_joints=morphgs.fixed_joint_indices,
        ) 
        
        precomp_xyz = deform_params["xyz"]
        joints_warped = deform_params["joints_warped"]
        d_rotation = deform_params['d_rotation']

        precomp_xyz *= model.scale

        render_rgb_batch = []
        render_alpha_batch = []
        render_pkg_main = None
        for v in all_views:
            render_pkg = render(
                v, gaussians, pipe, iter_background, render_features=True,
                precomp_xyz=precomp_xyz, override_color=None, d_rotation=d_rotation,
                use_feature_sh=False,
                rasterizer_backend=rasterizer_backend,
            )
            if render_pkg_main is None:
                render_pkg_main = render_pkg
            render_rgb_batch.append(render_pkg["render"].permute(1,2,0))
            render_alpha_batch.append(render_pkg["render_alpha"])
        render_rgb_batch = torch.stack(render_rgb_batch) # B, H, W, 3
        render_alpha_batch = torch.stack(render_alpha_batch) # B, 1, H, W

        ##################
        ##### Losses #####
        ##################
        loss = 0
        multiview_consistency_loss = 0
        mask_loss_main = None
        mask_loss_multiview = None

        use_mv_losses = True
        effective_mv_weight = multiview_loss_weight
        current_keyp_weight = _effective_keyp_weight(iteration)
        keyp_mv_active = (keyp_multiview_end_iter < 0) or (iteration <= keyp_multiview_end_iter)

        # Render loss
        for vidx, v in enumerate(all_views):
            Ll1 = l1_loss(render_rgb_batch[vidx], gt_images[vidx].permute(1,2,0))
            loss_ssim =  ssim(render_rgb_batch[vidx].unsqueeze(0).permute(0,3,1,2), gt_images[vidx].unsqueeze(0))
            Lrender = (1.0 - opt_cfg.lambda_dssim) * Ll1 + opt_cfg.lambda_dssim * (1.0 - loss_ssim)
            if vidx == 0:
                loss += opt_cfg.weight_render_loss * Lrender
            else:
                if use_mv_losses:
                    render_mv_pre_i = opt_cfg.weight_render_loss * Lrender
                    multiview_consistency_loss += render_mv_pre_i

        # Mask Loss
        mask_loss_start_iter = opt_cfg.mask_loss_start_iter
        if iteration >= mask_loss_start_iter and opt_cfg.weight_mask_loss > 0:
            for vidx, _ in enumerate(all_views):
                render_alpha = render_alpha_batch[vidx]
                mask_loss_i = F.l1_loss(render_alpha, gt_masks[vidx])
                if vidx == 0:
                    if mask_loss_main is None:
                        mask_loss_main = mask_loss_i
                    else:
                        mask_loss_main = mask_loss_main + mask_loss_i
                else:
                    if use_mv_losses:
                        if mask_loss_multiview is None:
                            mask_loss_multiview = mask_loss_i
                        else:
                            mask_loss_multiview = mask_loss_multiview + mask_loss_i
        
        # Keypoint Loss
        Lkeyp = None
        
        if current_keyp_weight > 0:
            B = len(all_views)
            
            # Gaussian index to mesh vertex index mapping
            gs_ivtx = gaussians.ivtx.squeeze(1)
            c_bin = torch.zeros(cid_num, dtype=torch.long, device=device) - 1
            c_bin[gs_ivtx] = torch.arange(gs_ivtx.shape[0], device=device, dtype=torch.long)
            c_valid_mask = (c_bin != -1)

            batch_sampled_pixels = []
            batch_xyz = []
            keyp_view_indices = []

            # extract high-confidence pixel-to-vertex correspondences for each view in the batch
            for b in range(B):
                cam_global_idx = cam_to_global_idx[id(all_views[b])]
                fg_pixels, vtx_ids, vtx_confs, map_h, map_w = _get_keyp_mapping(cam_global_idx)
                
                # confidence filtering + valid gs filtering
                mask = (vtx_confs > opt_cfg.keyp_threshold) & c_valid_mask[vtx_ids]
                valid_pixels = fg_pixels[mask]
                valid_vtxs = vtx_ids[mask]

                num_to_sample = min(int(opt_cfg.dense_keyp_num), len(valid_pixels))
                if num_to_sample > 0:
                    sample_idx = torch.randperm(len(valid_pixels), device=device)[:num_to_sample]
                    sample_idx, _ = torch.sort(sample_idx)
                    
                    # (y, x) -> (x, y)
                    s_pixels = valid_pixels[sample_idx][:, [1, 0]].float()
                    view_h = int(all_views[b].image_height)
                    view_w = int(all_views[b].image_width)
                    if map_w > 0 and map_h > 0 and (map_w != view_w or map_h != view_h):
                        s_pixels[:, 0] *= float(view_w) / float(map_w)
                        s_pixels[:, 1] *= float(view_h) / float(map_h)
                    s_xyz = precomp_xyz[c_bin[valid_vtxs[sample_idx]]]
                    
                    batch_sampled_pixels.append(s_pixels)
                    batch_xyz.append(s_xyz)
                    keyp_view_indices.append(b)

            if len(batch_sampled_pixels) > 0:
                min_keyp_count = min(p.shape[0] for p in batch_sampled_pixels)
                batch_sampled_pixels = [p[:min_keyp_count] for p in batch_sampled_pixels]
                batch_xyz = [xyz[:min_keyp_count] for xyz in batch_xyz]
                # 2. [B_valid, N_sample, 2/3]
                target_pixels = torch.stack(batch_sampled_pixels)
                input_xyz = torch.stack(batch_xyz)

                # 3. Batch Projection
                keyp_views = [all_views[view_idx] for view_idx in keyp_view_indices]
                xyz_2d, _ = project_points_2d(input_xyz, keyp_views)
                
                # 4. Element-wise L1 loss
                Lkeyp = F.smooth_l1_loss(
                    xyz_2d[0],                   # (n,2)
                    target_pixels[0].float(),   # (n,2)
                    beta=1.0,
                    reduction='none'
                ).mean(dim=1).mean()
                loss += current_keyp_weight * Lkeyp
                
                # 5. multiview consistency for keypoints
                if use_mv_losses and keyp_mv_active and len(all_views) > 1:
                    m_Lkeyp = F.smooth_l1_loss(
                        xyz_2d[1:],                   # (n,2)
                        target_pixels[1:].float(),   # (n,2)
                        beta=1.0,
                        reduction='none'
                    ).mean(dim=1)
                    multiview_consistency_loss += current_keyp_weight * m_Lkeyp.sum()
              
        # Motion Regularizers
        Ltreg = None
        Ltreg_raw = None
        Lsmooth = None
        Lsmooth_raw = None
        if opt_cfg.weight_transformation_reg > 0: # Large Motion Suppression: From APN
            Ltreg_raw = anim_field.get_transformation_regularisation_loss()
            Ltreg = opt_cfg.weight_transformation_reg * Ltreg_raw
            loss += Ltreg

        if opt_cfg.weight_smooth_reg > 0: # Motion smoothness loss
            Lsmooth_raw = anim_field.get_smoothness_loss(NF)
            Lsmooth = opt_cfg.weight_smooth_reg * Lsmooth_raw
            loss += Lsmooth

        # Twist (screw-rotation about the bone axis) suppression; off unless configured.
        Ltwist = None
        twist_reg_weight = float(opt_cfg.weight_twist_reg)
        if twist_reg_weight > 0:
            Ltwist_raw = anim_field.get_twist_regularisation_loss(model.rig)
            Ltwist = twist_reg_weight * Ltwist_raw
            loss += Ltwist

        Larap = None
        if iteration > opt_cfg.joint_rot_start_iterations:
            # 2D Chamfer Loss (2d skeleton loss)
            if opt_cfg.chamfer_2d_reg > 0: # 2d skeleton regularization
                sampling_points = model.sampling_skeleton_points(joints_warped * model.scale)
                for vi, v in enumerate(all_views):
                    proj_nodes_i, _ = project_points_2d(sampling_points, v)

                    cache_key = (int(v.uid), int(v.colmap_id))
                    gt_proj_nodes_i = thinning_cache_device[cache_key]

                    Lchamf_i, _ = chamfer_distance(x=proj_nodes_i, y=gt_proj_nodes_i, norm=1)
                    if vi == 0:
                        loss += opt_cfg.chamfer_2d_reg * Lchamf_i
                    else:
                        chamfer_w = 1.0
                        cam_global_idx = cam_to_global_idx[id(v)]
                        mv_chamfer_history[cam_global_idx] = Lchamf_i.detach()
                        valid_hist = mv_chamfer_history[mv_chamfer_history < 1.0e4]
                        if valid_hist.numel() > 1:
                            sigma = torch.clamp(torch.median(valid_hist) / adaptive_chamfer_sigma_div, min=1e-6)
                            chamfer_w = torch.exp(-(mv_chamfer_history[cam_global_idx] ** 2) / (2.0 * sigma ** 2))
                            chamfer_w = torch.clamp(chamfer_w, min=adaptive_chamfer_min_weight, max=1.0)
                                
                        chamfer_mv_pre_i = opt_cfg.chamfer_2d_reg * chamfer_w * Lchamf_i
                        multiview_consistency_loss += chamfer_mv_pre_i
            
            # ARAP Loss
            if opt_cfg.weight_arap > 0:   # ARAP Loss: From APN
                Larap = model.get_arap_loss(precomp_xyz)
                loss += opt_cfg.weight_arap * Larap

        loss += effective_mv_weight * multiview_consistency_loss

        weighted_mask_loss = None
        if mask_loss_main is not None:
            weighted_mask_loss = opt_cfg.weight_mask_loss * (
                mask_loss_main + effective_mv_weight * (mask_loss_multiview if mask_loss_multiview is not None else 0.0)
            )
            loss += weighted_mask_loss

        loss.backward()
        
        full_loss_for_log = loss.detach()

        if on_iteration_end is not None:
            on_iteration_end(iteration)

        total_loss.append(full_loss_for_log.item())
        current_loss_val = full_loss_for_log.item()
        loss_window.append(current_loss_val)
        
        if convergence_time is None and iteration > densify_until_iter and len(loss_window) == convergence_window_size:
            half_idx = convergence_window_size // 2
            prev_avg = sum(list(loss_window)[:half_idx]) / half_idx
            curr_avg = sum(list(loss_window)[half_idx:]) / half_idx
            
            relative_change = abs(prev_avg - curr_avg) / (prev_avg + 1e-8)
            
            if relative_change < convergence_threshold:
                convergence_time = time.time() - training_start_time
                print(f"\n[Convergence Detected] Iter: {iteration}, Time: {convergence_time/60:.2f} min, Loss Change: {relative_change:.6f}")
                
                if on_train_end is not None:
                    on_train_end(iteration)
                return

        with torch.no_grad():
            ema_loss_for_log = 0.4 * full_loss_for_log.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt_cfg.iterations:
                progress_bar.close()

            if iteration < opt_cfg.iterations:
                
                # UPDATE_PER_FRAME: Local motion, Local shape(GAUSSIANS), skinning weight
                if iteration > opt_cfg.shape_start_iterations:
                    if iteration < densify_until_iter:
                        visibility_filter = render_pkg_main["visibility_filter"]
                        radii = render_pkg_main["radii"]
                        viewspace_point_tensor = render_pkg_main["viewspace_points"]
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                        if iteration > densify_from_iter and iteration % opt_cfg.densification_interval == 0:
                            size_threshold = 20 if iteration > opt_cfg.opacity_reset_interval else None
                            gaussians.densify_and_prune(opt_cfg.densify_grad_threshold, 0.005, morphgs.cameras_extent, size_threshold)
                        if enable_opacity_reset and (
                            iteration % opt_cfg.opacity_reset_interval == 0
                        ):
                            gaussians.reset_opacity()
                    
                    assert gaussians.ivtx.shape[0] == gaussians.get_xyz.shape[0]
            
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                anim_field.optimizer_by_frame.step()
                anim_field.optimizer_by_frame.zero_grad(set_to_none = True)

                # UPDATE_PER_VIDEO: Global translation, bone length
                if iteration % opt_cfg.vid_window_size == 0:    
                    model.optimizer.step()
                    model.optimizer.zero_grad(set_to_none = True)
                    
                    anim_field.optimizer_by_video.step()
                    anim_field.optimizer_by_video.zero_grad(set_to_none = True)

                anim_field.step_lr()

            if iteration == opt_cfg.iterations or iteration % 5000 == 0:
                torch.save((gaussians.capture(), iteration), os.path.join(proj_path.gaussians_dir, f"iteration_{iteration}.pth"))
                model.save_params(os.path.join(proj_path.pm_dir, f"iteration_{iteration}.pth"))
                anim_field.save_params(os.path.join(proj_path.deform_dir, f"iteration_{iteration}.pth"))

                render_anim_w_mesh(morphgs, proj_path, iteration, device, cfg)
                model.skinning_apply_softmax = train_skinning_apply_softmax

    if on_train_end is not None:
        on_train_end(opt_cfg.iterations)
    
    return


if __name__ == '__main__':
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Load configuration
    parser = ArgumentParser(description="Testing script parameters")
    pipe = PipelineParams(parser) # Add pipeline arguments
    parser.add_argument('--config', required=True, help="Path to a config file. Demo aliases like demo/<experiment>.yaml are supported.")
    parser.add_argument('--out', default='output', help="Path to the output directory")
    parser.add_argument('--iteration', default=0, type=int, help="Iteration to load the model from")
    parser.add_argument('--experiment', default=None, help="Name of the experiment")

    # remaining unkown args: --key.subkey=value for overriding config values
    args, unknown = parser.parse_known_args()
    args.config = resolve_config_path(args.config)
    if args.experiment is None:
        inferred_experiment = infer_experiment_from_config(args.config)
        if inferred_experiment is not None:
            args.experiment = inferred_experiment
            print(f"[Experiment] Inferred experiment from config: {args.experiment}")
    base_config = load_config("configs/base.yaml")
    _config = load_config(args.config)
    config = merge_configs(base_config, _config)
    pipe = pipe.extract(args)

    # SET UP EXPERIMENT DATA
    ANIM_BASE, CHAR_BASE = get_data_bases(
        video_layout="processed",
    )
    print(ANIM_BASE, CHAR_BASE)

    if args.experiment:
        dirname = args.experiment
        config.project.name = dirname
        demo_config_path = get_demo_config_path(dirname)
        if os.path.exists(demo_config_path) and os.path.abspath(demo_config_path) != os.path.abspath(args.config):
            config = merge_configs(config, load_config(demo_config_path))
            print(f"[Config] Loaded experiment config: {demo_config_path}")

        source, target = split_experiment_name(dirname)
        config.project.source = os.path.join(ANIM_BASE, source)
        config.project.target = os.path.join(CHAR_BASE, target)

    # override config with unknown args
    for ov in unknown:
        if not ov.startswith('--') or '=' not in ov:
            continue
        parse_override_arg(ov[2:], config)

    # SET UP PROJECT PATH
    proj_cfg = config.project
    model_cfg = config.model

    #### PROJECT SETUP ####
    proj_path = ProjectPath(
        args.out,
        proj_cfg.name,
        model_cfg.name,
        proj_cfg.source,
        proj_cfg.target,
    ) # Assume that we already have rigging info
    print("Saving config to", proj_path.model_dir)
    save_config(config, os.path.join(proj_path.model_dir, "config.yaml"))

    torch.cuda.empty_cache()

    set_seed(proj_cfg.seed)

    morphgs = MorphGS(
        proj_path,
        device=device,
        model_cfg=model_cfg,
        loaded_iter=args.iteration,
    )
    background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")

    #### ANIMATION FIELD LEARNING ####
    single_view_cam_index = int(model_cfg.opt.single_view_cam_index)

    train_pose(
        morphgs, proj_path, background, model_cfg.opt, args, device,
        config,
        VN=model_cfg.opt.num_views,
        gt_views=[single_view_cam_index],
    )
    os._exit(0)
