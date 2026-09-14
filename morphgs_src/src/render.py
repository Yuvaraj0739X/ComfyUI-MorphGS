# Copyright (c) 2026 MorphGS Authors.
# Licensed under the MIT License.

import torch
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm
import imageio
import trimesh

from utils.setup_utils import (
    get_data_bases,
    get_demo_config_path,
    infer_experiment_from_config,
    load_config,
    merge_configs,
    parse_override_arg,
    ProjectPath,
    resolve_config_path,
    split_experiment_name,
)

from model.MorphGS import MorphGS

from utils.general_utils import set_seed

from utils.render_utils import Renderer
import time
from model.AnimationField import Animation

from pytorch3d.renderer import PointLights, Materials
import cv2

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.8)

from utils.render_utils import draw_skeleton, project_points_2d


def _cfg_get(obj, key, default=None):
    if obj is None or not hasattr(obj, key):
        return default
    return getattr(obj, key)


def _cfg_bool(obj, key, default=False):
    value = _cfg_get(obj, key, default)
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return bool(value)


def _render_arg_or_config(args, config, key, default):
    arg_value = getattr(args, key)
    if arg_value is not None:
        return arg_value
    return _cfg_get(_cfg_get(config, "render"), key, default)


def _find_latest_iteration(checkpoint_dir):
    latest = None
    pattern = os.path.join(checkpoint_dir, "iteration_*.pth")
    for path in glob.glob(pattern):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            iteration = int(stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if latest is None or iteration > latest:
            latest = iteration
    return latest


def _resize_fit_np(rgb, size, fill=(255, 255, 255)):
    target_w, target_h = size
    h, w = rgb.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    resized = cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    rh, rw = resized.shape[:2]
    canvas = np.empty((target_h, target_w, 3), dtype=resized.dtype)
    canvas[...] = np.array(fill, dtype=resized.dtype)
    x0 = max((target_w - rw) // 2, 0)
    y0 = max((target_h - rh) // 2, 0)
    canvas[y0:y0 + rh, x0:x0 + rw] = resized
    return canvas


def _resize_center_crop_np(rgb, size):
    target_w, target_h = size
    h, w = rgb.shape[:2]
    scale = max(target_w / max(w, 1), target_h / max(h, 1))
    resized = cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    rh, rw = resized.shape[:2]
    x0 = max((rw - target_w) // 2, 0)
    y0 = max((rh - target_h) // 2, 0)
    return resized[y0:y0 + target_h, x0:x0 + target_w]


def _resize_right_crop_np(rgb, size):
    target_w, target_h = size
    h, w = rgb.shape[:2]
    scale = max(target_w / max(w, 1), target_h / max(h, 1))
    resized = cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    rh, rw = resized.shape[:2]
    x0 = max(rw - target_w, 0)
    y0 = max((rh - target_h) // 2, 0)
    return resized[y0:y0 + target_h, x0:x0 + target_w]


def _resize_reduced_pad_np(rgb, size, fill=(255, 255, 255)):
    target_w, target_h = size
    h, w = rgb.shape[:2]
    fit_scale = min(target_w / max(w, 1), target_h / max(h, 1))
    fit_h = int(round(h * fit_scale))
    fit_pad_y = max((target_h - fit_h) // 2, 0)
    pad_y = fit_pad_y // 3
    content_h = max(target_h - 2 * pad_y, 1)
    scale = content_h / max(h, 1)
    resized = cv2.resize(rgb, (int(round(w * scale)), content_h), interpolation=cv2.INTER_AREA)
    rh, rw = resized.shape[:2]
    canvas = np.empty((target_h, target_w, 3), dtype=resized.dtype)
    canvas[...] = np.array(fill, dtype=resized.dtype)
    x0 = max((rw - target_w) // 2, 0)
    crop = resized[:, x0:x0 + target_w]
    y0 = max((target_h - rh) // 2, 0)
    canvas[y0:y0 + rh, :crop.shape[1]] = crop
    return canvas


def _resize_reduced_pad_left_crop_np(
    rgb,
    size,
    fill=(255, 255, 255),
    vertical_pad_ratio=0.12,
    left_crop_bias=1.0,
):
    target_w, target_h = size
    h, w = rgb.shape[:2]
    pad_y = int(round(target_h * float(vertical_pad_ratio)))
    content_h = max(target_h - 2 * pad_y, 1)
    scale = content_h / max(h, 1)
    resized = cv2.resize(rgb, (int(round(w * scale)), content_h), interpolation=cv2.INTER_AREA)
    rh, rw = resized.shape[:2]
    canvas = np.empty((target_h, target_w, 3), dtype=resized.dtype)
    canvas[...] = np.array(fill, dtype=resized.dtype)
    max_x0 = max(rw - target_w, 0)
    x0 = max(int(round(max_x0 * float(left_crop_bias))), 0)
    x0 = min(x0, max_x0)
    crop = resized[:, x0:x0 + target_w]
    x_dst = 0
    y0 = max((target_h - rh) // 2, 0)
    canvas[y0:y0 + crop.shape[0], x_dst:x_dst + crop.shape[1]] = crop
    return canvas


def _resize_preview_np(rgb, size, mode, vertical_pad_ratio=0.12, left_crop_bias=1.0):
    if mode == "center_crop":
        return _resize_center_crop_np(rgb, size)
    if mode == "right_crop":
        return _resize_right_crop_np(rgb, size)
    if mode == "reduced_pad":
        return _resize_reduced_pad_np(rgb, size)
    if mode == "reduced_pad_left_crop":
        return _resize_reduced_pad_left_crop_np(
            rgb,
            size,
            vertical_pad_ratio=vertical_pad_ratio,
            left_crop_bias=left_crop_bias,
        )
    if mode != "fit":
        print(f"[Render] Unknown source_preview_mode '{mode}', using fit.", flush=True)
    return _resize_fit_np(rgb, size)


def _foreground_center_np(rgb, background=(255, 255, 255), threshold=5):
    bg = np.array(background, dtype=np.float32).reshape(1, 1, 3)
    diff = np.abs(rgb.astype(np.float32) - bg).max(axis=2)
    ys, xs = np.nonzero(diff > threshold)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _recenter_scale_panel(panel, scale=1.0, center=(0.5, 0.5), source_center=None):
    if scale == 1.0 and center == (0.5, 0.5):
        return panel

    panel_np = panel.detach().cpu().numpy().astype(np.float32)
    h, w = panel_np.shape[:2]
    if source_center is None:
        source_center = _foreground_center_np(panel_np)
    if source_center is None:
        return panel

    dst_center = (float(center[0]) * w, float(center[1]) * h)
    matrix = np.array(
        [
            [scale, 0.0, dst_center[0] - scale * source_center[0]],
            [0.0, scale, dst_center[1] - scale * source_center[1]],
        ],
        dtype=np.float32,
    )
    rgb_max = float(panel_np[..., :3].max()) if panel_np.size else 255.0
    white = 1.0 if rgb_max <= 1.5 else 255.0
    if panel_np.shape[2] == 4:
        border_value = (white, white, white, 0.0)
    else:
        border_value = (white, white, white)
    warped = cv2.warpAffine(
        panel_np,
        matrix,
        (w, h),
        flags=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return torch.from_numpy(warped).to(device=panel.device, dtype=panel.dtype)


def _load_image_sequence_frames(
    frame_dir,
    num_frames,
    size,
    preview_mode,
    vertical_pad_ratio=0.12,
    left_crop_bias=1.0,
):
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(glob.glob(os.path.join(frame_dir, ext)))
    paths = sorted(paths)
    if len(paths) == 0:
        return None

    frames = []
    for path in paths[:num_frames]:
        rgb = np.array(Image.open(path).convert("RGB"))
        frames.append(
            torch.from_numpy(
                _resize_preview_np(
                    rgb,
                    size,
                    preview_mode,
                    vertical_pad_ratio=vertical_pad_ratio,
                    left_crop_bias=left_crop_bias,
                )
            ).float() / 255.0
        )
    return frames


def _load_video_frames(
    video_path,
    num_frames,
    size,
    preview_mode,
    vertical_pad_ratio=0.12,
    left_crop_bias=1.0,
):
    frames = []
    try:
        reader = imageio.get_reader(video_path)
        for idx, frame in enumerate(reader):
            if idx >= num_frames:
                break
            rgb = np.asarray(frame[..., :3])
            frames.append(
                torch.from_numpy(
                    _resize_preview_np(
                        rgb,
                        size,
                        preview_mode,
                        vertical_pad_ratio=vertical_pad_ratio,
                        left_crop_bias=left_crop_bias,
                    )
                ).float() / 255.0
            )
        reader.close()
    except Exception as exc:
        print(f"[Render] Could not read source video {video_path}: {exc}", flush=True)
        return None
    return frames if frames else None


def _source_preview_candidates(src_dir):
    scene = os.path.basename(src_dir.rstrip(os.sep))
    processed_root = os.path.dirname(src_dir.rstrip(os.sep))
    data_root = os.path.dirname(processed_root) if os.path.basename(processed_root) == "processed_videos" else None
    candidates = []

    if data_root is not None:
        raw_scene_dir = os.path.join(data_root, "videos", scene)
        for dirname in ("color_bg", "color_raw", "raw", "color"):
            candidates.append(("frames", os.path.join(raw_scene_dir, dirname)))
        for filename in ("rgb_bg.mp4", "rgb.mp4", f"{scene}.mp4"):
            candidates.append(("video", os.path.join(raw_scene_dir, filename)))

    return candidates


def _load_source_preview_frames(
    src_dir,
    num_frames,
    size,
    preview_mode="fit",
    vertical_pad_ratio=0.12,
    left_crop_bias=1.0,
):
    for kind, path in _source_preview_candidates(src_dir):
        if kind == "frames" and os.path.isdir(path):
            frames = _load_image_sequence_frames(
                path,
                num_frames,
                size,
                preview_mode,
                vertical_pad_ratio=vertical_pad_ratio,
                left_crop_bias=left_crop_bias,
            )
        elif kind == "video" and os.path.isfile(path):
            frames = _load_video_frames(
                path,
                num_frames,
                size,
                preview_mode,
                vertical_pad_ratio=vertical_pad_ratio,
                left_crop_bias=left_crop_bias,
            )
        else:
            frames = None

        if frames:
            print(
                f"[Render] Using source preview with background: {path} "
                f"({preview_mode}, vertical_pad_ratio={vertical_pad_ratio}, left_crop_bias={left_crop_bias})",
                flush=True,
            )
            return frames

    print("[Render] No raw/background source preview found; falling back to processed view_0.", flush=True)
    return None


def _composite_mesh_on_white(mesh_rgba):
    rgb = mesh_rgba[..., :3]
    if rgb.numel() > 0 and float(rgb.max()) <= 1.5:
        rgb = rgb * 255.0
    rgb = rgb.clamp(0.0, 255.0)
    alpha = (mesh_rgba[..., 3:4] / 255.0).clamp(0.0, 1.0)
    bg = torch.ones_like(rgb) * 255.0
    return rgb * alpha + bg * (1.0 - alpha)


def _parse_rgb_triplet(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        value = [float(v.strip()) for v in value.split(",")]
    if len(value) != 3:
        raise ValueError(f"Expected RGB triplet, got {value}")
    color = torch.tensor(value, dtype=torch.float32)
    if float(color.max()) > 1.5:
        color = color / 255.0
    return tuple(float(v) for v in color.tolist())


def _mesh_xy_center(vertices):
    min_coords = vertices.min(dim=0).values
    max_coords = vertices.max(dim=0).values
    center = torch.zeros(3, dtype=vertices.dtype, device=vertices.device)
    center[:2] = (min_coords[:2] + max_coords[:2]) * 0.5
    return center


def _rotate_yaw(vertices, degrees, center):
    if float(degrees) == 0.0:
        return vertices
    angle = torch.tensor(float(degrees) * np.pi / 180.0, dtype=vertices.dtype, device=vertices.device)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    rotation = torch.stack(
        [
            torch.stack([cos_a, torch.zeros_like(cos_a), sin_a]),
            torch.stack([torch.zeros_like(cos_a), torch.ones_like(cos_a), torch.zeros_like(cos_a)]),
            torch.stack([-sin_a, torch.zeros_like(cos_a), cos_a]),
        ]
    )
    return (vertices - center) @ rotation.T + center


def render_anim_w_mesh(morphgs:MorphGS, proj_path, iteration,
                       device, cfg, rot_params_list=None, precomp_joints=None,
                       out_dir=None, scale_factor=None, speed=None, zoom=None,
                       render_translation=None, max_frames=None, mesh_color=None,
                       light_boost=1.0, light_location=None,
                       specular_strength=0.0, shininess=1.0,
                       render_yaw_degrees=None, save_frame_meshes=False):
    """
        Animate Mesh with the given animation field and Render the animation.
    """
    render_cfg = _cfg_get(cfg, "render")
    if scale_factor is None:
        scale_factor = float(_cfg_get(render_cfg, "scale_factor", 1.0))
    if speed is None:
        speed = float(_cfg_get(render_cfg, "speed", 2.0))
    if zoom is None:
        zoom = float(_cfg_get(render_cfg, "zoom", 1.0))
    if render_translation is None:
        render_translation = _cfg_get(render_cfg, "render_translation", (0.0, 0.0, 0.0))
    if render_yaw_degrees is None:
        render_yaw_degrees = float(_cfg_get(render_cfg, "render_yaw_degrees", 0.0))
    center_mesh = _cfg_bool(render_cfg, "center_mesh", False)
    skinning_apply_softmax = _cfg_bool(render_cfg, "skinning_apply_softmax", False)

    camreas = morphgs.cameras
    cameras_p3d = morphgs.cameras_p3d
    
    image_size = (camreas[0].image_width, camreas[0].image_height)
    renderer = Renderer(image_size[0], background_color=(1, 1, 1))
    light_boost = float(light_boost)
    if light_location is None:
        light_location = [0.0, -3.0, 4.0]
    specular_strength = float(specular_strength)
    lights = PointLights(
        device=device,
        location=[light_location],
        ambient_color=((0.62 * light_boost, 0.62 * light_boost, 0.62 * light_boost),),
        diffuse_color=((0.48 * light_boost, 0.48 * light_boost, 0.48 * light_boost),),
        specular_color=((specular_strength, specular_strength, specular_strength),),
    )
    materials = Materials(
        device=device,
        ambient_color=((min(1.0, 0.7 * light_boost), min(1.0, 0.7 * light_boost), min(1.0, 0.7 * light_boost)),),
        diffuse_color=((min(1.0, 0.8 * light_boost), min(1.0, 0.8 * light_boost), min(1.0, 0.8 * light_boost)),),
        specular_color=((specular_strength, specular_strength, specular_strength),),
        shininess=float(shininess),
    )
    mesh_color = _parse_rgb_triplet(mesh_color, None)
    mesh_texture = None
    if mesh_color is not None:
        mesh_texture = torch.ones((1, len(morphgs.mesh.vertices), 3), dtype=torch.float32, device=device)
        mesh_texture = mesh_texture * torch.tensor(mesh_color, dtype=torch.float32, device=device).view(1, 1, 3)
    
    anim_field:Animation = morphgs.animation_field
    morphgs.model.skinning_apply_softmax = skinning_apply_softmax
    original_vertices = torch.tensor(morphgs.mesh.vertices, device=device, dtype=torch.float32)
    faces = torch.tensor(morphgs.mesh.faces, device=device, dtype=torch.int64)
    original_joints = morphgs.rig.joints_pos
    if center_mesh:
        mesh_center = _mesh_xy_center(original_vertices)
        original_vertices = original_vertices - mesh_center.view(1, 3)
        original_joints = original_joints - mesh_center.view(1, 3)
        print(f"[Render] Centering target mesh in XY by {-mesh_center.detach().cpu().numpy()}", flush=True)
    if not isinstance(render_translation, torch.Tensor):
        render_translation = torch.tensor(render_translation, dtype=torch.float32, device=device)

    output_dir = proj_path.render_dir if out_dir is None else out_dir
    color_save_dir = os.path.join(output_dir, "color") if precomp_joints is None else os.path.join(output_dir, "w_precomp_joints", "color")
    mesh_save_dir = os.path.join(output_dir, "mesh") if precomp_joints is None else os.path.join(output_dir, "w_precomp_joints", "mesh")
    output_video_path = os.path.join(output_dir, f"rendered_video_{iteration}.mp4") if precomp_joints is None else os.path.join(output_dir, "w_precomp_joints", f"rendered_video_{iteration}.mp4")
    os.makedirs(color_save_dir, exist_ok=True)
    if save_frame_meshes:
        os.makedirs(mesh_save_dir, exist_ok=True)
    
    rps = []
    stacked_frames = []
    joints_animation = []
    mesh_vertices_animation = []
    source_preview_frames = None
    source_preview_mode = _cfg_get(_cfg_get(cfg, "render"), "source_preview_mode", "fit")
    source_preview_vertical_pad_ratio = float(_cfg_get(_cfg_get(cfg, "render"), "source_preview_vertical_pad_ratio", 0.12))
    source_preview_left_crop_bias = float(_cfg_get(_cfg_get(cfg, "render"), "source_preview_left_crop_bias", 1.0))
    target_preview_scale = float(_cfg_get(_cfg_get(cfg, "render"), "target_preview_scale", 1.0))
    target_preview_center = _cfg_get(_cfg_get(cfg, "render"), "target_preview_center", (0.5, 0.5))
    target_preview_center = (float(target_preview_center[0]), float(target_preview_center[1]))
    
    # Release/demo renders show the input source video view (view_0) beside the
    # transferred target mesh. Other synthesized views are training-only.
    for v, vn in enumerate(list(cameras_p3d.keys())[:1]):
        total_time = 0 
        
        # Start from 1 since the dataset was built starting from frame 1, not 0.
        num_frames = len(cameras_p3d[vn]) # Assuming all views have the same number of frames
        start_frame = 1 if (rot_params_list is None and precomp_joints is None) else 0
        end_frame = num_frames if (rot_params_list is None and precomp_joints is None) else (rot_params_list.shape[0] if rot_params_list is not None else precomp_joints.shape[0])
        if max_frames is not None:
            end_frame = min(end_frame, start_frame + int(max_frames))
        source_preview_frames = _load_source_preview_frames(
            proj_path.src_dir,
            end_frame,
            (camreas[0].image_width, camreas[0].image_height),
            source_preview_mode,
            source_preview_vertical_pad_ratio,
            source_preview_left_crop_bias,
        )
        
        for i in tqdm(range(start_frame, end_frame), desc="Rendering frames"): 
            view = camreas[v*num_frames + i]
            t = view.uid / num_frames
            
            start_time = time.time()
            
            curr_vertices = original_vertices * scale_factor
            curr_joints = original_joints * scale_factor
            skinning_weights = morphgs.rig.skinning_weights
            deform_params = anim_field.step(
                curr_vertices, curr_joints, skinning_weights, morphgs.rig,
                torch.tensor([t], dtype=torch.float, device=device),
                rot_params=rot_params_list[i] if rot_params_list is not None else None,
                precomp_joints=precomp_joints[i] if precomp_joints is not None else None,
                iteration=None,
                fixed_joints=morphgs.fixed_joint_indices,
            )
            precomp_xyz = deform_params["xyz"]  
            pred_vertices = precomp_xyz.detach()
            joints_warped = deform_params["joints_warped"]
            joints_animation.append(joints_warped)
            joints_preview = joints_warped.detach()
            preview_center = pred_vertices.mean(dim=0, keepdim=True)
            if zoom != 1.0:
                pred_vertices = preview_center + (pred_vertices - preview_center) * zoom
                joints_preview = preview_center + (joints_preview - preview_center) * zoom
            if float(render_yaw_degrees) != 0.0:
                yaw_center = pred_vertices.mean(dim=0, keepdim=True)
                pred_vertices = _rotate_yaw(pred_vertices, render_yaw_degrees, yaw_center)
                joints_preview = _rotate_yaw(joints_preview, render_yaw_degrees, yaw_center)
            pred_vertices = pred_vertices + render_translation.view(1, 3)
            joints_preview = joints_preview + render_translation.view(1, 3)
            if v == 0:
                mesh_vertices_animation.append(pred_vertices.detach().cpu().numpy())
            
            # Save the rotation parameters for visualization
            rot_params = deform_params["rot_params"]
            rps.append(rot_params.cpu().numpy())

            end_time = time.time()
            total_time += end_time - start_time
            
            cameras_p3d_view0 = cameras_p3d[vn]
            pred_rgb = renderer.render_visualization(
                pred_vertices.unsqueeze(0),
                faces.unsqueeze(0),
                cameras_p3d_view0[i],
                texture=mesh_texture,
                lights=lights,
                materials=materials,
            ).permute(0,2,3,1).cpu()[0]
            mesh_rgba = pred_rgb
            pred_rgb = _composite_mesh_on_white(mesh_rgba)
            preview_source_center = _foreground_center_np(pred_rgb.detach().cpu().numpy())
            pred_rgb = _recenter_scale_panel(
                pred_rgb,
                scale=target_preview_scale,
                center=target_preview_center,
                source_center=preview_source_center,
            )
            preview_idx = i - start_frame
            if source_preview_frames is not None and 0 <= preview_idx < len(source_preview_frames):
                src_img = source_preview_frames[preview_idx] * 255.0
            else:
                src_img = view.original_image.permute(1, 2, 0).detach().cpu() * 255.0

            joints_2d, _ = project_points_2d(joints_preview, view)
            skel_img = draw_skeleton(
                joints_2d.detach(),
                morphgs.rig.bones,
                view,
                draw_joints=False,
                line_color=SKEL_LIGHT_BLUE,
                line_width=2,
            )
            skel_rgb = torch.from_numpy(np.array(skel_img.convert("RGB"))).float()
            skel_rgb = _recenter_scale_panel(
                skel_rgb,
                scale=target_preview_scale,
                center=target_preview_center,
                source_center=preview_source_center,
            )

            stacked_vis = torch.cat([src_img, skel_rgb, pred_rgb], dim=1)
            stacked_frames.append(stacked_vis)
            Image.fromarray((stacked_vis.detach().cpu().numpy()).astype(np.uint8)).save(os.path.join(color_save_dir, f"{i}.png")) 
            if save_frame_meshes and v == 0:
                mesh_save_path = os.path.join(mesh_save_dir, f"{i:03d}.obj")
                cur_mesh = trimesh.Trimesh(
                    vertices=pred_vertices.detach().cpu().numpy(),
                    faces=faces.detach().cpu().numpy(),
                    maintain_order=True,
                    process=False,
                )
                cur_mesh.export(mesh_save_path)

    with imageio.get_writer(output_video_path, mode='I', fps=10 * speed) as writer:
        for frame in stacked_frames:
            frame = frame.detach().cpu().numpy()  # Ensure the frame is on CPU as a numpy array
            frame = frame.astype(np.uint8)
            writer.append_data(frame)
        print(f"\nVideo saved at {output_video_path}\n")

    # Save joints animation
    if precomp_joints is None:
        joints_animation = torch.stack(joints_animation, dim=0).detach().cpu().numpy()
        joints_animation_path = os.path.join(output_dir, "pred_joints.npy")
        np.save(joints_animation_path, joints_animation)
        if mesh_vertices_animation:
            mesh_sequence_path = os.path.join(output_dir, "mesh_sequence.npz")
            np.savez_compressed(
                mesh_sequence_path,
                vertices=np.stack(mesh_vertices_animation, axis=0),
                faces=faces.detach().cpu().numpy(),
            )

    rps = np.stack(rps, axis=0)
    rot_params_path = os.path.join(output_dir, "rot_params.npy")
    np.save(rot_params_path, rps)
    
    print(f"Average time per frame: {total_time / num_frames:.4f} seconds")


SKEL_LIGHT_BLUE = (135, 206, 250)  # light sky blue


if __name__ == '__main__':
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Load configuration
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('--config', required=True, help="Path to a config file. Demo aliases like demo/<experiment>.yaml are supported.")
    parser.add_argument('--out', default='output', help="Path to the output directory")
    parser.add_argument('--iteration', default=None, help="Iteration to load the model from")
    parser.add_argument('--experiment', default=None, help="Name of the experiment")
    parser.add_argument('--speed', type=float, default=None, help="Playback speed multiplier for rendered mp4/gif outputs (10fps base); 2.0 = 2x faster")
    parser.add_argument('--zoom', type=float, default=None, help="Render preview zoom about the animated geometry centroid; 1.0 keeps the learned mesh scale")
    parser.add_argument('--scale_factor', type=float, default=None, help="Scale the target animation before rendering.")
    parser.add_argument('--render_translation', type=float, nargs=3, default=None, help="World-space xyz translation applied only to the rendered mesh preview.")
    parser.add_argument('--max_frames', type=int, default=None, help="Maximum number of frames to render.")
    parser.add_argument('--mesh_color', type=float, nargs=3, default=None, help="RGB mesh color override, either 0-1 or 0-255 values.")
    parser.add_argument('--light_boost', type=float, default=1.0, help="Multiplier for render lighting intensity.")
    parser.add_argument('--light_location', type=float, nargs=3, default=None, help="Point light xyz location for mesh rendering.")
    parser.add_argument('--specular_strength', type=float, default=0.0, help="Specular color strength for both light and material.")
    parser.add_argument('--shininess', type=float, default=1.0, help="Material shininess for specular highlights.")
    parser.add_argument('--render_yaw_degrees', type=float, default=None, help="Render-only yaw rotation around the animated mesh center.")
    parser.add_argument('--save_frame_meshes', action='store_true', help="Also export one OBJ mesh per rendered frame for inspection.")

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

    # SET UP EXPERIMENT DATA
    ANIM_BASE, CHAR_BASE = get_data_bases(
        video_layout="processed",
    )

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

    for ov in unknown:
        if not ov.startswith('--') or '=' not in ov:
            continue
        parse_override_arg(ov[2:], config)

    render_speed = float(_render_arg_or_config(args, config, "speed", 2.0))
    render_zoom = float(_render_arg_or_config(args, config, "zoom", 1.0))
    render_translation = _render_arg_or_config(args, config, "render_translation", (0.0, 0.0, 0.0))

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

    if args.iteration is None:
        latest_iteration = _find_latest_iteration(proj_path.deform_dir)
        if latest_iteration is None:
            raise FileNotFoundError(
                f"No deform checkpoint found in {proj_path.deform_dir}. "
                "Run training first or pass --iteration explicitly."
            )
        args.iteration = str(latest_iteration)
        print(f"[Render] Using latest checkpoint iteration: {args.iteration}")

    set_seed(proj_cfg.seed)

    morphgs = MorphGS(
        proj_path,
        device=device,
        model_cfg=model_cfg,
        loaded_iter=args.iteration,
        render_mode=True,
        load_deform_only=True,
    )
    #######################

    torch.cuda.empty_cache()

    with torch.no_grad():
        render_anim_w_mesh(
            morphgs,
            proj_path,
            args.iteration,
            device,
            config,
            scale_factor=args.scale_factor,
            speed=render_speed,
            zoom=render_zoom,
            render_translation=render_translation,
            max_frames=args.max_frames,
            mesh_color=args.mesh_color,
            light_boost=args.light_boost,
            light_location=args.light_location,
            specular_strength=args.specular_strength,
            shininess=args.shininess,
            render_yaw_degrees=args.render_yaw_degrees,
            save_frame_meshes=args.save_frame_meshes,
        )
