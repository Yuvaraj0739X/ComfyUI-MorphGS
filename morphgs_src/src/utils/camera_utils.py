# Copyright (c) 2026 MorphGS Authors.
# Licensed under the MIT License.

import json
import os
from PIL import Image
from pytorch3d.renderer import PerspectiveCameras
import torch
import numpy as np
from feature_splatting.cameras import Camera

from .general_utils import PILtoTorch


def _view_sort_key(name: str):
    if name.startswith("view_"):
        suffix = name.split("_", 1)[1]
        try:
            return (0, int(suffix))
        except ValueError:
            return (0, suffix)
    return (1, name)


def _should_skip_view(name: str) -> bool:
    if not name.startswith("view_"):
        return False
    suffix = name.split("_", 1)[1]
    try:
        angle_deg = int(suffix)
    except ValueError:
        return False
    return angle_deg in {90, 900}


def filter_active_view_names(view_names):
    return [name for name in view_names if not _should_skip_view(name)]


def get_all_view_names(src_dir: str):
    return sorted(
        [d for d in os.listdir(src_dir) if d.startswith("view_")],
        key=_view_sort_key,
    )


def _collect_active_view_dirs(src_dir: str):
    all_view_dirs = get_all_view_names(src_dir)
    active_view_dirs = filter_active_view_names(all_view_dirs)
    skipped_view_dirs = [d for d in all_view_dirs if d not in active_view_dirs]
    return active_view_dirs, skipped_view_dirs


def get_aligned_mapping_paths(mapping_dir: str, src_dir: str):
    mapping_paths = sorted(
        [
            os.path.join(mapping_dir, f)
            for f in os.listdir(mapping_dir)
            if f.endswith(".pt")
        ]
    )
    all_view_names = get_all_view_names(src_dir)
    active_view_names = filter_active_view_names(all_view_names)
    if not all_view_names or len(all_view_names) == len(active_view_names):
        return mapping_paths

    frame_counts = []
    total_expected = 0
    for view_name in all_view_names:
        color_dir = os.path.join(src_dir, view_name, "color")
        frame_count = len(
            [
                f
                for f in os.listdir(color_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        ) if os.path.isdir(color_dir) else 0
        frame_counts.append(frame_count)
        total_expected += frame_count

    if total_expected == 0 or len(mapping_paths) == 0:
        return mapping_paths
    if len(mapping_paths) != total_expected:
        return mapping_paths

    active_view_set = set(active_view_names)
    aligned_mapping_paths = []
    offset = 0
    for view_name, frame_count in zip(all_view_names, frame_counts):
        next_offset = offset + frame_count
        if view_name in active_view_set:
            aligned_mapping_paths.extend(mapping_paths[offset:next_offset])
        offset = next_offset
    if len(aligned_mapping_paths) != len(mapping_paths):
        print(
            f"[Info] Aligned mapping paths from {len(mapping_paths)} to "
            f"{len(aligned_mapping_paths)} by skipping filtered views."
        )
    return aligned_mapping_paths

import trimesh
from pytorch3d.renderer import look_at_view_transform

def load_camera_params(proj_path, in_ndc=True, new_img_size=None, device='cuda'):
    view_dirs, skipped_view_dirs = _collect_active_view_dirs(proj_path.src_dir)
    
    if not view_dirs:
        raise ValueError(f"No view directories found in {proj_path.src_dir}.")
    if skipped_view_dirs:
        skipped_views = ", ".join(skipped_view_dirs)
        print(f"Skipping source views: {skipped_views}")
    
    cameras_dict = {}
    camera_path_0 = os.path.join(proj_path.src_dir, view_dirs[0], 'camera.json')
    
    if not os.path.exists(camera_path_0):
        camera_dir = os.path.join(proj_path.tgt_dir, 'cameras')
        camera_paths = sorted([f for f in os.listdir(camera_dir) if f.endswith('.pth')])
        if not camera_paths:
            raise FileNotFoundError(f"No .pth cameras found in {camera_dir}")
            
        base_camera = torch.load(os.path.join(camera_dir, camera_paths[0]), weights_only=False)
        FX, FY = base_camera.focal_length[0].cpu().tolist()
        CX, CY = base_camera.principal_point[0].cpu().tolist()
        imsize = new_img_size if new_img_size is not None else int(base_camera.image_size.view(-1)[0].item())
        
        mesh = trimesh.load_mesh(proj_path.tgt_obj_path)
        vertices = torch.from_numpy(mesh.vertices).float().to(device)
        min_coords, _ = torch.min(vertices, dim=0)
        max_coords, _ = torch.max(vertices, dim=0)
        mesh_center = (min_coords + max_coords) / 2
        distance = torch.norm(base_camera.T[0] - mesh_center).item()

        for vd in view_dirs:
            color_dir = os.path.join(proj_path.src_dir, vd, 'color')
            FRAME_NUM = len([f for f in os.listdir(color_dir) if f.endswith('.png')])
            
            angle_deg = int(vd.split('_')[-1])
            elev_deg = 0
            # Special view names: view_90 encodes a top view (elev +90),
            # view_900 encodes a bottom view (elev -90).
            if angle_deg == 90:
                angle_deg = 0
                elev_deg = 90
            elif angle_deg == 900:
                angle_deg = 0
                elev_deg = -90

            vR, vT = look_at_view_transform(
                dist=distance,
                elev=elev_deg, 
                azim=angle_deg,
                at=mesh_center.unsqueeze(0),
                device=device
            )
            vR, vT = vR[0], vT[0]
            
            Rs = vR.unsqueeze(0).repeat(FRAME_NUM, 1, 1).to(device)
            Ts = vT.unsqueeze(0).repeat(FRAME_NUM, 1).to(device)
            focals = torch.tensor([[FX, FY]], device=device).repeat(FRAME_NUM, 1)
            sizes = torch.tensor([[imsize, imsize]], device=device).repeat(FRAME_NUM, 1)
            
            if in_ndc:
                cameras_dict[vd] = PerspectiveCameras(
                    R=Rs, T=Ts, focal_length=focals, image_size=sizes, device=device
                )
            else:
                pps = torch.tensor([[CX, CY]], device=device).repeat(FRAME_NUM, 1)
                cameras_dict[vd] = PerspectiveCameras(
                    R=Rs, T=Ts, focal_length=focals, image_size=sizes, 
                    principal_point=pps, device=device, in_ndc=False
                )
    else:
        for vd in view_dirs:
            camera_path = os.path.join(proj_path.src_dir, vd, 'camera.json')
            if not os.path.exists(camera_path):
                continue
                
            with open(camera_path, 'r') as f:
                params_dict = json.load(f)
            
            frames = sorted(params_dict.keys(), key=lambda k: int(k))
            Rs, Ts, focals, sizes, pps = [], [], [], [], []

            for f_key in frames:
                p = params_dict[f_key]
                c2w = torch.tensor(p['extrinsics'], dtype=torch.float32)
                Rs.append(c2w[:3, :3])
                Ts.append(c2w[:3, 3])
                focals.append(p['focal_length'])
                sizes.append(p['image_size'] if new_img_size is None else [new_img_size, new_img_size])
                if not in_ndc:
                    pps.append(p['principal_point'])
            
            Rs = torch.stack(Rs).to(device)
            Ts = torch.stack(Ts).to(device)
            focals = torch.tensor(focals, device=device, dtype=torch.float32)
            sizes = torch.tensor(sizes, device=device, dtype=torch.float32)
            
            if in_ndc:
                cameras_dict[vd] = PerspectiveCameras(
                    R=Rs, T=Ts, focal_length=focals, image_size=sizes, device=device
                )
            else:
                pps = torch.tensor(pps, device=device, dtype=torch.float32)
                cameras_dict[vd] = PerspectiveCameras(
                    R=Rs, T=Ts, focal_length=focals, image_size=sizes, 
                    principal_point=pps, device=device, in_ndc=False
                )

    return cameras_dict


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def get_raster_cameras(
        cameras_dict, proj_path, data_device='cuda', in_ndc=False):
    """
    Convert PyTorch3D PerspectiveCameras (in NDC) to Gaussian Splatting Camera class.
    """
    
    raster_cam_list = []
    views = sorted(cameras_dict.keys(), key=_view_sort_key)
    cameras_list = [cameras_dict[v] for v in views]
    
    for vid, cameras in enumerate(cameras_list):
        
        image_dir = os.path.join(proj_path.src_dir, views[vid], 'color')
        mask_dir = os.path.join(proj_path.src_dir, views[vid], 'mask')
        feat_dir = os.path.join(proj_path.src_dir, views[vid], 'feature')
        thinned_dir = os.path.join(proj_path.src_dir, views[vid], 'thinned')
        
        image_paths = sorted(os.listdir(image_dir))
        mask_paths = sorted(os.listdir(mask_dir))
        thinned_paths = sorted(os.listdir(thinned_dir)) if thinned_dir is not None else None

        Rs_c2w = cameras.R.cpu().numpy()
        Ts_c2w = cameras.T.cpu().numpy()
        
        image_sizes = cameras.image_size.cpu().numpy()  # (N, 2) → (H, W)
        focal_lengths = -cameras.focal_length.cpu().numpy()  # (N, 2), Need to flip sign for Gaussian Splatting

        fx, fy = focal_lengths[:, 0], focal_lengths[:, 1]
        if image_sizes.shape[1] == 1:
            W = image_sizes[0]
            H = image_sizes[0]
        else:
            W, H = image_sizes[0]

        if in_ndc:
            FoVx = 2 * np.arctan(0.5 * 2 / fx) # [-1,1]
            FoVy = 2 * np.arctan(0.5 * 2 / fy)
        else:
            FoVx = 2 * np.arctan(0.5 * W / fx)
            FoVy = 2 * np.arctan(0.5 * H / fy) 
            
        for frame in range(len(cameras)):
            # Load images and masks
            gt_image = Image.open(os.path.join(image_dir, image_paths[frame])).convert('RGB')   
            gt_image = PILtoTorch(gt_image, (H, W))
            
            frame_name = os.path.basename(image_paths[frame])

            loaded_mask = None
            if mask_dir is not None:
                mask_path = os.path.join(mask_dir, mask_paths[frame])
                loaded_mask = Image.open(mask_path).convert("L")
                loaded_mask = loaded_mask.resize((W, H), Image.BILINEAR)
                mask_array = np.array(loaded_mask, dtype=np.float32)
                mask_array /= 255.0  # 0~255 -> 0~1
                loaded_mask = torch.from_numpy(mask_array).unsqueeze(0)  # (H, W) -> (1, H, W)
            else:
                # make mask from image alpha channel
                if gt_image.shape[0] == 4:
                    loaded_mask = gt_image[3:4, ...]
                else:
                    loaded_mask = None

            if feat_dir is not None:
                feat_path = os.path.join(feat_dir, frame_name.replace('.png', '_feat.pt'))

            thinned = None
            if thinned_dir is not None and os.listdir(thinned_dir):
                thinned_path = os.path.join(thinned_dir, thinned_paths[frame])
                thinned = Image.open(thinned_path).convert('L')
                thinned = PILtoTorch(thinned, (H, W))
            
            if '_' in frame_name:
                img_uid = int(frame_name.split('_')[-1].split('.')[0])
            else:
                img_uid = frame

            raster_cam_list.append(Camera(
                colmap_id=vid,
                R=Rs_c2w[frame],  
                T=Ts_c2w[frame],
                FoVx=FoVx[frame],
                FoVy=FoVy[frame],
                image=gt_image,
                gt_alpha_mask=loaded_mask,
                image_name=frame_name,
                uid=img_uid,
                feat_path=feat_path if feat_dir is not None else None,
                mask_path=mask_path if mask_dir is not None else None,
                thinned=thinned,
                data_device=data_device
            ))

    nerf_normalization = getNerfppNorm(raster_cam_list)

    return raster_cam_list, nerf_normalization
