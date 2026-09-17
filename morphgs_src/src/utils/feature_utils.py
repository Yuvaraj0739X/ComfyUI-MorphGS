# Copyright (c) 2026 MorphGS Authors.
# Licensed under the MIT License.
#
# DINOv2-only replacement for the upstream GeoAware-SC-based feature_utils.py.
#
# The original set_feature_extraction()/get_processed_features() fused Stable-Diffusion-UNet
# features (sourced from ODISE's internal diffusion model) with DINOv2 ViT features via a
# pretrained AggregationNetwork. That SD half requires ODISE + full Detectron2 + Mask2Former,
# none of which GeoAware-SC ships a working extlibs/GeoAware for (its own feature_utils.py
# module referenced by that repo does not exist upstream either).
#
# This version drops the SD/ODISE half and the aggregation network entirely, and instead uses
# raw, L2-normalized DINOv2 patch tokens directly as the dense descriptor -- the same technique
# used by plain "deep ViT features as dense visual descriptors" correspondence methods. Lower
# fidelity than the paper's fused descriptor, but has zero dependency beyond torch/torchvision
# and downloads its backbone directly from torch.hub.

import os
import sys
import time
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.general_utils import PILtoTorch
from utils.render_utils import project_points_3d_to_2d

_DINO_MODEL_NAME = "dinov2_vitb14"
_DINO_PATCH_SIZE = 14

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _resize_for_patches(img, num_patches, patch_size):
    target = num_patches * patch_size
    return img.resize((target, target), Image.BICUBIC)


def _preprocess(img, device):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    t = (t - _IMAGENET_MEAN.to(device)) / _IMAGENET_STD.to(device)
    return t


def set_feature_extraction(num_patches=60, weights_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", _DINO_MODEL_NAME)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # sd_model, sd_aug, aggre_net slots are kept in the return tuple only to preserve the
    # call signature used by preprocess_src.py / preprocess_tgt.py; unused in this path.
    return None, None, model, None, num_patches


def get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=None, img_path=None, resolution=None):
    device = next(extractor_vit.parameters()).device
    if img is None:
        img = Image.open(img_path).convert("RGB")
    img_in = _resize_for_patches(img, num_patches, _DINO_PATCH_SIZE)
    x = _preprocess(img_in, device)
    with torch.no_grad():
        feats = extractor_vit.forward_features(x)
        patch_tokens = feats["x_norm_patchtokens"]  # (1, N, C)
    C = patch_tokens.shape[-1]
    desc = patch_tokens.permute(0, 2, 1).reshape(1, C, num_patches, num_patches)
    norms = torch.linalg.norm(desc, dim=1, keepdim=True)
    desc = desc / (norms + 1e-8)
    return desc


def extract_feature(img_dir, feat_dir, sd_model, sd_aug, aggre_net, extractor_vit, num_patches, raw_feat_dir=None):
    with torch.no_grad():
        for fname in sorted(os.listdir(img_dir)):
            if fname.endswith(".png"):
                print(f"Extracting features for image: {fname}")
                render_img_path = os.path.join(img_dir, fname)
                render_img = Image.open(render_img_path).convert("RGB")
                render_feat = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=render_img)
                feat_path = os.path.join(feat_dir, fname.replace(".png", "_feat.pt"))
                torch.save(render_feat, feat_path)
                torch.cuda.empty_cache()

    print("Feature extraction completed (DINOv2-only path).")


def _normalize_vis_map(arr):
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.uint8)
    vals = arr[finite]
    lo = float(vals.min())
    hi = float(vals.max())
    if hi - lo < 1e-8:
        out = np.zeros_like(arr, dtype=np.float32)
        out[finite] = 1.0
    else:
        out = np.zeros_like(arr, dtype=np.float32)
        out[finite] = (arr[finite] - lo) / (hi - lo)
    return (out * 255.0).clip(0, 255).astype(np.uint8)


def _save_mapping_debug_frame(debug_dir, frame_idx, src_mask, src_feat_map, pseudo_gt, conf):
    os.makedirs(debug_dir, exist_ok=True)
    src_mask_np = src_mask.detach().cpu().numpy().astype(np.uint8) * 255
    feat_energy = torch.linalg.norm(src_feat_map.detach(), dim=0).cpu().numpy()
    feat_energy_u8 = _normalize_vis_map(feat_energy)
    pseudo_gt_np = pseudo_gt.detach().cpu().numpy()
    pseudo_gt_occ = (pseudo_gt_np >= 0).astype(np.uint8) * 255
    conf_np = conf.detach().cpu().numpy()
    conf_u8 = _normalize_vis_map(conf_np)

    cv2.imwrite(os.path.join(debug_dir, f"{frame_idx:04d}_mask.png"), src_mask_np)
    cv2.imwrite(os.path.join(debug_dir, f"{frame_idx:04d}_feat_energy.png"), feat_energy_u8)
    cv2.imwrite(os.path.join(debug_dir, f"{frame_idx:04d}_pseudo_occ.png"), pseudo_gt_occ)
    cv2.imwrite(os.path.join(debug_dir, f"{frame_idx:04d}_conf.png"), conf_u8)

    overlay = np.zeros((src_mask_np.shape[0], src_mask_np.shape[1], 3), dtype=np.uint8)
    overlay[..., 1] = src_mask_np
    overlay[..., 2] = pseudo_gt_occ
    cv2.imwrite(os.path.join(debug_dir, f"{frame_idx:04d}_mask_vs_pseudo.png"), overlay)


def get_tgt_mapping(proj_points_batch, visibility_mask_batch, H, W, fg_mask):
    vtx2px = proj_points_batch.permute(1, 0, 2)
    for view_idx in range(proj_points_batch.shape[0]):
        if visibility_mask_batch is None:
            break
        mask = visibility_mask_batch[view_idx]
        coords = vtx2px[:, view_idx, :]
        coords[mask == 0] = -1
        vtx2px[:, view_idx, :] = coords
    vtx2px = vtx2px.long()

    _, NR, _ = vtx2px.shape
    device = vtx2px.device
    px2vtx = -torch.ones((NR, H, W), dtype=torch.long, device=device)
    fill_iters = 5
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for rendered in range(NR):
        xs = vtx2px[:, rendered, 0]
        ys = vtx2px[:, rendered, 1]
        xs = xs.clamp(0, W - 1)
        ys = ys.clamp(0, H - 1)
        valid = ((xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)) & (fg_mask[rendered, ys, xs] > 0)
        vs = torch.nonzero(valid, as_tuple=False).squeeze(1)
        px2vtx[rendered, ys[vs], xs[vs]] = vs

        mask_unmapped = fg_mask[rendered] & (px2vtx[rendered] == -1)
        for _ in range(fill_iters):
            updated = False
            for dx, dy in neighbors:
                shifted = torch.roll(px2vtx[rendered], shifts=(dy, dx), dims=(0, 1))
                m = mask_unmapped & (shifted != -1)
                if m.any():
                    px2vtx[rendered][m] = shifted[m]
                    updated = True
            if not updated:
                break
            mask_unmapped = fg_mask[rendered] & (px2vtx[rendered] == -1)

    return vtx2px, px2vtx


def get_src2tgt_mapping(
    src_feats_batch,
    src_fg_masks,
    tgt_feats_batch,
    tgt_masks,
    px_to_vtx,
    out_dir,
    temp=0.07,
    src_chunk_size=2048,
    work_device=None,
    debug_dir=None,
    debug_frame_indices=None,
    resolution=None,
):
    if isinstance(src_feats_batch, (list, tuple)):
        NF = len(src_feats_batch)
        if NF == 0:
            raise RuntimeError("src_feats_batch path list is empty.")
    else:
        NF, _, _, _ = src_feats_batch.shape
    if isinstance(tgt_feats_batch, (list, tuple)):
        NV = len(tgt_feats_batch)
        if NV == 0:
            raise RuntimeError("tgt_feats_batch path list is empty.")
    else:
        NV, _, _, _ = tgt_feats_batch.shape

    if isinstance(src_fg_masks, (list, tuple)):
        if len(src_fg_masks) != NF:
            raise RuntimeError("src_fg_masks path list length does not match source feature count.")
        if resolution is not None:
            H, W = int(resolution[0]), int(resolution[1])
        else:
            first_mask = np.array(Image.open(src_fg_masks[0]).convert("L"))
            H, W = first_mask.shape[:2]
    else:
        NF, H, W = src_fg_masks.shape

    device = work_device if work_device is not None else px_to_vtx.device
    debug_frame_indices = set(debug_frame_indices or [])
    os.makedirs(out_dir, exist_ok=True)

    tgt_feats_loaded = None
    if isinstance(tgt_feats_batch, (list, tuple)):
        print(f"[Mapping] Loading {NV} target feature maps once...", flush=True)
        load_start = time.time()
        tgt_feats_loaded = [torch.load(path, map_location="cpu", weights_only=False) for path in tgt_feats_batch]
        print(f"[Mapping] Loaded target feature maps in {time.time() - load_start:.1f}s", flush=True)

    existing = sum(1 for n in range(NF) if os.path.exists(os.path.join(out_dir, f"{n:04d}.pt")))
    if existing:
        print(f"[Mapping] Reusing {existing}/{NF} existing source-to-target maps in {out_dir}", flush=True)

    progress_every = max(1, int(os.environ.get("MORPHGS_MAPPING_PROGRESS_EVERY", "25")))
    progress_start = time.time()

    for n in range(NF):
        mapping_path = os.path.join(out_dir, f"{n:04d}.pt")
        if os.path.exists(mapping_path):
            if (n + 1) % progress_every == 0 or n + 1 == NF:
                elapsed = time.time() - progress_start
                print(f"[Mapping] {n + 1}/{NF} frames checked ({elapsed:.1f}s)", flush=True)
            continue

        if isinstance(src_fg_masks, (list, tuple)):
            src_mask_n = PILtoTorch(Image.open(src_fg_masks[n]).convert("L"), (H, W)).squeeze(0)
            src_mask_n = (src_mask_n > 0).to(device=device)
        else:
            src_mask_n = src_fg_masks[n].to(device=device)
        ys, xs = (src_mask_n > 0).nonzero(as_tuple=True)

        P_src = ys.numel()
        if P_src == 0:
            pseudo_gt = torch.full((H, W), -1, dtype=torch.long, device=device)
            conf_dtype = torch.float32 if isinstance(tgt_feats_batch, (list, tuple)) else tgt_feats_batch.dtype
            conf = torch.zeros((H, W), device=device, dtype=conf_dtype)
            torch.save({"pseudo_gt": pseudo_gt, "conf": conf}, mapping_path)
            continue

        src_feat_n = torch.load(src_feats_batch[n], map_location="cpu", weights_only=False) if isinstance(src_feats_batch, (list, tuple)) else src_feats_batch[n:n + 1]
        best_dtype = src_feat_n.dtype if torch.is_tensor(src_feat_n) else torch.float32
        best_conf = torch.full((P_src,), -float("inf"), device=device, dtype=best_dtype)
        best_vtx = torch.full((P_src,), -1, dtype=torch.long, device=device)
        vf_n = F.normalize(src_feat_n.to(device=device, non_blocking=True), dim=1)
        vf_n = F.interpolate(vf_n, size=(H, W), mode="bilinear", align_corners=True).squeeze(0)
        v_feats = vf_n[:, ys, xs].permute(1, 0)

        for i in range(NV):
            tgt_mask_i = tgt_masks[i].to(device=device)
            r_ys, r_xs = (tgt_mask_i > 0).nonzero(as_tuple=True)
            Ri = r_ys.numel()
            if Ri == 0:
                continue

            tgt_feat_i = tgt_feats_loaded[i] if isinstance(tgt_feats_batch, (list, tuple)) else tgt_feats_batch[i:i + 1]
            rf_i = F.normalize(tgt_feat_i.to(device=device, non_blocking=True), dim=1)
            rf_i = F.interpolate(rf_i, size=(H, W), mode="bilinear", align_corners=True).squeeze(0)

            r_feats = rf_i[:, r_ys, r_xs].permute(1, 0)
            vtx_idx = px_to_vtx[i, r_ys, r_xs].reshape(Ri)
            valid_render = vtx_idx >= 0
            if not valid_render.any():
                continue
            r_feats = r_feats[valid_render]
            vtx_idx = vtx_idx[valid_render]

            vals_chunks = []
            idxs_chunks = []
            for start in range(0, P_src, src_chunk_size):
                end = min(start + src_chunk_size, P_src)
                sims_chunk = v_feats[start:end] @ r_feats.T
                sims_chunk /= temp
                vals_chunk, idxs_chunk = sims_chunk.max(dim=1)
                vals_chunks.append(vals_chunk)
                idxs_chunks.append(idxs_chunk)

            vals = torch.cat(vals_chunks, dim=0)
            idxs = torch.cat(idxs_chunks, dim=0)
            better = vals > best_conf
            best_conf[better] = vals[better]
            best_vtx[better] = vtx_idx[idxs[better]]

        pseudo_gt = torch.full((H, W), -1, dtype=torch.long, device=device)
        conf = torch.full((H, W), -float("inf"), device=device, dtype=best_conf.dtype)
        pseudo_gt[ys, xs] = best_vtx
        conf[ys, xs] = torch.sigmoid(best_conf)

        torch.save({"pseudo_gt": pseudo_gt, "conf": conf}, mapping_path)
        if debug_dir is not None and (len(debug_frame_indices) == 0 or n in debug_frame_indices):
            _save_mapping_debug_frame(debug_dir, n, src_mask_n, vf_n, pseudo_gt, conf)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (n + 1) % progress_every == 0 or n + 1 == NF:
            elapsed = time.time() - progress_start
            print(f"[Mapping] {n + 1}/{NF} frames processed ({elapsed:.1f}s)", flush=True)


@torch.no_grad()
def aggregate_features_3d(
    points_3d,
    tgt_feat_batch,
    src_feat_batch,
    camera_params,
    visibility_mask_batch,
    tgt_fg_masks,
    src_fg_masks,
    mapping_dir,
    color_dir=None,
    visualize_path=None,
    resolution=(256, 256),
    mapping_debug_dir=None,
    mapping_debug_frame_indices=None,
):
    points_2d_batch = project_points_3d_to_2d(points_3d, resolution, camera_params)
    _, px_to_vtx = get_tgt_mapping(points_2d_batch, visibility_mask_batch, resolution[0], resolution[1], tgt_fg_masks)
    work_device = points_3d.device
    get_src2tgt_mapping(
        src_feat_batch,
        src_fg_masks,
        tgt_feat_batch,
        tgt_fg_masks,
        px_to_vtx,
        mapping_dir,
        work_device=work_device,
        debug_dir=mapping_debug_dir,
        debug_frame_indices=mapping_debug_frame_indices,
        resolution=resolution,
    )

    image_height, image_width = resolution
    points_2d_batch[:, :, 0] = 2.0 * (points_2d_batch[:, :, 0] / image_width) - 1.0
    points_2d_batch[:, :, 1] = 2.0 * (points_2d_batch[:, :, 1] / image_height) - 1.0

    feats_sum = None
    feats_count = None
    num_tgt_views = len(tgt_feat_batch) if isinstance(tgt_feat_batch, (list, tuple)) else tgt_feat_batch.shape[0]
    for b in range(num_tgt_views):
        tgt_feat_b = torch.load(tgt_feat_batch[b], map_location="cpu", weights_only=False) if isinstance(tgt_feat_batch, (list, tuple)) else tgt_feat_batch[b:b + 1]
        matching_feat_b = F.interpolate(
            tgt_feat_b.to(device=work_device, non_blocking=True),
            size=(image_height, image_width),
            mode="bilinear",
            align_corners=False,
        )
        points_2d_grid_b = points_2d_batch[b:b + 1].unsqueeze(2).to(dtype=matching_feat_b.dtype)
        feats_3d_b = F.grid_sample(
            matching_feat_b,
            points_2d_grid_b,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(-1).transpose(0, 1)

        if visibility_mask_batch is not None:
            vis_b = visibility_mask_batch[b].unsqueeze(-1).to(device=feats_3d_b.device, dtype=feats_3d_b.dtype)
        else:
            vis_b = torch.ones(feats_3d_b.shape[0], 1, device=feats_3d_b.device, dtype=feats_3d_b.dtype)

        if feats_sum is None:
            feats_sum = feats_3d_b * vis_b
            feats_count = vis_b
        else:
            feats_sum = feats_sum + feats_3d_b * vis_b
            feats_count = feats_count + vis_b

    return feats_sum / (feats_count + 1e-8)


if __name__ == "__main__":
    img_path = "data/DeformingThings4D/animals/bear3EP_Agression/screenshots/00001.jpg"
    img = Image.open(img_path).convert("RGB")

    num_patches = 60
    sd_model, sd_aug, extractor_vit, aggre_net, num_patches = set_feature_extraction(num_patches)
    feat = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=img, img_path=img_path)
    print(feat.shape)
