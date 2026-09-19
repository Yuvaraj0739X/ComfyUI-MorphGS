# Copyright (c) 2026 MorphGS Authors.
# Licensed under the MIT License.

import sys
import os
import shutil
import argparse
import time
import torch
import numpy as np
import cv2
from PIL import Image
from glob import glob
from tqdm import tqdm
import imageio

from contextlib import contextmanager

# Add generative-models to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../extlibs/generative-models"))

from scripts.demo.sv4d_helpers import (
    load_model,
    preprocess_video,
    read_video,
    run_img2vid,
)
from sgm.modules.encoders.modules import VideoPredictionEmbedderWithEncoder

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".PNG", ".JPG", ".JPEG", ".BMP", ".WEBP"}
VIDEO_EXTS = {".mp4", ".gif", ".mov", ".avi", ".mkv", ".webm", ".MP4", ".GIF", ".MOV", ".AVI", ".MKV", ".WEBM"}


sp4d_configs = {
    "sp4d": {
        "T": 4,  # number of frames per sample
        "V": 12,  # number of views per sample
        "model_config": "scripts/sampling/configs/sp4d.yaml",
        "version_dict": {
            "T": 48,
            "options": {
                "discretization": 1,
                "cfg": 3.0,
                "min_cfg": 1.5,
                "num_views": 12,
                "sigma_min": 0.002,
                "sigma_max": 700.0,
                "rho": 7.0,
                "guider": 2,
                "force_uc_zero_embeddings": [
                    "cond_frames",
                    "cond_frames_without_noise",
                    "cond_view",
                    "cond_motion",
                ],
                "additional_guider_kwargs": {
                    "additional_cond_keys": ["cond_view", "cond_motion"]
                },
            },
        },
    },
}


sv4d2_configs = {
    "sv4d2": {
        "T": 12,  # number of frames per sample
        "V": 4,  # number of views per sample
        "model_config": "scripts/sampling/configs/sv4d2.yaml",
        "version_dict": {
            "T": 12 * 4,
            "options": {
                "discretization": 1,
                "cfg": 2.0,
                "min_cfg": 2.0,
                "num_views": 4,
                "sigma_min": 0.002,
                "sigma_max": 700.0,
                "rho": 7.0,
                "guider": 2,
                "force_uc_zero_embeddings": [
                    "cond_frames",
                    "cond_frames_without_noise",
                    "cond_view",
                    "cond_motion",
                ],
                "additional_guider_kwargs": {
                    "additional_cond_keys": ["cond_view", "cond_motion"]
                },
            },
        },
    },
    "sv4d2_8views": {
        "T": 5,  # number of frames per sample
        "V": 8,  # number of views per sample
        "model_config": "scripts/sampling/configs/sv4d2_8views.yaml",
        "version_dict": {
            "T": 5 * 8,
            "options": {
                "discretization": 1,
                "cfg": 2.5,
                "min_cfg": 1.5,
                "num_views": 8,
                "sigma_min": 0.002,
                "sigma_max": 700.0,
                "rho": 7.0,
                "guider": 5,
                "force_uc_zero_embeddings": [
                    "cond_frames",
                    "cond_frames_without_noise",
                    "cond_view",
                    "cond_motion",
                ],
                "additional_guider_kwargs": {
                    "additional_cond_keys": ["cond_view", "cond_motion"]
                },
            },
        },
    },
}


def sample_sp4d(
    input_path: str,
    output_folder: str,
    model_path: str = "checkpoints/sp4d.safetensors",
    num_steps: int = 50,
    img_size: int = 512,
    n_frames: int = 21,
    seed: int = 23,
    encoding_t: int = 8,
    decoding_t: int = 4,
    device: str = "cuda",
    elevations_deg=0.0,
    azimuths_deg=[0, 60, 120, 180, 240], # Fixed views
    image_frame_ratio=0.9,
    verbose=False,
    remove_bg=False,
):
    sp4d_model = "sp4d" 
    config = sp4d_configs[sp4d_model]
    T = config["T"]
    V = config["V"]
    model_config = config["model_config"]
    version_dict = config["version_dict"]
    H, W = img_size, img_size
    F = 8
    C = 4
    n_views = V + 1 # 13
    
    subsampled_views = np.arange(n_views)
    version_dict["H"] = H
    version_dict["W"] = W
    version_dict["C"] = C
    version_dict["f"] = F
    version_dict["options"]["num_steps"] = num_steps

    torch.manual_seed(seed)
    os.makedirs(output_folder, exist_ok=True)

    # Read input video
    print(f"Reading {input_path}")
    processed_input_path = preprocess_video(
        input_path,
        remove_bg=remove_bg,
        n_frames=n_frames,
        W=W,
        H=H,
        output_folder=output_folder, # Intermediate processing
        image_frame_ratio=image_frame_ratio,
    )
    images_v0 = read_video(processed_input_path, n_frames=n_frames, device=device)
    images_v0 = [img.half() for img in images_v0]
    images_t0 = torch.zeros(n_views, 3, H, W).half().to(device)

    elevations_deg = [0.0] * n_views
    azimuths_deg = np.linspace(0, 360, n_views + 1)[1:] % 360 # 30, 60, ..., 0
    
    polars_rad = np.array([np.deg2rad(90 - e) for e in elevations_deg])
    azimuths_rad = np.array([np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg])

    img_matrix = [[None] * n_views for _ in range(n_frames)]
    for i, v in enumerate(subsampled_views):
        img_matrix[0][i] = images_t0[v].unsqueeze(0).half()
    for t in range(n_frames):
        img_matrix[t][0] = images_v0[t].half()

    gen_models_path = os.path.join(os.path.dirname(__file__), "../extlibs/generative-models")
    model_config_path = os.path.join(gen_models_path, model_config)
    
    print(f"Loading model from {model_config_path}")
    
    model, _ = load_model(
        model_config_path,
        device,
        version_dict["T"],
        num_steps,
        verbose,
        model_path,
    )
    model.to(device).half()
    model.en_and_decode_n_samples_a_time = decoding_t
    for emb in model.conditioner.embedders:
        if isinstance(emb, VideoPredictionEmbedderWithEncoder):
            emb.en_and_decode_n_samples_a_time = encoding_t

    # Sampling
    v0 = 0
    view_indices = np.arange(V) + 1
    t0_list = range(0, n_frames - T + 1, T - 1)
    
    for t0 in tqdm(t0_list):
        if t0 + T > n_frames:
            t0 = n_frames - T
        frame_indices = t0 + np.arange(T)
        
        image = img_matrix[t0][v0]
        cond_motion = torch.cat([img_matrix[t][v0] for t in frame_indices], 0)
        cond_view = torch.cat([img_matrix[t0][v] for v in view_indices], 0)
        
        polars = polars_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
        azims = azimuths_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
        
        polars = (polars - polars_rad[v0] + torch.pi / 2) % (torch.pi * 2)
        azims = (azims - azimuths_rad[v0]) % (torch.pi * 2)
        
        samples = run_img2vid(
            version_dict,
            model,
            image,
            seed,
            polars,
            azims,
            cond_motion,
            cond_view,
            decoding_t,
            cond_mv=False,
            part_maps=True, 
        )
        
        samples = samples.view(T, V, 3, H, -1) # (T, V, 3, H, 2W)

        # Store the full tensor (RGB and part maps concatenated along width); split at save time.
        for i, t in enumerate(frame_indices):
            for j, v in enumerate(view_indices):
                img_matrix[t][v] = samples[i, j][None] * 2 - 1
    
    target_views_indices = {
        60: 2,
        120: 4,
        180: 6,
        240: 8
    }
    
    vid_name = os.path.basename(input_path).split('.')[0]

    # Save View 0 (GT)
    base_dir_v0 = os.path.join(output_folder, vid_name, "view_0")
    color_dir_v0 = os.path.join(base_dir_v0, "color")
    os.makedirs(color_dir_v0, exist_ok=True)
    print(f"Saving view 0 (GT) to {base_dir_v0}")
    
    for t in range(n_frames):
        img_tensor = img_matrix[t][0] # View 0 is at index 0
        if img_tensor is None:
            continue
        img_np = (((img_tensor.permute(1, 2, 0) + 1) / 2).cpu().numpy() * 255.0).astype(np.uint8)
        cv2.imwrite(os.path.join(color_dir_v0, f"{t:03d}.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    # Save Generated Views
    for deg, v_idx in target_views_indices.items():
        base_dir = os.path.join(output_folder, vid_name, f"view_{deg}")
        color_dir = os.path.join(base_dir, "color")
        parts_dir = os.path.join(base_dir, "parts")
        os.makedirs(color_dir, exist_ok=True)
        os.makedirs(parts_dir, exist_ok=True)
        
        print(f"Saving view {deg} to {base_dir}")
        
        for t in range(n_frames):
            img_tensor = img_matrix[t][v_idx]
            if img_tensor is None:
                continue
                
            img_np = (((img_tensor[0].permute(1, 2, 0) + 1) / 2).cpu().numpy() * 255.0).astype(np.uint8)
            H_img, W_total, C_img = img_np.shape
            W_img = W_total // 2
            
            rgb_img = img_np[:, :W_img, :]
            parts_img = img_np[:, W_img:, :]

            # Save
            cv2.imwrite(os.path.join(color_dir, f"{t:03d}.png"), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(parts_dir, f"{t:03d}.png"), cv2.cvtColor(parts_img, cv2.COLOR_RGB2BGR))

    # Cleanup intermediate mp4
    if os.path.exists(processed_input_path):
        os.remove(processed_input_path)

import gc 
def sample_sv4d(
    input_path: str,
    output_folder: str,
    model_path: str = "checkpoints/sv4d2.safetensors",
    num_steps: int = 50,
    img_size: int = 512,
    n_frames: int = 21,
    seed: int = 23,
    encoding_t: int = 8,
    decoding_t: int = 4,
    device: str = "cuda",
    elevations_deg=0.0,
    azimuths_deg=None,
    image_frame_ratio=0.9,
    verbose=False,
    remove_bg=False,
):
    """
    Simple script to generate multiple novel-view videos conditioned on a video `input_path` or multiple frames, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t` and `encoding_t`.
    """
    # Set model config
    assert os.path.basename(model_path) in [
        "sv4d2.safetensors",
        "sv4d2_8views.safetensors",
    ]
    sv4d2_model = os.path.splitext(os.path.basename(model_path))[0]
    config = sv4d2_configs[sv4d2_model]
    print(sv4d2_model, config)
    T = config["T"]
    V = config["V"]
    model_config = config["model_config"]
    version_dict = config["version_dict"]
    F = 8  # vae factor to downsize image->latent
    C = 4
    H, W = img_size, img_size
    n_views = V + 1  # number of output video views (1 input view + 8 novel views)
    subsampled_views = np.arange(n_views)
    version_dict["H"] = H
    version_dict["W"] = W
    version_dict["C"] = C
    version_dict["f"] = F
    version_dict["options"]["num_steps"] = num_steps

    torch.manual_seed(seed)
    os.makedirs(output_folder, exist_ok=True)
    processed_input_path = None
    
    try:
        # Read input video frames i.e. images at view 0
        print(f"Reading {input_path}")
        base_count = len(glob(os.path.join(output_folder, "*.mp4"))) // n_views
        processed_input_path = preprocess_video(
            input_path,
            remove_bg=remove_bg,
            n_frames=n_frames,
            W=W,
            H=H,
            output_folder=output_folder,
            image_frame_ratio=image_frame_ratio,
            base_count=base_count,
        )
        images_v0 = read_video(processed_input_path, n_frames=n_frames, device=device)
        images_t0 = torch.zeros(n_views, 3, H, W).float().to(device)

        # Get camera viewpoints
        if isinstance(elevations_deg, float) or isinstance(elevations_deg, int):
            elevations_deg = [elevations_deg] * n_views
        assert (
            len(elevations_deg) == n_views
        ), f"Please provide 1 value, or a list of {n_views} values for elevations_deg! Given {len(elevations_deg)}"
        if azimuths_deg is None:
            azimuths_deg = (
                np.array([0, 60, 120, 180, 240])
                if sv4d2_model == "sv4d2"
                else np.array([0, 30, 75, 120, 165, 210, 255, 300, 330])
            )
        assert (
            len(azimuths_deg) == n_views
        ), f"Please provide a list of {n_views} values for azimuths_deg! Given {len(azimuths_deg)}"
        polars_rad = np.array([np.deg2rad(90 - e) for e in elevations_deg])
        azimuths_rad = np.array(
            [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
        )

        # Initialize image matrix
        img_matrix = [[None] * n_views for _ in range(n_frames)]
        for i, v in enumerate(subsampled_views):
            img_matrix[0][i] = images_t0[v].unsqueeze(0)
        for t in range(n_frames):
            img_matrix[t][0] = images_v0[t]

        # Load SV4D++ model
        model, _ = load_model(
            model_config,
            device,
            version_dict["T"],
            num_steps,
            verbose,
            model_path,
        )
        model.to(device).half()  # fp32 ckpt (~12GB) OOMs at 512 on 24GB; mirror sample_sp4d
        model.en_and_decode_n_samples_a_time = decoding_t
        for emb in model.conditioner.embedders:
            if isinstance(emb, VideoPredictionEmbedderWithEncoder):
                emb.en_and_decode_n_samples_a_time = encoding_t

        # Sampling novel-view videos
        v0 = 0
        view_indices = np.arange(V) + 1
        t0_list = (
            range(0, n_frames, T-1)
            if sv4d2_model == "sv4d2"
            else range(0, n_frames - T + 1, T - 1)
        )
        for t0 in tqdm(t0_list):
            if t0 + T > n_frames:
                t0 = n_frames - T
            frame_indices = t0 + np.arange(T)
            
            print(f"Sampling frames {frame_indices}")
            
            image = img_matrix[t0][v0].to(device).to(next(model.parameters()).dtype)
            cond_motion = torch.cat([img_matrix[t][v0].to(device).to(next(model.parameters()).dtype) for t in frame_indices], 0)
            cond_view = torch.cat([img_matrix[t0][v].to(device).to(next(model.parameters()).dtype) for v in view_indices], 0)
            polars = polars_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
            azims = azimuths_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
            polars = (polars - polars_rad[v0] + torch.pi / 2) % (torch.pi * 2)
            azims = (azims - azimuths_rad[v0]) % (torch.pi * 2)
            cond_mv = False if t0 == 0 else True
            with torch.no_grad():
                samples = run_img2vid(
                    version_dict,
                    model,
                    image,
                    seed,
                    polars,
                    azims,
                    cond_motion,
                    cond_view,
                    decoding_t,
                    cond_mv=cond_mv,
                )
            samples = samples.detach().view(T, V, 3, H, W)
            
            for i, t in enumerate(frame_indices):
                for j, v in enumerate(view_indices):
                    img_matrix[t][v] = (samples[i, j][None] * 2 - 1).cpu()
                    
            del samples, cond_motion, cond_view
            gc.collect()
            torch.cuda.empty_cache()

        # Save generated views as image sequences
        for v in range(n_views):
            img_folder = os.path.join(output_folder, f"view_{int(azimuths_deg[v])}", "color")
            os.makedirs(img_folder, exist_ok=True)
            for t in range(n_frames):
                if img_matrix[t][v] is not None:
                    img_file = os.path.join(img_folder, f"{t:03d}.png")
                    img_to_save = img_matrix[t][v]
                    if hasattr(img_to_save, 'detach'):
                        img_to_save = img_to_save.squeeze().permute(1, 2, 0).detach().cpu().float()
                        img_to_save = (img_to_save + 1.0) / 2.0
                        img_to_save = img_to_save.clamp(0, 1).numpy()
                        img_to_save = (img_to_save * 255).astype('uint8')
                    
                    imageio.imwrite(img_file, img_to_save)

    # Cleanup intermediate mp4
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise
    except Exception:
        raise
    finally:
        if processed_input_path and os.path.exists(processed_input_path):
            try:
                os.remove(processed_input_path)
                print(f"Successfully deleted: {processed_input_path}")
            except Exception as cleanup_err:
                print(f"Failed to delete {processed_input_path}: {cleanup_err}")
                
        temp_files = glob(os.path.join(output_folder, "*_process_input.mp4"))
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
                if verbose:
                    print(f"Deleted intermediate file: {f}")
                

def _list_image_frames(src_video_path):
    if not os.path.isdir(src_video_path):
        return []
    return sorted(
        [
            f for f in os.listdir(src_video_path)
            if os.path.isfile(os.path.join(src_video_path, f))
            and os.path.splitext(f)[1] in IMAGE_EXTS
        ]
    )


def _build_file_input_dir(src_video_path, sparse_dir, stride=1, max_frames=None):
    stride = max(1, int(stride))
    max_frames = int(max_frames) if max_frames is not None and max_frames > 0 else None
    os.makedirs(sparse_dir, exist_ok=True)

    ext = os.path.splitext(src_video_path)[1]
    if ext in IMAGE_EXTS:
        dst_file = os.path.join(sparse_dir, "00000.png")
        Image.open(src_video_path).convert("RGB").save(dst_file)
        return 1

    if ext not in VIDEO_EXTS:
        raise ValueError(f"Unsupported source input file type: {src_video_path}")

    reader = imageio.get_reader(src_video_path)
    written = 0
    try:
        for frame_idx, frame in enumerate(reader):
            if frame_idx % stride != 0:
                continue
            if max_frames is not None and written >= max_frames:
                break
            rgb = np.asarray(frame[..., :3])
            imageio.imwrite(os.path.join(sparse_dir, f"{written:05d}.png"), rgb)
            written += 1
    finally:
        reader.close()

    if written == 0:
        raise RuntimeError(f"No frames could be read from source input: {src_video_path}")
    return written


def _build_sparse_input_dir(src_video_path, out_dir, stride=1, max_frames=None):
    if not os.path.exists(src_video_path):
        raise FileNotFoundError(f"Source input does not exist: {src_video_path}")

    sparse_dir = os.path.join(out_dir, "_tmp_sv4d_input")
    if os.path.exists(sparse_dir):
        shutil.rmtree(sparse_dir)

    if os.path.isfile(src_video_path):
        frame_count = _build_file_input_dir(
            src_video_path,
            sparse_dir,
            stride=stride,
            max_frames=max_frames,
        )
        return sparse_dir, sparse_dir, frame_count

    frame_files = _list_image_frames(src_video_path)
    if not frame_files:
        raise RuntimeError(f"No image frames found in source input directory: {src_video_path}")

    stride = max(1, int(stride))
    selected = frame_files[::stride]
    if max_frames is not None and max_frames > 0:
        selected = selected[:int(max_frames)]

    if len(selected) == len(frame_files):
        return src_video_path, None, len(selected)

    os.makedirs(sparse_dir, exist_ok=True)

    for i, name in enumerate(selected):
        src_file = os.path.join(src_video_path, name)
        ext = os.path.splitext(name)[1].lower()
        dst_file = os.path.join(sparse_dir, f"{i:05d}{ext}")
        try:
            os.symlink(src_file, dst_file)
        except Exception:
            shutil.copy2(src_file, dst_file)

    return sparse_dir, sparse_dir, len(selected)


# Expected output views per mode (matches azimuths used in sample_sv4d / sample_sp4d)
EXPECTED_VIEWS = {
    "sv4d": [0, 60, 120, 180, 240],
    "sp4d": [0, 60, 120, 180, 240],
    "sv4d2_8views": [0, 30, 75, 120, 165, 210, 255, 300, 330],
}


def _count_color_frames(view_dir):
    color_dir = os.path.join(view_dir, "color")
    if not os.path.isdir(color_dir):
        return 0
    return len([f for f in os.listdir(color_dir) if os.path.splitext(f)[1] in IMAGE_EXTS])


def generation_complete(output_dir, mode):
    """A run is complete only if every expected view has a non-empty color folder
    and all the *generated* views (every view except the input view_0) share the same
    frame count, with view_0 having at least that many. A crashed run (e.g. empty
    view_60, or a view written only partway) is treated as incomplete so it gets
    regenerated instead of silently skipped.

    Note: view_0 may legitimately hold one more frame than the generated views
    (windowed sampling can skip the last source frame), so it is not required to match.
    """
    if not os.path.isdir(output_dir):
        return False
    expected = EXPECTED_VIEWS.get(mode, EXPECTED_VIEWS["sv4d"])
    view0_count = None
    gen_counts = []
    for deg in expected:
        n = _count_color_frames(os.path.join(output_dir, f"view_{deg}"))
        if n == 0:
            return False  # missing or empty view -> incomplete
        if deg == 0:
            view0_count = n
        else:
            gen_counts.append(n)
    if len(set(gen_counts)) != 1:
        return False  # generated views disagree -> a view was written only partway
    return view0_count >= gen_counts[0]


def generate_multi_views(src_video_path, out_dir, mode="sv4d", sv4d_num_steps=50, sv4d_img_size=512, sv4d_max_frames=None, sv4d_stride=1):

    @contextmanager
    def change_dir(destination):
        prev_cwd = os.getcwd()
        os.chdir(destination)
        try:
            yield
        finally:
            os.chdir(prev_cwd)
    model_root = "src/extlibs/generative-models"
    sparse_input_path, sparse_cleanup_dir, sparse_nf = _build_sparse_input_dir(
        src_video_path,
        out_dir,
        stride=sv4d_stride,
        max_frames=sv4d_max_frames,
    )
    if sparse_input_path != src_video_path:
        print(
            f"[SV4D] Using sampled frames: {sparse_nf} "
            f"(stride={sv4d_stride}, max_frames={sv4d_max_frames})"
        )

    frame_files = _list_image_frames(sparse_input_path)
    nf = len(frame_files) if frame_files else len(os.listdir(sparse_input_path))

    try:
        with change_dir(model_root):
            if mode == "sp4d":
                sample_sp4d(
                    input_path=sparse_input_path,
                    output_folder=out_dir,
                    model_path="checkpoints/sp4d.safetensors", 
                    n_frames=nf,
                    num_steps=sv4d_num_steps,
                    img_size=sv4d_img_size,
                )
            elif mode == "sv4d":
                sample_sv4d(
                    input_path=sparse_input_path,
                    output_folder=out_dir,
                    model_path="checkpoints/sv4d2.safetensors", 
                    n_frames=nf,
                    num_steps=sv4d_num_steps,
                    img_size=sv4d_img_size,
                )
            elif mode == "sv4d2_8views":
                sample_sv4d(
                    input_path=sparse_input_path,
                    output_folder=out_dir,
                    model_path="checkpoints/sv4d2_8views.safetensors",
                    n_frames=nf,
                    num_steps=sv4d_num_steps,
                    img_size=sv4d_img_size,
                )
    finally:
        if sparse_cleanup_dir and os.path.isdir(sparse_cleanup_dir):
            shutil.rmtree(sparse_cleanup_dir, ignore_errors=True)


def extract_masks_thinned(out_dir):
    from utils.mask_utils import thin_foreground

    # Save thinned image from masks
    for vd in os.listdir(out_dir):
        if vd.startswith("view"):
            base_dir = os.path.join(out_dir, vd)
        else:
            continue

        # we can use part as mask
        parts_imgs_dir = os.path.join(base_dir, "parts")
        if not os.path.exists(parts_imgs_dir):
            parts_imgs_dir = os.path.join(base_dir, "color")
        mask_imgs_dir = os.path.join(base_dir, "mask")
        
        os.makedirs(mask_imgs_dir, exist_ok=True)
        thinned_imgs_dir = os.path.join(base_dir, "thinned")
        os.makedirs(thinned_imgs_dir, exist_ok=True)
        started = time.perf_counter()
        frame_names = sorted(n for n in os.listdir(parts_imgs_dir) if n.endswith(".png"))
        print(f"[MorphGS] CPU mask/thinning: {vd}, {len(frame_names)} frames", flush=True)

        # make masks: non-white pixels -> 1, white pixels -> 0
        for index, mask_name in enumerate(frame_names, 1):
            if mask_name.endswith(".png"):
                mask_raw = Image.open(os.path.join(parts_imgs_dir, mask_name)).convert('L')
                mask_np = np.array(mask_raw)
                mask_binary = (mask_np < 250).astype(np.uint8) * 255
                Image.fromarray(mask_binary).save(os.path.join(mask_imgs_dir, mask_name), compress_level=1)
                Image.fromarray(thin_foreground(mask_binary)).save(
                    os.path.join(thinned_imgs_dir, mask_name), compress_level=1)
                if index % 12 == 0 or index == len(frame_names):
                    print(f"[MorphGS] CPU mask/thinning {vd}: {index}/{len(frame_names)} "
                          f"({time.perf_counter() - started:.1f}s)", flush=True)
    

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.feature_utils import extract_feature, set_feature_extraction

def extract_src_features(out_dir):

    # load model
    NUM_PATCHES = 60
    with torch.no_grad():
        sd_model, sd_aug, extractor_vit, aggre_net, num_patches = set_feature_extraction(NUM_PATCHES)

    for vd in os.listdir(out_dir):
        if vd.startswith("view"):
            base_dir = os.path.join(out_dir, vd)
        else:
            continue

        color_dir = os.path.join(base_dir, "color")
        feature_dir = os.path.join(base_dir, "feature")
        os.makedirs(feature_dir, exist_ok=True)
        if not os.listdir(feature_dir):
            extract_feature(color_dir, feature_dir, sd_model, sd_aug, aggre_net, extractor_vit, num_patches)



if __name__ == "__main__":
    """
    Preprocess source data for motion transfer.
    Steps:
    3. extract features from rendered images. save to feature/ folder.
    """
    
    parser = argparse.ArgumentParser(description="Preprocess source data for MorphGS.")
    parser.add_argument("src_video_path", help="Path to source frames folder / mp4 / gif.")
    parser.add_argument("--mode", default="sv4d", choices=["sv4d", "sp4d", "sv4d2_8views"])
    parser.add_argument("--sv4d_num_steps", type=int, default=50)
    parser.add_argument("--sv4d_img_size", type=int, default=512)
    parser.add_argument("--sv4d_max_frames", type=int, default=None)
    parser.add_argument("--sv4d_stride", type=int, default=1)
    parser.add_argument("--output_suffix", default="", help="Suffix appended to processed_videos/<video_name>.")
    parser.add_argument("--fastmode", action="store_true", help="Use faster SV4D preset.")
    args = parser.parse_args()

    print("Starting preprocess_src.py")
    src_video_path = args.src_video_path

    sv4d_num_steps = args.sv4d_num_steps
    sv4d_img_size = args.sv4d_img_size
    sv4d_max_frames = args.sv4d_max_frames
    sv4d_stride = max(1, int(args.sv4d_stride))
    if args.fastmode:
        if sv4d_num_steps == 50:
            sv4d_num_steps = 24
        if sv4d_img_size == 512:
            sv4d_img_size = 384

    print(f"Input path: {src_video_path}")

    src_video_path = os.path.abspath(src_video_path)
    output_root = os.path.dirname(os.path.dirname(os.path.dirname(src_video_path)))
    print(f"Output root: {output_root}")
    
    video_name = os.path.basename(os.path.dirname(src_video_path))
    if args.output_suffix:
        video_name = f"{video_name}{args.output_suffix}"
    output_dir = os.path.join(output_root, "processed_videos", video_name)
    
    print(f"Output directory: {output_dir}")

    # 1. load source video and generate multi-views: view0, 60,120,180,240 will be generated: save to processed_videos/view_{view_id}
    #     - save part grouping as well.
    if not generation_complete(output_dir, args.mode):
        stage_started = time.perf_counter()
        print("[MorphGS] SV4D GPU synthesis starting (CPU model loading/frame I/O included)", flush=True)
        # Clear any partial output from a previously crashed run so it regenerates cleanly.
        if os.path.exists(output_dir) and os.listdir(output_dir):
            print(f"Incomplete output detected, regenerating: {output_dir}")
            shutil.rmtree(output_dir, ignore_errors=True)
        processed_path = generate_multi_views(
            src_video_path,
            output_dir,
            mode=args.mode,
            sv4d_num_steps=sv4d_num_steps,
            sv4d_img_size=sv4d_img_size,
            sv4d_max_frames=sv4d_max_frames,
            sv4d_stride=sv4d_stride,
        )
        print(f"[MorphGS] SV4D synthesis finished in {time.perf_counter() - stage_started:.1f}s", flush=True)

    view_dirs = sorted([vd for vd in os.listdir(output_dir) if vd.startswith("view")])
    if len(view_dirs) == 0:
        raise RuntimeError(f"No generated view_* directories found under {output_dir}")
    sentinel_view = view_dirs[1] if len(view_dirs) > 1 else view_dirs[0]

    # 2. for all view folders, extract masks & thinned masks. save to mask/ and thinned/ folders.
    if not os.path.exists(os.path.join(output_dir, sentinel_view, "thinned")):
        extract_masks_thinned(output_dir)

    # 3. extract features from rendered images. save to feature/ folder.
    if not os.path.exists(os.path.join(output_dir, sentinel_view, "feature")):
        stage_started = time.perf_counter()
        print("[MorphGS] DINO GPU feature extraction starting", flush=True)
        extract_src_features(output_dir)
        print(f"[MorphGS] DINO finished in {time.perf_counter() - stage_started:.1f}s", flush=True)

    print("Finished preprocess_src.py")
