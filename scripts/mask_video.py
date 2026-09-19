"""
Segment an object-centric video and composite it onto a white square background,
matching MorphGS's required demo/videos/<scene>/rgb.mp4 convention
(subject centered, consistent scale across the clip, pure white background).

Usage: python mask_video.py <input_video> <output_rgb_mp4> [--size 1080] [--ratio 0.9] [--skip-mask]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image


def extract_frames(video_path, raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, os.path.join(raw_dir, "%05d.png")],
        check=True, capture_output=True,
    )
    return sorted(os.listdir(raw_dir))


def segment_and_composite(raw_dir, frame_files, out_dir, out_size, image_frame_ratio, alpha_thresh=30):
    import onnxruntime as ort
    from rembg import remove, new_session

    os.makedirs(out_dir, exist_ok=True)
    available = ort.get_available_providers()
    requested = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"])
    session = new_session("u2net", providers=requested)
    active = session.inner_session.get_providers()
    print(f"rembg/ONNX providers: available={available}; active={active}")
    if "CUDAExecutionProvider" in available and "CUDAExecutionProvider" not in active:
        print(
            "WARNING: ONNX Runtime advertised CUDA but rembg fell back to CPU. "
            "Check the CUDA/cuDNN runtime required by onnxruntime-gpu.",
            file=sys.stderr,
        )

    rgba_cache = {}
    bboxes = []
    for fname in frame_files:
        img = Image.open(os.path.join(raw_dir, fname)).convert("RGB")
        rgba = remove(img, session=session)
        rgba_cache[fname] = rgba
        alpha = np.array(rgba)[:, :, 3]
        ys, xs = np.where(alpha > alpha_thresh)
        bboxes.append((xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None)

    valid = [b for b in bboxes if b is not None]
    if not valid:
        raise RuntimeError("No subject detected in any frame during segmentation.")

    x0 = min(b[0] for b in valid)
    y0 = min(b[1] for b in valid)
    x1 = max(b[2] for b in valid)
    y1 = max(b[3] for b in valid)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    subj_w, subj_h = x1 - x0, y1 - y0

    square_side = max(subj_w, subj_h) / image_frame_ratio
    half = square_side / 2
    crop_x0, crop_y0 = cx - half, cy - half
    crop_x1, crop_y1 = cx + half, cy + half

    white_bg = Image.new("RGB", (out_size, out_size), (255, 255, 255))

    for fname in frame_files:
        rgba = rgba_cache[fname]
        W, H = rgba.size
        pad_l = max(0, int(np.ceil(-crop_x0)))
        pad_t = max(0, int(np.ceil(-crop_y0)))
        pad_r = max(0, int(np.ceil(crop_x1 - W)))
        pad_b = max(0, int(np.ceil(crop_y1 - H)))
        if pad_l or pad_t or pad_r or pad_b:
            padded = Image.new("RGBA", (W + pad_l + pad_r, H + pad_t + pad_b), (0, 0, 0, 0))
            padded.paste(rgba, (pad_l, pad_t))
            rgba = padded

        cx0, cy0 = crop_x0 + pad_l, crop_y0 + pad_t
        cx1, cy1 = crop_x1 + pad_l, crop_y1 + pad_t
        cropped = rgba.crop((int(round(cx0)), int(round(cy0)), int(round(cx1)), int(round(cy1))))
        cropped = cropped.resize((out_size, out_size), Image.LANCZOS)

        canvas = white_bg.copy()
        canvas.paste(cropped, (0, 0), mask=cropped.split()[3])
        canvas.save(os.path.join(out_dir, fname))


def encode_video(frames_dir, out_path, fps):
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(frames_dir, "%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out_path],
        check=True, capture_output=True,
    )


def get_fps(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        check=True, capture_output=True, text=True,
    )
    num, den = result.stdout.strip().split("/")
    return float(num) / float(den)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video")
    parser.add_argument("output_rgb_mp4")
    parser.add_argument("--size", type=int, default=1080)
    parser.add_argument("--ratio", type=float, default=0.9)
    parser.add_argument("--skip-mask", action="store_true",
                         help="Video is already background-masked/square; just re-encode/copy into place.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_rgb_mp4), exist_ok=True)
    fps = get_fps(args.input_video)

    if args.skip_mask:
        shutil.copy(args.input_video, args.output_rgb_mp4)
        print(f"DONE (copied without re-masking): {args.output_rgb_mp4}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = os.path.join(tmp, "raw")
        comp_dir = os.path.join(tmp, "composited")
        frame_files = extract_frames(args.input_video, raw_dir)
        print(f"Extracted {len(frame_files)} frames")
        segment_and_composite(raw_dir, frame_files, comp_dir, args.size, args.ratio)
        print("Segmentation + compositing done")
        encode_video(comp_dir, args.output_rgb_mp4, fps)

    print(f"DONE: {args.output_rgb_mp4}")


if __name__ == "__main__":
    main()
