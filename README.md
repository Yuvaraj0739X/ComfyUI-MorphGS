# ComfyUI-MorphGS

ComfyUI custom nodes for [MorphGS](https://github.com/xodus777/MorphGS) — video-to-4D
character motion transfer. Drives a full pipeline from a rigged character mesh and a source
video to a trained, animated Gaussian-splat render: character rig conversion, DINOv2 target
feature extraction, SV4D/SP4D source multi-view preprocessing, and gsplat-based training.

## Why this exists

MorphGS needs its own pinned Python environment (a specific torch/CUDA build plus compiled
CUDA extensions) that generally differs from ComfyUI's own environment. Rather than trying to
force everything into one environment, these nodes orchestrate a **separate MorphGS
installation** as a subprocess — you point the nodes at that environment via a few
environment variables, and they handle staging inputs, running each pipeline stage, and
pulling the results back into ComfyUI.

## Nodes

| Node | Does |
|---|---|
| **MorphGS: Preprocess Character** | Accepts a rigged `.fbx` (e.g. Mixamo) or `.glb` (e.g. output from [SkinTokens](https://github.com/VAST-AI-Research/SkinTokens)/TokenRig, or any Blender-importable rigged mesh) and converts it into MorphGS's expected `mesh.obj` + RigNet-format rig, then runs MorphGS's target-side preprocessing (canonical-view rendering + feature extraction). |
| **MorphGS: Preprocess Video** | Segments a raw video onto a white square background if needed, then runs SV4D/SP4D multi-view synthesis + source-side feature extraction. |
| **MorphGS: Train & Render** | Registers the `<scene>_to_<character>` experiment, trains it, and returns the rendered result both as a file path and as an `IMAGE` batch for in-graph preview. |

Each node caches its own outputs and skips re-running a stage that's already done (unless
`force_reprocess`/`force_retrain` is set), so you can safely re-run an upstream node without
redoing an expensive downstream step.

## Requirements

- A working [MorphGS](https://github.com/xodus777/MorphGS) installation, in its own Python
  environment (conda or venv), reachable from wherever ComfyUI runs.
- `ffmpeg` and `ffprobe` on `PATH` in that environment.
- `rembg` installed in that environment (used for background segmentation in the video node).
- `blender` (4.2+) on `PATH` — the same requirement as
  [ComfyUI-SkinTokens](https://github.com/Aero-Ex/ComfyUI-SkinTokens)'s headless Blender
  server, so one Blender install can serve both node packs.
- SV4D/SP4D's `extlibs/generative-models` and the relevant checkpoint set up inside the
  MorphGS environment, if you intend to use the video-preprocessing node. If they aren't set
  up, that node's underlying error surfaces verbatim rather than being hidden.

## Configuration

Set these environment variables before launching ComfyUI (defaults shown):

| Variable | Default | Meaning |
|---|---|---|
| `MORPHGS_BACKEND` | `linux` | `linux` for a native install alongside ComfyUI, or `wsl` for developing/testing this node package on Windows against a MorphGS install running inside WSL |
| `MORPHGS_HOME` | `/workspace/MorphGS` | Path to the MorphGS repo root |
| `MORPHGS_CONDA_ENV` | `morphgs` | Name of the conda/venv environment MorphGS was installed into |
| `MORPHGS_CONDA_BASE` | `/opt/conda` | Path to the conda installation's `activate` script |
| `MORPHGS_BLENDER_BIN` | `blender` | Path to (or bare name of) the Blender executable |
| `MORPHGS_WSL_DISTRO` | `Ubuntu-22.04` | WSL distro name (only used when `MORPHGS_BACKEND=wsl`) |

## Typical workflow

1. **MorphGS: Preprocess Character** — point `character_source_path` at your rigged mesh
   (`.fbx`/`.glb`), give it a `character_name`.
2. **MorphGS: Preprocess Video** — point `video_path` at your source clip, give it a
   `scene_name`, pick an `sv4d_mode`.
3. **MorphGS: Train & Render** — pass the `scene_name` and `character_name` from the two
   nodes above, set `iterations`, run.

## Known good pairing

Pairs well with [ComfyUI-SkinTokens](https://github.com/Aero-Ex/ComfyUI-SkinTokens) for
automatic rigging: rig your raw mesh with SkinTokens first, then feed its `.glb` output
directly into **MorphGS: Preprocess Character** as `character_source_path`.

## License

MIT — see [LICENSE](LICENSE). MorphGS itself is MIT-licensed; its Gaussian-splatting core
runs on [gsplat](https://github.com/nerfstudio-project/gsplat) (Apache-2.0) rather than the
original non-commercial `diff-gaussian-rasterization`.
