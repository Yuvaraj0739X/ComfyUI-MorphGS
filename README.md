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
| **MorphGS: Export Animated Mesh** | Turns a trained experiment into a real, standalone animated 3D asset (`.glb`/`.fbx`) instead of only a rendered video. Replays the trained `AnimationField` checkpoint frame-by-frame to get absolute per-joint transforms, then bakes them onto a skinned mesh in headless Blender: onto the character's *original* rigged file when one is available (which also carries over that file's own materials/textures automatically), or -- for characters with no such file on disk (e.g. MorphGS's own bundled demo characters) -- onto a fresh armature built directly from `mesh.obj` + the RigNet-format rig file's own joint positions and per-vertex skin weights, first re-resolving those weights (`resolve_skinning_weights.py`) exactly as MorphGS's own `Rig` class would for that character's config (some characters' configs apply heat-diffusion smoothing to the raw rig-file weights before training), and reading UVs plus a `.mtl`-referenced texture image if present (or, for characters with no UV/material data at all -- like MorphGS's own bundled `spot` -- per-vertex colors, if `mesh.obj` uses trimesh's "v x y z r g b" extension) so the exported mesh keeps its appearance too. Both `.glb` (self-contained, textures embedded) and `.fbx` (textures embedded via `embed_textures`) carry textures through when the source has them. Shows the result directly in ComfyUI's own native interactive 3D viewer (the same widget its built-in **Save 3D Model** node uses) as soon as it finishes — no separate Preview 3D node needed. |
| **MorphGS: Setup SV4D** | One-time environment setup for the SV4D/SP4D dependency used by Preprocess Video: clones Stability AI's `generative-models` repo, installs its dependencies, and downloads the checkpoint for the mode you pick. Not needed for the DINOv2 features Preprocess Character uses (those download automatically via `torch.hub`), and not needed for SkinTokens (that's handled by ComfyUI-SkinTokens's own node). No login or token is required for either the repo clone or the checkpoint download. |

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

### Checkpoint folder discoverability

On load, this package registers two extra ComfyUI model-folder categories purely for
visibility in ComfyUI's own model folder listings — this doesn't change how any node loads
these files (they're still read from inside the MorphGS environment via the pipeline itself,
which may be a different machine entirely), it just makes them show up:

| Category | Points at |
|---|---|
| `morphgs_sv4d_checkpoints` | `$MORPHGS_HOME/src/extlibs/generative-models/checkpoints` |
| `morphgs_deform_checkpoints` | `$MORPHGS_HOME/output` (every trained experiment's checkpoints) |

Registration is skipped silently if the resolved path isn't reachable from wherever ComfyUI
itself runs (e.g. a fully remote MorphGS deployment with no local mount) — that's a normal
configuration, not an error. On setups where MorphGS lives on a different drive, you can also
point ComfyUI's standard `extra_model_paths.yaml` at these same directories under whatever
category names you like.

## Typical workflow

0. **MorphGS: Setup SV4D** (one-time, only if you'll use Preprocess Video) — pick your
   `sv4d_mode`, run once. This clones ~1GB of code and downloads a ~12GB checkpoint, so
   expect it to take a while the first time; it's cached and skipped on subsequent runs.
1. **MorphGS: Preprocess Character** — point `character_source_path` at your rigged mesh
   (`.fbx`/`.glb`), give it a `character_name`.
2. **MorphGS: Preprocess Video** — point `video_path` at your source clip, give it a
   `scene_name`, pick an `sv4d_mode`.
3. **MorphGS: Train & Render** — pass the `scene_name` and `character_name` from the two
   nodes above, set `iterations`, run.
4. **MorphGS: Export Animated Mesh** *(optional)* — once training is done, pass the same
   `scene_name`/`character_name`/`iterations` to get a real animated `.glb`/`.fbx` you can
   drop into Blender, Unity, Unreal, etc. — not just a rendered video. The result shows up
   directly in ComfyUI's own 3D viewer on the node itself once it finishes running.

A ready-to-load example wiring nodes 1-3 together is in
[`workflows/example_morphgs_pipeline.json`](workflows/example_morphgs_pipeline.json) —
drag it into ComfyUI to see the graph.

## Known good pairing

Pairs well with [ComfyUI-SkinTokens](https://github.com/Aero-Ex/ComfyUI-SkinTokens) for
automatic rigging: rig your raw mesh with SkinTokens first, then feed its `.glb` output
directly into **MorphGS: Preprocess Character** as `character_source_path`.

## License

MIT — see [LICENSE](LICENSE). MorphGS itself is MIT-licensed; its Gaussian-splatting core
runs on [gsplat](https://github.com/nerfstudio-project/gsplat) (Apache-2.0) rather than the
original non-commercial `diff-gaussian-rasterization`.
