# ComfyUI-MorphGS

ComfyUI custom nodes for [MorphGS](https://github.com/xodus777/MorphGS) — video-to-4D
character motion transfer. Drives a full pipeline from a rigged character mesh and a source
video to a trained, animated Gaussian-splat render: character rig conversion, DINOv2 target
feature extraction, SV4D/SP4D source multi-view preprocessing, and gsplat-based training.

## Why this exists

This installs MorphGS directly into **this same ComfyUI environment** — install the node,
run `install.py`, download the checkpoint, and go, the same as any other ComfyUI custom node.

That's a real tradeoff worth understanding before you install it, not a detail to skip:
MorphGS is pinned to `torch==2.0.1+cu118` and needs two compiled CUDA extensions plus
`pytorch3d` built from source against that exact CUDA build. `install.py` only replaces your
existing torch if it isn't already a CUDA 11.8 build — but if it does replace it, that will
likely break any *other* custom node in the same ComfyUI install that wants a different/newer
torch. **This is only appropriate for a ComfyUI instance dedicated to running this pipeline**,
not a general-purpose install with lots of other custom nodes.

## Nodes

| Node | Does |
|---|---|
| **MorphGS: Preprocess Character** | Accepts a rigged `.fbx` (e.g. Mixamo) or `.glb` (e.g. output from [SkinTokens](https://github.com/VAST-AI-Research/SkinTokens)/TokenRig, or any Blender-importable rigged mesh) and converts it into MorphGS's expected `mesh.obj` + RigNet-format rig, then runs MorphGS's target-side preprocessing (canonical-view rendering + feature extraction). |
| **MorphGS: Preprocess Video** | Segments a raw video onto a white square background if needed, then runs SV4D/SP4D multi-view synthesis + source-side feature extraction. `sv4d_mode` is a real dropdown of SV4D/SP4D checkpoints found in the `morphgs_sv4d_checkpoints` folder (registered by this package) — not a fixed list — reflecting whatever `MorphGS: Setup SV4D` has actually downloaded. |
| **MorphGS: Train & Render** | Registers the `<scene>_to_<character>` experiment, trains it, and returns the rendered result both as a file path and as an `IMAGE` batch for in-graph preview. |
| **MorphGS: Export Animated Mesh** | Turns a trained experiment into a real, standalone animated 3D asset (`.glb`/`.fbx`) instead of only a rendered video. Replays the trained `AnimationField` checkpoint frame-by-frame to get absolute per-joint transforms, then bakes them onto a skinned mesh in headless Blender: onto the character's *original* rigged file when one is available (which also carries over that file's own materials/textures automatically), or -- for characters with no such file on disk (e.g. MorphGS's own bundled demo characters) -- onto a fresh armature built directly from `mesh.obj` + the RigNet-format rig file's own joint positions and per-vertex skin weights, first re-resolving those weights (`resolve_skinning_weights.py`) exactly as MorphGS's own `Rig` class would for that character's config (some characters' configs apply heat-diffusion smoothing to the raw rig-file weights before training), and reading UVs plus a `.mtl`-referenced texture image if present (or, for characters with no UV/material data at all -- like MorphGS's own bundled `spot` -- per-vertex colors, if `mesh.obj` uses trimesh's "v x y z r g b" extension) so the exported mesh keeps its appearance too. Both `.glb` (self-contained, textures embedded) and `.fbx` (textures embedded via `embed_textures`) carry textures through when the source has them. Shows the result directly in ComfyUI's own native interactive 3D viewer (the same widget its built-in **Save 3D Model** node uses) as soon as it finishes — no separate Preview 3D node needed. |
| **MorphGS: Setup SV4D** | One-time setup for the SV4D/SP4D dependency used by Preprocess Video: clones Stability AI's `generative-models` repo, installs its dependencies into this same environment, and downloads the checkpoint for the mode you pick straight into MorphGS's own checkpoints folder — SV4D has no native ComfyUI model architecture to run through the built-in Load Checkpoint node directly (unlike SV3D/SVD, which ComfyUI does support natively), so this is as close to "install and run" as it can get for this specific model. Not needed for the DINOv2 features Preprocess Character uses (those download automatically via `torch.hub`), and not needed for SkinTokens (that's handled by ComfyUI-SkinTokens's own node). No login or token is required for either the repo clone or the checkpoint download. |

Each node caches its own outputs and skips re-running a stage that's already done (unless
`force_reprocess`/`force_retrain` is set), so you can safely re-run an upstream node without
redoing an expensive downstream step. Every node is also an `OUTPUT_NODE`, so any one of them
can be queued and will actually execute on its own while you're building out a graph step by
step -- without this, ComfyUI's execution engine prunes out a node with nothing downstream
consuming its result, and queuing it alone silently does nothing.

## Installation

1. Install this node the normal way — via ComfyUI Manager, or `git clone
   https://github.com/Yuvaraj0739X/ComfyUI-MorphGS` into `custom_nodes/`.
2. Run `python install.py` from this package's directory (Manager runs this for you
   automatically). This is the real setup step, and it's substantial — not a quick pip
   install:
   - Checks for the CUDA 11.8 toolkit (`nvcc`) and fails with a clear message if it's missing
     (needed to compile `pytorch3d` and MorphGS's own CUDA extensions from source).
   - Installs `torch==2.0.1+cu118`/`torchvision==0.15.2+cu118` **only if your existing torch
     isn't already a CUDA 11.8 build** — otherwise it builds MorphGS's extensions against
     what you already have, no downgrade needed.
   - Builds `pytorch3d` from source, installs `gsplat` and MorphGS's other pip dependencies
     (with `numpy<2` enforced regardless of what MorphGS's own `requirements.txt` pins — numpy
     2.x is a known, already-encountered break for `torch.from_numpy`/`pytorch3d` on this
     dependency stack).
   - Compiles the two custom CUDA extensions from MorphGS's own source, which ships bundled
     right in this repo's `morphgs_src/` directory (a customized copy with fixes: DINOv2-only
     feature matching, `gsplat`-based rendering, topology-aware ARAP regularization) — no
     separate clone or repo to keep in sync.
   - Verifies torch/gsplat/pytorch3d all import correctly before finishing.
3. `blender` (4.2+) needs to be on `PATH` separately — it's **not** a MorphGS dependency (only
   this package's own mesh/rig conversion and export scripts use it), so `install.py` doesn't
   install it. Same requirement as
   [ComfyUI-SkinTokens](https://github.com/Aero-Ex/ComfyUI-SkinTokens)'s headless Blender
   server, so one install serves both node packs. Point `MORPHGS_BLENDER_BIN` at it if it's
   not on `PATH`.
4. Run **MorphGS: Setup SV4D** once if you'll use the video-preprocessing node (see below).

### Configuration

Only two environment variables, both optional:

| Variable | Default | Meaning |
|---|---|---|
| `MORPHGS_HOME` | `<this package>/morphgs_src` | Path to the bundled MorphGS source — override only for an advanced/manual setup pointing at a checkout elsewhere |
| `MORPHGS_BLENDER_BIN` | `blender` | Path to (or bare name of) the Blender executable |

### Checkpoint folder discoverability

On load, this package registers two ComfyUI model-folder categories pointing at MorphGS's own
checkpoint locations, so they show up in ComfyUI's own model folder listings the same way any
other checkpoint does, and so **MorphGS: Preprocess Video**'s `sv4d_mode` dropdown reflects
what's actually there:

| Category | Points at |
|---|---|
| `morphgs_sv4d_checkpoints` | `$MORPHGS_HOME/src/extlibs/generative-models/checkpoints` |
| `morphgs_deform_checkpoints` | `$MORPHGS_HOME/output` (every trained experiment's checkpoints) |

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
