# ComfyUI-MorphGS

ComfyUI custom nodes for [MorphGS](https://github.com/xodus777/MorphGS) — video-to-4D
character motion transfer. Drives a full pipeline from a rigged character mesh and a source
video to a trained, animated Gaussian-splat render: character rig conversion, DINOv2 target
feature extraction, SV4D/SP4D source multi-view preprocessing, and gsplat-based training.

## Why this exists

This installs MorphGS directly into **this same ComfyUI environment** — install the node
from the Manager (which runs `install.py` for you automatically, same as any other custom
node with install-time dependencies), download the SV4D/SP4D checkpoint from Hugging Face and
drop it in your `models/checkpoints` folder like any other checkpoint, and go.

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
| **MorphGS: Preprocess Video** | Segments a raw video onto a white square background if needed, then runs SV4D/SP4D multi-view synthesis + source-side feature extraction. `sv4d_mode` is a real dropdown of SV4D/SP4D checkpoints found in your ComfyUI `models/checkpoints` folder (and MorphGS's own `generative-models/checkpoints`, registered by this package) — not a fixed list — reflecting whatever checkpoint file you've actually downloaded and placed there. |
| **MorphGS: Train & Render** | Registers the `<scene>_to_<character>` experiment, trains it, and returns the rendered result both as a file path and as an `IMAGE` batch for in-graph preview. |
| **MorphGS: Export Animated Mesh** | Turns a trained experiment into a real, standalone animated 3D asset (`.glb`/`.fbx`) instead of only a rendered video. Replays the trained `AnimationField` checkpoint frame-by-frame to get absolute per-joint transforms, then bakes them onto a skinned mesh in headless Blender: onto the character's *original* rigged file when one is available (which also carries over that file's own materials/textures automatically), or -- for characters with no such file on disk (e.g. MorphGS's own bundled demo characters) -- onto a fresh armature built directly from `mesh.obj` + the RigNet-format rig file's own joint positions and per-vertex skin weights, first re-resolving those weights (`resolve_skinning_weights.py`) exactly as MorphGS's own `Rig` class would for that character's config (some characters' configs apply heat-diffusion smoothing to the raw rig-file weights before training), and reading UVs plus a `.mtl`-referenced texture image if present (or, for characters with no UV/material data at all -- like MorphGS's own bundled `spot` -- per-vertex colors, if `mesh.obj` uses trimesh's "v x y z r g b" extension) so the exported mesh keeps its appearance too. Both `.glb` (self-contained, textures embedded) and `.fbx` (textures embedded via `embed_textures`) carry textures through when the source has them. Shows the result directly in ComfyUI's own native interactive 3D viewer (the same widget its built-in **Save 3D Model** node uses) as soon as it finishes — no separate Preview 3D node needed. |

There is deliberately no "Setup SV4D" node. `install.py` (run automatically by the Manager)
already clones and installs Stability AI's `generative-models` (the SV4D/SP4D code) into this
environment; the only thing left to you is downloading the checkpoint file itself from Hugging
Face and placing it in `models/checkpoints`, same as any other checkpoint you use in ComfyUI —
SV4D has no native ComfyUI model architecture to run through the built-in Load Checkpoint node
directly (unlike SV3D/SVD, which ComfyUI does support natively), so there's no such node to
offer here either way. Not needed at all for the DINOv2 features Preprocess Character uses
(those download automatically via `torch.hub`), or for SkinTokens (handled by
ComfyUI-SkinTokens's own node).

Each node caches its own outputs and skips re-running a stage that's already done (unless
`force_reprocess`/`force_retrain` is set), so you can safely re-run an upstream node without
redoing an expensive downstream step. Every node is also an `OUTPUT_NODE`, so any one of them
can be queued and will actually execute on its own while you're building out a graph step by
step -- without this, ComfyUI's execution engine prunes out a node with nothing downstream
consuming its result, and queuing it alone silently does nothing.

## Installation

1. Install this node the normal way — via ComfyUI Manager, or `git clone
   https://github.com/Yuvaraj0739X/ComfyUI-MorphGS` into `custom_nodes/`. The Manager runs
   `install.py` for you automatically right after (same as it does `requirements.txt` for any
   other custom node with install-time dependencies) — there's no separate command to run
   yourself. It's a substantial step, not a quick pip install:
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
   - Clones and installs Stability AI's `generative-models` (the SV4D/SP4D code) so
     **MorphGS: Preprocess Video** is ready with no separate setup node of its own.
   - Verifies torch/gsplat/pytorch3d all import correctly before finishing.
2. `blender` (4.2+) needs to be on `PATH` separately — it's **not** a MorphGS dependency (only
   this package's own mesh/rig conversion and export scripts use it), so `install.py` doesn't
   install it. Same requirement as
   [ComfyUI-SkinTokens](https://github.com/Aero-Ex/ComfyUI-SkinTokens)'s headless Blender
   server, so one install serves both node packs. Point `MORPHGS_BLENDER_BIN` at it if it's
   not on `PATH`.
3. If you'll use **MorphGS: Preprocess Video**, download an SV4D/SP4D checkpoint from Hugging
   Face and place it in your ComfyUI `models/checkpoints` folder, exactly the same way as any
   other checkpoint (no separate node downloads this for you):
   - `sv4d` / `sv4d2_8views` mode → [stabilityai/sv4d2.0](https://huggingface.co/stabilityai/sv4d2.0)
   - `sp4d` mode → [stabilityai/sp4d](https://huggingface.co/stabilityai/sp4d)

### Configuration

Only two environment variables, both optional:

| Variable | Default | Meaning |
|---|---|---|
| `MORPHGS_HOME` | `<this package>/morphgs_src` | Path to the bundled MorphGS source — override only for an advanced/manual setup pointing at a checkout elsewhere |
| `MORPHGS_BLENDER_BIN` | `blender` | Path to (or bare name of) the Blender executable |

### Checkpoint folder discoverability

On load, this package registers two ComfyUI model-folder categories, so **MorphGS: Preprocess
Video**'s `sv4d_mode` dropdown reflects whatever checkpoint file you've actually placed:

| Category | Points at |
|---|---|
| `morphgs_sv4d_checkpoints` | Your ComfyUI `models/checkpoints` folder(s) **and** `$MORPHGS_HOME/src/extlibs/generative-models/checkpoints` |
| `morphgs_deform_checkpoints` | `$MORPHGS_HOME/output` (every trained experiment's checkpoints) |

## Typical workflow

0. Download an SV4D/SP4D checkpoint from Hugging Face into `models/checkpoints` (one-time,
   only if you'll use Preprocess Video — see Installation above).
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
