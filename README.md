# ComfyUI-MorphGS

ComfyUI custom nodes for [MorphGS](https://github.com/xodus777/MorphGS) — video-to-4D
character motion transfer. Drives a full pipeline from a rigged character mesh and a source
video to a trained, animated Gaussian-splat render and an exportable animated `.glb`/`.fbx`:
character rig conversion, DINOv2 target feature extraction, SV4D/SP4D source multi-view
preprocessing, and gsplat-based training.

## Installation

Install it like any other custom node — via ComfyUI Manager, or

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Yuvaraj0739X/ComfyUI-MorphGS
cd ComfyUI-MorphGS
pip install -r requirements.txt
python install.py
```

(Manager runs those last two steps for you.) There is no separate environment, no conda, and
no compiler: everything installs into the ComfyUI environment you already have, and **torch is
never touched** — whatever torch build ComfyUI runs on is what MorphGS runs on.

What `install.py` does beyond `requirements.txt`:

- **Installs prebuilt `pytorch3d` and `gsplat` wheels** matched to your Python / torch / CUDA
  combination, from the [cuda-wheels](https://github.com/PozzettiAndrea/cuda-wheels) index
  (the same one the `comfy-env` tooling behind ComfyUI-TRELLIS2 and ComfyUI-3D-Pack uses).
  Coverage: Linux and Windows, Python 3.10–3.14, torch 2.4–2.13, CUDA 12.4–13.2, with kernels
  for every NVIDIA generation from Turing (RTX 20) through Blackwell (RTX 50). Only if your
  torch/CUDA pair has no wheel there does it fall back to building from source, which then
  needs `nvcc`; it says so loudly if that happens.
- **Clones Stability AI's `generative-models`** (the SV4D/SP4D code behind Preprocess Video)
  into the bundled MorphGS tree, and installs the handful of packages SV4D inference imports
  (unpinned — it does not apply Stability's frozen `pt2.txt`, which would downgrade
  `transformers`, `opencv-python` and torch across your whole ComfyUI).
- **Verifies** that torch, pytorch3d and gsplat's compiled kernels all import before finishing.

Two things it can't do for you:

1. **Blender 4.2+ on `PATH`** (or set `MORPHGS_BLENDER_BIN`). Only this package's own rig
   conversion and export scripts use it, always in `--background` mode, so on a headless GPU
   box `apt-get install blender` is enough. Needed by **Preprocess Character** and **Export
   Animated Mesh**.
2. **The SV4D/SP4D checkpoint**, if you'll use **Preprocess Video**. Download it from Hugging
   Face into ComfyUI's standard `models/diffusion_models` folder:
   - `sv4d` / `sv4d2_8views` modes → [stabilityai/sv4d2.0](https://huggingface.co/stabilityai/sv4d2.0)
   - `sp4d` mode → [stabilityai/sp4d](https://huggingface.co/stabilityai/sp4d)

   The example workflow declares `sv4d2.safetensors` in its model metadata, so loading it in
   ComfyUI brings up the standard missing-models dialog with the download link. No node
   downloads checkpoints on its own. `models/checkpoints` also works if you'd rather keep it
   there.

MorphGS's own source ships inside this repo at `morphgs_src/` — a customized copy with
DINOv2-only feature matching, gsplat rendering (Apache-2.0; the original non-commercial Inria
rasterizer is not used or bundled) and topology-aware ARAP regularization. Nothing else to
clone or keep in sync.

### Hardware and torch compatibility

The nodes run wherever ComfyUI runs on an NVIDIA GPU with a CUDA build of torch that the
prebuilt wheel index covers. `install.py` checks this and says exactly what to do when it
doesn't hold.

| Your ComfyUI torch build | RTX 50 (Blackwell, sm_120) | RTX 40 (Ada, sm_89) | RTX 30 (Ampere) | RTX 20 (Turing) |
|---|---|---|---|---|
| CUDA 12.8 / 12.9 / 13.0 / 13.2 | yes | yes | yes | gsplat yes; pytorch3d needs CUDA 12.8 (the 13.x wheel ships sm_80+ only) |
| CUDA 12.4 / 12.6 | no (torch itself has no Blackwell kernels on these builds) | yes | yes | yes |
| CUDA 11.8 / 12.1, or torch older than 2.4 | not covered by the prebuilt index: update torch (see below) | | | |
| CPU-only, ROCm, Apple Silicon | not supported: training and rendering run CUDA kernels | | | |

Kernel coverage above was read directly from the published wheels' fatbins, not from the
index's description. An RTX 40 card runs the sm_86/sm_80 kernels in the CUDA 13 wheels (same
major architecture); an RTX 50 card needs a CUDA 12.8+ torch, which is also what ComfyUI itself
requires on that hardware.

If your torch is outside the covered set, `install.py` prints the exact `pip install` line to
move ComfyUI onto a current CUDA 12.8 torch build, and only then attempts a source build (which
needs `nvcc` and a C++ compiler). Re-running the install (Manager → *Try fix* on this node)
after any torch update re-resolves the matching wheels automatically.

Platform notes:

- **Windows** (including the portable build and ComfyUI Desktop): wheels exist for every
  covered combination, and nothing here needs `git` — Stability's SV4D code is fetched as an
  archive when `git` is absent. Manager runs `install.py` at the next ComfyUI start on Windows
  (its normal deferred-install behaviour), so expect one restart. Blender is auto-detected in
  `C:\Program Files\Blender Foundation\Blender x.y` if it's not on `PATH`.
- **Linux** (bare, Docker, RunPod/Vast-style images): `apt-get install blender` is enough for
  the Blender side; nothing needs a CUDA toolkit on the box.

### Configuration

Only two environment variables, both optional:

| Variable | Default | Meaning |
|---|---|---|
| `MORPHGS_HOME` | `<this package>/morphgs_src` | Path to the bundled MorphGS source — override only for an advanced/manual setup pointing at a checkout elsewhere |
| `MORPHGS_BLENDER_BIN` | `blender` | Path to (or bare name of) the Blender executable |

## Nodes

| Node | Does |
|---|---|
| **MorphGS: Preprocess Character** | `character_source_path` is a dropdown listing rigged `.fbx`/`.glb`/`.gltf` files **and** already-prepared character folders found under ComfyUI's own `input/` directory — drop your file there (the normal ComfyUI upload location) and pick it here, no manual path-typing. Accepts a rigged `.fbx` (e.g. Mixamo) or `.glb` (e.g. output from [SkinTokens](https://github.com/VAST-AI-Research/SkinTokens)/TokenRig, or any Blender-importable rigged mesh) and converts it into MorphGS's expected `mesh.obj` + RigNet-format rig, then runs MorphGS's target-side preprocessing (canonical-view rendering + feature extraction). |
| **MorphGS: Preprocess Video** | `video_path` is likewise a refreshable dropdown of video files (`.mp4`/`.mov`/`.avi`/`.mkv`/`.webm`) found under `input/`. Segments the clip onto a white square background if needed, then runs SV4D/SP4D multi-view synthesis + source-side feature extraction. `sv4d_mode` scans ComfyUI's standard `models/diffusion_models` folder first, with `models/sv4d`, `models/checkpoints`, and MorphGS's own `generative-models/checkpoints` as fallbacks. |
| **MorphGS: Train & Render** | Registers the `<scene>_to_<character>` experiment, trains it, and returns the rendered result both as a file path and as an `IMAGE` batch for in-graph preview. `seed` has the standard ComfyUI seed widget (fixed/increment/decrement/randomize) and controls MorphGS's own training-time randomness. |
| **MorphGS: Export Animated Mesh** | Turns a trained experiment into a real, standalone animated 3D asset (`.glb`/`.fbx`) instead of only a rendered video. Replays the trained `AnimationField` checkpoint frame-by-frame to get absolute per-joint transforms, then bakes them onto a skinned mesh in headless Blender: onto the character's *original* rigged file when one is available (which also carries over that file's own materials/textures automatically), or -- for characters with no such file on disk (e.g. MorphGS's own bundled demo characters) -- onto a fresh armature built directly from `mesh.obj` + the RigNet-format rig file's own joint positions and per-vertex skin weights, first re-resolving those weights (`resolve_skinning_weights.py`) exactly as MorphGS's own `Rig` class would for that character's config (some characters' configs apply heat-diffusion smoothing to the raw rig-file weights before training), and reading UVs plus a `.mtl`-referenced texture image if present (or, for characters with no UV/material data at all -- like MorphGS's own bundled `spot` -- per-vertex colors, if `mesh.obj` uses trimesh's "v x y z r g b" extension) so the exported mesh keeps its appearance too. Both `.glb` (self-contained, textures embedded) and `.fbx` (textures embedded via `embed_textures`) carry textures through when the source has them. Shows the result directly on the node itself as soon as it finishes (no separate node needed for that), **and** also outputs `preview_path` -- the same output-dir-relative string ComfyUI-Hunyuan3DWrapper's own `Hy3DExportMesh` returns -- so you can additionally wire it into ComfyUI's native **Preview 3D & Animation** (`Preview3D`) node, exactly like Hunyuan3DWrapper's own example workflow does, if you want that as a separate, movable node in the graph. |

DINOv2 features for Preprocess Character download automatically via `torch.hub` (the same
"auto-download a secondary encoder, no dedicated folder" pattern ComfyUI-Hunyuan3DWrapper and
ComfyUI-HY-Motion1 use for their own helper models). SV4D has no native ComfyUI model
architecture (unlike SV3D/SVD), so it can't go through the built-in Load Checkpoint node;
Preprocess Video loads it from `models/diffusion_models` itself.

Each preprocessing node caches its outputs **on disk** with a source-and-settings manifest and
skips re-running only when that manifest still matches
(unless `force_reprocess`/`force_retrain`/`force_reexport` is set) -- deliberately not relying
on ComfyUI's own in-memory result cache, since that doesn't survive a ComfyUI restart and
training here can take hours. This is what actually lets you restart ComfyUI mid-pipeline
without losing finished work. The manifests propagate through Train & Render and Export:
changing a source file, preprocessing setting, checkpoint, training seed, or trained deform
checkpoint invalidates the affected downstream cache automatically. The force switches remain
available for an unconditional rerun.

Every node is also an `OUTPUT_NODE`, so any one of them can be queued and will actually execute
on its own while you're building out a graph step by step -- without this, ComfyUI's execution
engine prunes out a node with nothing downstream consuming its result, and queuing it alone
silently does nothing.

### Checkpoint folder discoverability

On load, this package creates a fallback `models/sv4d` folder and registers two ComfyUI
model-folder categories, so **MorphGS: Preprocess Video**'s `sv4d_mode` dropdown reflects
whatever checkpoint file you've actually placed:

| Category | Points at |
|---|---|
| `morphgs_sv4d_checkpoints` | ComfyUI's `models/diffusion_models` folder first, then `models/sv4d` (created automatically), `models/checkpoints`, and `$MORPHGS_HOME/src/extlibs/generative-models/checkpoints` |
| `morphgs_deform_checkpoints` | `$MORPHGS_HOME/output` (every trained experiment's checkpoints) |

## Typical workflow

0. Download an SV4D/SP4D checkpoint from Hugging Face into `models/diffusion_models` (one-time, only if
   you'll use Preprocess Video — see Installation above).
1. **MorphGS: Preprocess Character** — point `character_source_path` at your rigged mesh
   (`.fbx`/`.glb`), give it a `character_name`.
2. **MorphGS: Preprocess Video** — point `video_path` at your source clip, give it a
   `scene_name`, pick an `sv4d_mode`.
3. **MorphGS: Train & Render** — pass the `scene_name` and `character_name` from the two
   nodes above, set `iterations`, run.
4. **MorphGS: Export Animated Mesh** *(optional)* — once training is done, pass the same
   `scene_name`/`character_name`/`iterations` to get a real animated `.glb`/`.fbx` you can
   drop into Blender, Unity, Unreal, etc. — not just a rendered video. The result shows up
   directly on the node itself once it finishes running; wire its `preview_path` output into
   ComfyUI's native **Preview 3D & Animation** node too if you'd rather have that as its own
   node in the graph.

A ready-to-load example wiring all four nodes together (plus a **Preview 3D & Animation** node
after Export) is in
[`workflows/example_morphgs_pipeline.json`](workflows/example_morphgs_pipeline.json) —
drag it into ComfyUI to see the graph.

## Known good pairing

Pairs well with [ComfyUI-SkinTokens](https://github.com/Aero-Ex/ComfyUI-SkinTokens) for
automatic rigging: rig your raw mesh with SkinTokens first, then feed its `.glb` output
directly into **MorphGS: Preprocess Character** as `character_source_path`. Both packs share
the same Blender-on-PATH requirement, so one install serves both.

## License

MIT — see [LICENSE](LICENSE). MorphGS itself is MIT-licensed; its Gaussian-splatting core
runs on [gsplat](https://github.com/nerfstudio-project/gsplat) (Apache-2.0) rather than the
original non-commercial `diff-gaussian-rasterization`.
