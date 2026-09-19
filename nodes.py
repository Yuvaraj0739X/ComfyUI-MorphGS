import glob
import json
import os
import re
import shutil
import subprocess
import sys

# folder_paths/torch/numpy/cv2/pytorch3d/gsplat are deliberately NOT imported at module scope:
# folder_paths only exists inside a running ComfyUI process, and the rest are installed by
# install.py into that same environment, not present in the Comfy Registry's isolated node
# scanner (which inspects this file without a full ComfyUI installation, or any of this
# package's own dependencies, alongside it). Importing any of them here would break that
# scanner with ModuleNotFoundError, even though they all work fine once ComfyUI itself loads
# this node. They're imported lazily inside the specific methods that need them. Plain stdlib
# modules (os, shutil, subprocess, sys, glob, json, urllib) are always safe at module scope.

from . import config
from .process_utils import node_script_path, run_blender_script, run_python, subprocess_env

CATEGORY = "MorphGS"
_CACHE_MANIFEST = ".morphgs_cache.json"
_AUTO_HEIGHT_FALLBACK_M = 1.6
_AUTO_HEIGHT_MODE = "auto_from_file_units"


def _safe_stage_name(value, label):
    """A user-facing scene/character name that cannot escape its pipeline directory."""
    value = str(value).strip()
    if not value or value in (".", "..") or os.path.basename(value) != value:
        raise ValueError(f"{label} must be a non-empty name without path separators: {value!r}")
    return value


def _stage_name_from_input(selection, label):
    """Stable pipeline name derived from the uploaded/selected input's filename.

    Users should not need to keep a second, manually typed name in sync with the file picker.
    Keep names readable while making them safe as a directory and MorphGS experiment component.
    """
    selection = str(selection).replace("\\", "/").rstrip("/")
    basename = os.path.basename(selection)
    stem = os.path.splitext(basename)[0]
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    if not value:
        raise ValueError(f"Could not derive a safe {label} from input: {selection!r}")
    return _safe_stage_name(value, label)


def _path_signature(path):
    """Cheap, stable fingerprint for a selected file or prepared-character directory."""
    if os.path.isfile(path):
        stat = os.stat(path)
        return {"type": "file", "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if os.path.isdir(path):
        entries = []
        for root, dirs, files in os.walk(path):
            dirs.sort()
            for name in sorted(files):
                file_path = os.path.join(root, name)
                stat = os.stat(file_path)
                entries.append({
                    "path": os.path.relpath(file_path, path).replace(os.sep, "/"),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                })
        return {"type": "directory", "files": entries}
    raise FileNotFoundError(f"Selected input no longer exists: {path}")


def _read_manifest(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _write_manifest(path, data):
    """Write only after a stage succeeds, so an interrupted run is never treated as cached."""
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(temp_path, path)


def _reset_stage_dir(path, expected_parent):
    """Remove one generated stage directory after proving it is under the expected root."""
    path = os.path.realpath(path)
    expected_parent = os.path.realpath(expected_parent)
    if os.path.commonpath([path, expected_parent]) != expected_parent or path == expected_parent:
        raise RuntimeError(f"Refusing to clear unsafe cache path: {path}")
    if os.path.isdir(path):
        shutil.rmtree(path)


def _input_change_token(selection, force_reprocess):
    if force_reprocess:
        return float("NaN")
    try:
        path = _resolve_input_path(selection)
        return json.dumps(_path_signature(path), sort_keys=True, separators=(",", ":"))
    except (OSError, ValueError):
        return float("NaN")


def _stage_signature(path):
    """Manifest contents when available, otherwise a filesystem signature for legacy caches."""
    manifest = _read_manifest(os.path.join(path, _CACHE_MANIFEST))
    if manifest is not None:
        return {"manifest": manifest}
    try:
        return {"legacy_signature": _path_signature(path)}
    except OSError:
        return None


def _list_input_files(extensions):
    """Files under ComfyUI's own input/ directory (recursively, "subfolder/name.ext" style,
    the same convention LoadImage's own dropdown uses) matching one of the given extensions.
    Falls back to an empty list if folder_paths isn't importable (the Comfy Registry's
    isolated node scanner, which has no such module) -- INPUT_TYPES must still return
    something usable in that case, an empty COMBO list is valid."""
    try:
        import folder_paths

        input_dir = folder_paths.get_input_directory()
    except Exception:
        return []
    results = []
    for root, _dirs, files in os.walk(input_dir):
        rel_root = os.path.relpath(root, input_dir)
        for name in files:
            if os.path.splitext(name)[1].lower() in extensions:
                rel_path = name if rel_root == "." else os.path.join(rel_root, name)
                results.append(rel_path.replace(os.sep, "/"))
    return sorted(results)


def _list_input_prepared_folders():
    """Subfolders under ComfyUI's input/ directory that already look like a prepared MorphGS
    character (contain mesh.obj directly) -- so a previously-converted character can be
    re-selected without re-running Blender. Same folder_paths caveat as _list_input_files."""
    try:
        import folder_paths

        input_dir = folder_paths.get_input_directory()
    except Exception:
        return []
    results = []
    for root, _dirs, files in os.walk(input_dir):
        if "mesh.obj" in files and root != input_dir:
            rel_path = os.path.relpath(root, input_dir).replace(os.sep, "/")
            results.append(rel_path)
    return sorted(results)


def _character_input_options():
    return _list_input_files({".fbx", ".glb", ".gltf"}) + _list_input_prepared_folders()


def _video_input_options():
    return _list_input_files({".mp4", ".mov", ".avi", ".mkv", ".webm"})


def _resolve_input_path(selection):
    """Resolve a value picked from _list_input_files/_list_input_prepared_folders's dropdown
    back into a real filesystem path under ComfyUI's input/ directory."""
    import folder_paths

    input_dir = os.path.realpath(folder_paths.get_input_directory())
    resolved = os.path.realpath(os.path.join(input_dir, *selection.replace("\\", "/").split("/")))
    if os.path.commonpath([input_dir, resolved]) != input_dir:
        raise ValueError(f"Input must be inside ComfyUI's input directory: {selection!r}")
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Selected input no longer exists: {selection!r}")
    return resolved


class MorphGSPreprocessCharacter:
    """
    Prepares a rigged character for MorphGS. character_source_path is a dropdown of files
    (.fbx/.glb/.gltf) and already-prepared folders (containing mesh.obj) found under ComfyUI's
    own input/ directory -- drop your rigged character in there (ComfyUI's normal upload
    location) and it shows up here, no manual path-typing needed. Handles either:
      - a rigged .fbx (e.g. Mixamo) or .glb (e.g. SkinTokens/TokenRig output) -- anything
        Blender can import with an armature + skinned mesh -- auto-converted to mesh.obj
        + a RigNet-format rig, or
      - an already-prepared folder containing mesh.obj + rigging/mesh_ori_rig.txt
    then runs MorphGS's own preprocess_tgt.py (360-view render + feature extraction).
    """

    @classmethod
    def INPUT_TYPES(cls):
        options = _character_input_options()
        if not options:
            options = [""]
        return {
            "required": {
                "character_source_path": (options, {
                    "tooltip": "Choose a rigged character already under ComfyUI/input, or use Upload character.",
                }),
                "force_reprocess": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Normally leave off: completed outputs are reused when the source and settings "
                        "match. Turn on only to discard that on-disk cache and rebuild the character."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("character_name", "log", "detected_height_m")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True  # Lets this run (and its result be visible) standalone before it's
    # wired to anything downstream -- otherwise ComfyUI's execution graph would prune it out
    # entirely and queuing it alone would do nothing (confirmed via a real API test on an
    # OUTPUT_NODE-less node: "Prompt has no outputs").

    @classmethod
    def IS_CHANGED(cls, character_source_path, force_reprocess):
        if force_reprocess:
            return float("NaN")
        return json.dumps({
            "source": _input_change_token(character_source_path, False),
            "height_policy": _AUTO_HEIGHT_MODE,
            "fallback_height_m": _AUTO_HEIGHT_FALLBACK_M,
        }, sort_keys=True, separators=(",", ":"))

    def run(self, character_source_path, force_reprocess):
        log = []
        source_selection = character_source_path
        character_source_path = _resolve_input_path(source_selection)
        character_name = _stage_name_from_input(source_selection, "character_name")
        characters_root = os.path.join(config.MORPHGS_HOME, "demo", "characters")
        char_dir = os.path.join(characters_root, character_name)
        manifest_path = os.path.join(char_dir, _CACHE_MANIFEST)
        desired_manifest = {
            "schema": 2,
            "source": source_selection,
            "source_signature": _path_signature(character_source_path),
            "height_policy": _AUTO_HEIGHT_MODE,
            "fallback_height_m": _AUTO_HEIGHT_FALLBACK_M,
        }

        ext = os.path.splitext(character_source_path)[1].lower()
        is_mesh_file = ext in (".fbx", ".glb", ".gltf")

        mesh_path = os.path.join(char_dir, "mesh.obj")
        rig_path = os.path.join(char_dir, "rigging", "mesh_ori_rig.txt")
        feature_dir = os.path.join(char_dir, "feature")
        outputs_ready = (
            os.path.isfile(mesh_path)
            and os.path.isfile(rig_path)
            and os.path.isdir(feature_dir)
            and bool(os.listdir(feature_dir))
        )
        cache_valid = outputs_ready and _read_manifest(manifest_path) == desired_manifest

        if force_reprocess or not cache_valid:
            _reset_stage_dir(char_dir, characters_root)
            os.makedirs(char_dir, exist_ok=True)
            if is_mesh_file:
                pipeline_src_path = os.path.join(char_dir, f"_source{ext}")
                shutil.copy(character_source_path, pipeline_src_path)
                log.append(f"Copied {ext} source into {pipeline_src_path}")
                out = run_blender_script(
                    node_script_path("mesh_to_morphgs.py"),
                    [pipeline_src_path, char_dir, _AUTO_HEIGHT_FALLBACK_M, _AUTO_HEIGHT_MODE],
                    timeout=300,
                )
                log.append(out)
            else:
                shutil.copytree(character_source_path, char_dir, dirs_exist_ok=True)
                log.append(f"Copied prepared character folder into {char_dir}")

            if not (os.path.isfile(mesh_path) and os.path.isfile(rig_path)):
                raise RuntimeError(
                    f"Character not ready after conversion: expected mesh.obj + rigging/mesh_ori_rig.txt "
                    f"under {char_dir}. Full log:\n" + "\n".join(log)
                )

            out = run_python(
                os.path.join(config.MORPHGS_HOME, "src", "preprocess", "preprocess_tgt.py"),
                [char_dir],
                timeout=1800,
            )
            log.append(out)
            if not os.path.isdir(feature_dir) or not os.listdir(feature_dir):
                raise RuntimeError(f"Character preprocessing produced no features under {feature_dir}")
            _write_manifest(manifest_path, desired_manifest)
        else:
            log.append("Character source and settings match the completed on-disk cache; skipping preprocessing")

        detected_height = 0.0
        conversion_meta = os.path.join(char_dir, "rigging", "conversion_meta.json")
        try:
            with open(conversion_meta, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            detected_height = float(metadata.get("detected_height_m", metadata.get("target_height", 0.0)))
            effective_height = float(metadata.get("target_height", detected_height))
            decision = metadata.get("height_decision", _AUTO_HEIGHT_MODE)
            log.append(
                f"Height: detected {detected_height:.4f} m; using {effective_height:.4f} m "
                f"({decision})"
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            log.append(
                "Height metadata is unavailable (expected for an already-prepared folder); "
                "the prepared mesh scale was preserved"
            )

        return (character_name, "\n".join(log), detected_height)


_SV4D_CHECKPOINTS = {
    "sv4d": ("stabilityai/sv4d2.0", "sv4d2.safetensors"),
    "sv4d2_8views": ("stabilityai/sv4d2.0", "sv4d2_8views.safetensors"),
    "sp4d": ("stabilityai/sp4d", "sp4d.safetensors"),
}


def _require_cuda_for_sv4d():
    """SV4D/DINO are GPU stages; never let a broken CUDA setup crawl on CPU unnoticed."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "MorphGS SV4D preprocessing requires a working CUDA GPU, but PyTorch reports "
            "torch.cuda.is_available() == False. The earlier FFmpeg/rembg preparation may use "
            "CPU, but SV4D will not be allowed to fall back to CPU. Check the ComfyUI console "
            "and its PyTorch/CUDA installation."
        )
    device = torch.cuda.current_device()
    return f"SV4D/DINO device: cuda:{device} ({torch.cuda.get_device_name(device)})"


def _sv4d_checkpoint_options():
    options = list(_SV4D_CHECKPOINTS.keys())
    try:
        import folder_paths

        known_filenames = {fname for _, fname in _SV4D_CHECKPOINTS.values()}
        available = [
            filename for filename in folder_paths.get_filename_list("morphgs_sv4d_checkpoints")
            if os.path.basename(filename) in known_filenames
        ]
        if available:
            options = available
    except Exception:
        pass  # No folder_paths available -- fall back to mode names for registry scanning.
    return options


# sgm's own attention implementations, and their mathematically-equivalent non-xformers
# counterparts. "vanilla" -> AttnBlock and "softmax" -> CrossAttention, both plain PyTorch.
_XFORMERS_ATTENTION_SUBSTITUTIONS = (
    ("vanilla-xformers", "vanilla"),
    ("softmax-xformers", "softmax"),
)


def _xformers_attention_works():
    """Whether xformers' fused attention can actually run on this GPU at the head dimension
    SV4D's VAE uses (512). Probed in a subprocess so a hard CUDA/native failure can't take
    ComfyUI down with it.

    Not assumed either way: xformers works fine on the GPUs SV4D was built for, and forcing
    the fallback everywhere would cost those users real VRAM headroom for no reason."""
    probe = (
        "import torch, xformers.ops as xops\n"
        "assert torch.cuda.is_available()\n"
        "q = torch.zeros((1, 64, 1, 512), dtype=torch.float16, device='cuda')\n"
        "xops.memory_efficient_attention(q, q, q)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, encoding="utf-8", errors="replace", cwd=config.MORPHGS_HOME,
        env=subprocess_env(),
    )
    return proc.returncode == 0


def _disable_xformers_in_sv4d_config(filename):
    """Switch the SV4D sampling config's attention blocks from xformers to plain PyTorch.

    sgm hardcodes `attn_type: vanilla-xformers` / `spatial_transformer_attn_type:
    softmax-xformers` in the SV4D sampling configs, and make_attn() in
    sgm/modules/diffusionmodules/model.py honours that with no fallback -- so on a GPU xformers
    has no kernel for, sampling dies with "No operator found for
    memory_efficient_attention_forward ... your GPU has capability (12, 0) (too new)".
    Confirmed on an RTX 5090: every candidate kernel was rejected, some for the architecture
    and some because SV4D's VAE attention uses head dim 512 (> the 256 flash-attention
    supports), so no xformers version would have helped.

    Blocking `import xformers` instead would be worse: temporal_ae.py falls back gracefully,
    but model.py would build a MemoryEfficientAttnBlock anyway and fail with a NameError.
    Rewriting the config is what actually routes around it.

    Only touches the config for the mode being run, and only when the probe above shows
    xformers genuinely can't serve it."""
    config_path = os.path.join(
        config.MORPHGS_HOME, "src", "extlibs", "generative-models",
        "scripts", "sampling", "configs", f"{os.path.splitext(filename)[0]}.yaml",
    )
    if not os.path.isfile(config_path):
        return f"No SV4D config at {config_path} to check for xformers attention."

    with open(config_path, encoding="utf-8") as f:
        original = f.read()
    patched = original
    for xformers_attn, plain_attn in _XFORMERS_ATTENTION_SUBSTITUTIONS:
        patched = patched.replace(xformers_attn, plain_attn)
    if patched == original:
        return f"{os.path.basename(config_path)} already uses non-xformers attention."

    if _xformers_attention_works():
        return (
            f"{os.path.basename(config_path)} uses xformers attention and this GPU supports "
            f"it -- leaving it alone."
        )

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(patched)
    return (
        f"Switched {os.path.basename(config_path)} to plain PyTorch attention: xformers has no "
        f"working kernel for this GPU at SV4D's attention head dim. Sampling will use more "
        f"VRAM and run slower, but it will actually run."
    )


_SGM_SDPA_CALL = "out = F.scaled_dot_product_attention("
_SGM_SDPA_CHUNKED_CALL = "out = _morphgs_sdpa("

# Appended to sgm's attention.py by the patch below. Written as source text rather than a
# monkeypatch because sgm calls F.scaled_dot_product_attention directly at module level in a
# subprocess we don't otherwise get to run code in.
_SGM_SDPA_HELPER = '''

# --- added by ComfyUI-MorphGS -------------------------------------------------------------
# CUDA caps gridDim.y and gridDim.z at 65535. Attention kernels map the batch dimension onto
# one of those, so a launch with a larger batch fails at launch time with cudaErrorInvalidValue
# -- surfaced by torch as a bare "CUDA error: invalid argument", before any real compute
# happens (which is why the GPU looks idle while it fails).
#
# SV4D hits this in its temporal blocks: spacetime_attention folds the spatial grid into the
# batch dimension, so batch becomes latents x H x W -- on the order of 500,000 at 576px with
# CFG, roughly 8x over the limit. The head dim and sequence length are unremarkable; it is
# purely the batch dimension that overflows.
#
# Attention is independent per batch element, so splitting the batch and concatenating is
# mathematically identical -- no approximation, no quality change, and the fast kernels stay
# in play instead of falling back to the math backend.
_MORPHGS_MAX_SDPA_BATCH = 32768


def _morphgs_sdpa(query, key, value, attn_mask=None, **kwargs):
    batch = query.shape[0]
    try:
        if batch <= _MORPHGS_MAX_SDPA_BATCH:
            return F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, **kwargs
            )
        chunks = []
        for start in range(0, batch, _MORPHGS_MAX_SDPA_BATCH):
            stop = min(start + _MORPHGS_MAX_SDPA_BATCH, batch)
            mask = attn_mask
            if torch.is_tensor(mask) and mask.dim() == query.dim() and mask.shape[0] == batch:
                mask = mask[start:stop]
            chunks.append(F.scaled_dot_product_attention(
                query[start:stop], key[start:stop], value[start:stop],
                attn_mask=mask, **kwargs
            ))
        return torch.cat(chunks, dim=0)
    except Exception as exc:
        # Never let this fail namelessly again: whatever goes wrong, say what was being asked
        # of the kernel, so the shape is in the error instead of having to be guessed at.
        raise type(exc)(
            f"{exc}\\n[ComfyUI-MorphGS] attention shapes: query={tuple(query.shape)} "
            f"key={tuple(key.shape)} value={tuple(value.shape)} dtype={query.dtype} "
            f"batch_chunk_limit={_MORPHGS_MAX_SDPA_BATCH}"
        ) from None
'''


def _chunk_sgm_attention_batches():
    """Split sgm's attention over the batch dimension so its kernel launches stay legal.

    Unconditional, deliberately. The previous two attempts at this failure each gated the fix
    behind a probe that ran SDPA standalone and asked "does this work here?", and each time the
    probe passed while the real run died -- because the probe reproduced the head dim, the
    sequence length and the backends, but always at batch=2, and the batch dimension is the one
    that actually overflows. A probe can only be trusted if it reproduces the real shape, and
    the real shape isn't known until the model is built. Chunking is exact and cheap, so it
    costs nothing to simply always apply it rather than predict whether it's needed."""
    attention_path = os.path.join(
        config.MORPHGS_HOME, "src", "extlibs", "generative-models",
        "sgm", "modules", "attention.py",
    )
    if not os.path.isfile(attention_path):
        return f"No sgm attention.py at {attention_path} to patch."

    with open(attention_path, encoding="utf-8") as f:
        original = f.read()
    if "_morphgs_sdpa" in original:
        return "sgm attention already chunks large batches."
    if _SGM_SDPA_CALL not in original:
        return (
            f"WARNING: could not find sgm's scaled_dot_product_attention call in "
            f"{attention_path} -- large-batch chunking NOT applied. Sampling may fail with "
            f"'CUDA error: invalid argument'."
        )

    patched = original.replace(_SGM_SDPA_CALL, _SGM_SDPA_CHUNKED_CALL) + _SGM_SDPA_HELPER
    with open(attention_path, "w", encoding="utf-8") as f:
        f.write(patched)
    return (
        "Patched sgm attention to split batches over 32768 before calling SDPA: CUDA's 65535 "
        "grid-dimension limit rejects the launch otherwise, and SV4D's temporal blocks fold "
        "the spatial grid into the batch dimension. Exact, not an approximation."
    )


_SGM_DECODE_SCALE = "        z = 1.0 / self.scale_factor * z\n"
_SGM_DECODE_SCALE_CAST = (
    "        z = 1.0 / self.scale_factor * z\n"
    "        z = _morphgs_match_dtype(z, self.first_stage_model)  # ComfyUI-MorphGS\n"
)

_SGM_DTYPE_HELPER = '''

# --- added by ComfyUI-MorphGS -------------------------------------------------------------
def _morphgs_match_dtype(z, module):
    """Cast latents to the dtype of the module that is about to consume them.

    SV4D's config sets disable_first_stage_autocast: True, so decode_first_stage runs with
    autocast off -- nothing casts anything automatically. The VAE is loaded in fp16 but the
    sampler returns fp32 latents, so the decoder's first conv gets "Input type (float) and bias
    type (c10::Half) should be the same". Under autocast this cast would have happened
    implicitly; doing it explicitly matches that behaviour without re-enabling autocast for the
    rest of the decode."""
    for param in module.parameters():
        return z.to(param.dtype)
    return z
'''


def _align_vae_decode_dtype():
    """Make sgm's first-stage decode cast latents to the VAE's own dtype."""
    diffusion_path = os.path.join(
        config.MORPHGS_HOME, "src", "extlibs", "generative-models",
        "sgm", "models", "diffusion.py",
    )
    if not os.path.isfile(diffusion_path):
        return f"No sgm diffusion.py at {diffusion_path} to patch."

    with open(diffusion_path, encoding="utf-8") as f:
        original = f.read()
    if "_morphgs_match_dtype" in original:
        return "sgm decode_first_stage already casts latents to the VAE dtype."
    if original.count(_SGM_DECODE_SCALE) != 1:
        return (
            f"WARNING: could not find the latent rescale line in {diffusion_path} -- VAE decode "
            f"dtype alignment NOT applied. Decoding may fail with 'Input type (float) and bias "
            f"type (c10::Half) should be the same'."
        )

    patched = original.replace(_SGM_DECODE_SCALE, _SGM_DECODE_SCALE_CAST) + _SGM_DTYPE_HELPER
    with open(diffusion_path, "w", encoding="utf-8") as f:
        f.write(patched)
    return (
        "Patched sgm decode_first_stage to cast latents to the VAE's dtype: SV4D disables "
        "autocast for decoding, so fp32 latents reach an fp16 VAE with nothing to reconcile "
        "them."
    )


def _experiment_model_dir(experiment):
    return os.path.join(config.MORPHGS_HOME, "output", experiment, "model", "morphgs")


def _available_iterations(experiment):
    """Training iterations that actually have a deform checkpoint on disk, ascending.

    Mirrors what MorphGS's own render.py does with _find_latest_iteration: the checkpoint
    that exists is the source of truth, not the iteration number a node happens to be set to."""
    deform_dir = os.path.join(_experiment_model_dir(experiment), "deform")
    found = []
    for path in glob.glob(os.path.join(deform_dir, "iteration_*.pth")):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            found.append(int(stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(found)


def _describe_iteration_artifacts(experiment):
    """What is actually on disk for this experiment, for use in an error message."""
    model_dir = _experiment_model_dir(experiment)
    if not os.path.isdir(model_dir):
        return f"Nothing has been trained yet: {model_dir} does not exist."
    parts = []
    for name in ("deform", "gaussians", "parametric", "render"):
        sub = os.path.join(model_dir, name)
        if not os.path.isdir(sub):
            parts.append(f"{name}/: missing")
            continue
        entries = sorted(os.listdir(sub))
        parts.append(
            f"{name}/: {', '.join(entries) if entries else 'empty'}"
            if len(entries) <= 8
            else f"{name}/: {len(entries)} files, e.g. {', '.join(entries[:8])}"
        )
    return f"Contents of {model_dir} --\n  " + "\n  ".join(parts)


def _reporting_setup_log(log, call):
    """Run a pipeline step, and if it fails, put the setup log in front of the error.

    The environment fixes this node applies before sampling (checkpoint staging, the xformers
    config rewrite, the SDPA backend pin) each report what they decided into `log` -- but `log`
    is only returned on success, so a failure downstream discarded exactly the information
    needed to tell "the fix didn't work" apart from "the fix never ran". Chained with `from
    None`: str(exc) already carries the subprocess's full output, so re-showing the original
    traceback would only duplicate it."""
    try:
        return call()
    except RuntimeError as exc:
        raise RuntimeError(
            "--- setup steps before the failure ---\n" + "\n".join(log) + f"\n\n{exc}"
        ) from None


def _stage_sv4d_checkpoint(ckpt_path, filename):
    """Make the checkpoint reachable at the exact path SV4D actually loads it from.

    MorphGS's preprocess_src.py chdir's into src/extlibs/generative-models and loads the
    checkpoint by the relative path "checkpoints/<name>.safetensors" hardcoded in
    generative-models' own sampling configs. So wherever the user keeps the actual file
    (ComfyUI's models/diffusion_models, models/sv4d, models/checkpoints, ...), it ALSO has to be reachable at that one
    relative location, or the load dies with a bare "FileNotFoundError: No such file or
    directory: checkpoints/sv4d2.safetensors" deep inside sgm -- confirmed in practice, with
    the file sitting correctly in a registered ComfyUI model folder the whole time.

    Linked rather than copied: these checkpoints are ~12GB. Falls back hardlink -> copy for
    filesystems that don't allow symlinks."""
    staged_dir = os.path.join(
        config.MORPHGS_HOME, "src", "extlibs", "generative-models", "checkpoints"
    )
    staged_path = os.path.join(staged_dir, filename)
    os.makedirs(staged_dir, exist_ok=True)

    # lexists, not exists: a BROKEN symlink (e.g. left by a previously-moved checkpoint) is
    # invisible to exists() but still blocks creating a new link over it.
    if os.path.lexists(staged_path):
        if os.path.exists(staged_path) and os.path.realpath(staged_path) == os.path.realpath(ckpt_path):
            return f"SV4D checkpoint already in place at {staged_path}"
        os.remove(staged_path)

    for link, describe in (
        (os.symlink, "Symlinked"),
        (os.link, "Hard-linked"),
    ):
        try:
            link(ckpt_path, staged_path)
            return f"{describe} {ckpt_path} -> {staged_path}"
        except OSError:
            continue
    shutil.copy(ckpt_path, staged_path)
    return f"Copied {ckpt_path} -> {staged_path} (neither symlink nor hardlink was available)"


def _resolve_sv4d_selection(selection):
    """Accepts either a friendly mode name (sv4d/sp4d/sv4d2_8views, the fallback shown when
    the morphgs_sv4d_checkpoints folder can't be listed -- e.g. the Comfy Registry's isolated
    node scanner) or a real checkpoint filename picked from that folder, and returns
    (mode, filename) either way."""
    if selection in _SV4D_CHECKPOINTS:
        _, filename = _SV4D_CHECKPOINTS[selection]
        return selection, filename
    filename = os.path.basename(selection)
    for mode, (_, fname) in _SV4D_CHECKPOINTS.items():
        if fname == filename:
            return mode, fname
    raise ValueError(f"Unrecognized SV4D checkpoint selection: {selection!r}")


class MorphGSPreprocessVideo:
    """
    Prepares a source video for MorphGS: segments + composites onto a white square
    background if needed, then runs SV4D/SP4D multi-view synthesis + feature extraction.
    video_path is a dropdown of video files found under ComfyUI's own input/ directory --
    drop your clip in there and it shows up here, no manual path-typing needed.

    sv4d_mode is a real dropdown of SV4D/SP4D checkpoints found under the
    morphgs_sv4d_checkpoints category (registered by this package at load time, via
    folder_paths.get_filename_list) -- not a fixed list of names. This package creates and
    scans ComfyUI's standard models/diffusion_models folder first, then the dedicated
    models/sv4d fallback, models/checkpoints, and MorphGS's own generative-models checkout.
    There is no node that downloads it for you:
    download the file yourself from
      - sv4d / sv4d2_8views: https://huggingface.co/stabilityai/sv4d2.0
      - sp4d: https://huggingface.co/stabilityai/sp4d
    and place it in your ComfyUI models/diffusion_models folder. Falls back to mode names
    when folder_paths can't be listed (e.g. the Comfy Registry's isolated node scanner, which
    has no `folder_paths` module at all). The web extension's explicit refresh button rescans
    this list at runtime; it does not refresh after each generation.

    max_frames caps how much of the clip SV4D synthesises, 0 meaning the whole thing. SV4D is
    by far the most expensive stage here -- it diffuses 12 frames at a time, so cost scales
    directly with length and a couple of hundred frames is a ~20-minute run. Set it to 12 or 24
    to take the whole pipeline end-to-end for a fraction of that while checking a setup, then
    put it back to 0 for the real render.
    """

    @classmethod
    def INPUT_TYPES(cls):
        sv4d_options = _sv4d_checkpoint_options()

        video_options = _video_input_options() or [""]

        return {
            "required": {
                "video_path": (video_options, {
                    "tooltip": "Choose a video already under ComfyUI/input, or use Upload video.",
                }),
                "already_masked": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Enable only when the clip already has MorphGS-ready masking/framing. "
                        "This skips rembg segmentation (the main CPU/ONNX preparation stage)."
                    ),
                }),
                "sv4d_mode": (sv4d_options, {
                    "default": sv4d_options[0],
                    "tooltip": "SV4D/SP4D checkpoint scanned from ComfyUI models/diffusion_models and fallback model folders.",
                }),
                "fastmode": ("BOOLEAN", {"default": True}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 12}),
                "force_reprocess": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Normally leave off: completed outputs are reused when the video, checkpoint, "
                        "and settings match. Turn on only to discard them and run masking + SV4D again."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("scene_name", "log")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True  # Lets this run (and its result be visible) standalone before it's
    # wired to anything downstream -- otherwise ComfyUI's execution graph would prune it out
    # entirely and queuing it alone would do nothing (confirmed via a real API test on an
    # OUTPUT_NODE-less node: "Prompt has no outputs").

    @classmethod
    def IS_CHANGED(cls, video_path, already_masked, sv4d_mode, fastmode, max_frames,
                   force_reprocess):
        if force_reprocess:
            return float("NaN")
        return json.dumps({
            "source": _input_change_token(video_path, False),
            "already_masked": bool(already_masked),
            "sv4d_mode": sv4d_mode,
            "fastmode": bool(fastmode),
            "max_frames": int(max_frames),
        }, sort_keys=True, separators=(",", ":"))

    def run(self, video_path, already_masked, sv4d_mode, fastmode, max_frames,
            force_reprocess):
        log = []
        source_selection = video_path
        video_path = _resolve_input_path(source_selection)
        scene_name = _stage_name_from_input(source_selection, "scene_name")
        mode, filename = _resolve_sv4d_selection(sv4d_mode)

        import folder_paths

        ckpt_path = folder_paths.get_full_path("morphgs_sv4d_checkpoints", filename)
        if not ckpt_path:
            hf_repo, _ = _SV4D_CHECKPOINTS[mode]
            raise RuntimeError(
                f"SV4D checkpoint '{filename}' not found. Download it from "
                f"https://huggingface.co/{hf_repo} and place it in your ComfyUI "
                f"models/diffusion_models folder (models/sv4d and models/checkpoints also work)."
            )
        log.append(_stage_sv4d_checkpoint(ckpt_path, filename))
        log.append(_disable_xformers_in_sv4d_config(filename))
        log.append(_chunk_sgm_attention_batches())
        log.append(_align_vae_decode_dtype())

        videos_root = os.path.join(config.MORPHGS_HOME, "demo", "videos")
        processed_root = os.path.join(config.MORPHGS_HOME, "demo", "processed_videos")
        scene_dir = os.path.join(videos_root, scene_name)
        rgb_path = os.path.join(scene_dir, "rgb.mp4")
        processed_dir = os.path.join(processed_root, scene_name)
        manifest_path = os.path.join(processed_dir, _CACHE_MANIFEST)
        desired_manifest = {
            "schema": 1,
            "source": source_selection,
            "source_signature": _path_signature(video_path),
            "already_masked": bool(already_masked),
            "sv4d_mode": mode,
            "checkpoint": filename,
            "checkpoint_signature": _path_signature(ckpt_path),
            "fastmode": bool(fastmode),
            "max_frames": int(max_frames),
        }
        cache_valid = (
            os.path.isfile(rgb_path)
            and os.path.isdir(processed_dir)
            and _read_manifest(manifest_path) == desired_manifest
        )

        if force_reprocess or not cache_valid:
            log.append(_require_cuda_for_sv4d())
            _reset_stage_dir(scene_dir, videos_root)
            _reset_stage_dir(processed_dir, processed_root)
            os.makedirs(scene_dir, exist_ok=True)
            pipeline_src_video = os.path.join(scene_dir, f"_source{os.path.splitext(video_path)[1]}")
            shutil.copy(video_path, pipeline_src_video)

            mask_args = [pipeline_src_video, rgb_path]
            if already_masked:
                mask_args.append("--skip-mask")
            out = _reporting_setup_log(
                log, lambda: run_python(node_script_path("mask_video.py"), mask_args, timeout=1800)
            )
            log.append(out)

            args = [rgb_path, "--mode", mode]
            if fastmode:
                args.append("--fastmode")
            if max_frames > 0:
                args += ["--sv4d_max_frames", max_frames]
                log.append(
                    f"Limiting SV4D to the first {max_frames} frames of {scene_name}."
                )
            out = _reporting_setup_log(log, lambda: run_python(
                os.path.join(config.MORPHGS_HOME, "src", "preprocess", "preprocess_src.py"),
                args,
                timeout=3600,
            ))
            log.append(out)
            if not os.path.isdir(processed_dir):
                raise RuntimeError(f"Video preprocessing produced no output directory at {processed_dir}")
            _write_manifest(manifest_path, desired_manifest)
        else:
            log.append("Video source and settings match the completed on-disk cache; skipping preprocessing")

        return (scene_name, "\n".join(log))


class MorphGSTrainAndRender:
    """
    Registers/trains the <scene>_to_<character> experiment and returns the rendered
    output video, both as a file path and as an IMAGE batch for in-graph preview.

    seed controls MorphGS's own training-time randomness (Gaussian initialization, sampling --
    threaded through to main.py as --project.seed, the same config field MorphGS's own configs
    set to 43 by default) and has the standard ComfyUI seed widget next to it
    (fixed/increment/decrement/randomize). The disk-cache manifest includes this seed and both
    preprocessing manifests, so a changed seed, character, video, or preprocessing setting
    automatically invalidates the corresponding training result.

    Training uses a real on-disk manifest rather than only ComfyUI's in-memory result cache,
    because training can take hours and must survive a ComfyUI restart. Changes to the source
    preprocessing manifests, iteration count, or seed automatically invalidate it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_name": ("STRING", {"default": ""}),
                "character_name": ("STRING", {"default": ""}),
                "iterations": ("INT", {"default": 5000, "min": 100, "max": 100000, "step": 100}),
                "seed": ("INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            }
        }

    # scene_name/character_name are passed straight through, unchanged, purely so Export
    # Animated Mesh can take them from HERE rather than from the preprocess nodes. Without
    # that edge, Export depends only on the preprocess nodes -- exactly as this node does --
    # so the two are siblings in the graph with no ordering between them, and ComfyUI is free
    # to run Export first. It then fails with "no deform checkpoint" because training hasn't
    # happened yet. Appended after the existing outputs rather than inserted before them, so
    # link indices in graphs built against earlier versions keep pointing at the same sockets.
    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_path", "frames", "log", "scene_name", "character_name")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True  # Lets this run (and its result be visible) standalone before it's
    # wired to anything downstream -- otherwise ComfyUI's execution graph would prune it out
    # entirely and queuing it alone would do nothing (confirmed via a real API test on an
    # OUTPUT_NODE-less node: "Prompt has no outputs").

    @classmethod
    def IS_CHANGED(cls, scene_name, character_name, iterations, seed):
        try:
            scene_name = _safe_stage_name(scene_name, "scene_name")
            character_name = _safe_stage_name(character_name, "character_name")
            token = {
                "scene": _stage_signature(os.path.join(config.MORPHGS_HOME, "demo", "processed_videos", scene_name)),
                "character": _stage_signature(os.path.join(config.MORPHGS_HOME, "demo", "characters", character_name)),
            }
            return json.dumps(token, sort_keys=True, separators=(",", ":"))
        except (OSError, ValueError):
            return float("NaN")

    def run(self, scene_name, character_name, iterations, seed):
        log = []
        scene_name = _safe_stage_name(scene_name, "scene_name")
        character_name = _safe_stage_name(character_name, "character_name")
        experiment = f"{scene_name}_to_{character_name}"
        config_path = os.path.join(config.MORPHGS_HOME, "configs", "demo", f"{experiment}.yaml")
        model_dir = _experiment_model_dir(experiment)
        render_path = os.path.join(
            model_dir, "render",
            f"rendered_video_{iterations}.mp4",
        )
        # main.py writes the deform checkpoint and the render video in the same block, so a run
        # that finished produces both. Treating the video alone as "already trained" meant a
        # half-populated output directory silently skipped training and reported success, and
        # the failure only surfaced later in Export Animated Mesh as a missing checkpoint with
        # nothing to explain it. Skip only when everything downstream needs is actually present.
        deform_path = os.path.join(
            model_dir, "deform",
            f"iteration_{iterations}.pth",
        )
        manifest_path = os.path.join(model_dir, f".morphgs_train_cache_{iterations}.json")
        desired_manifest = {
            "schema": 1,
            "scene": _stage_signature(os.path.join(config.MORPHGS_HOME, "demo", "processed_videos", scene_name)),
            "character": _stage_signature(os.path.join(config.MORPHGS_HOME, "demo", "characters", character_name)),
            "iterations": int(iterations),
            "seed": int(seed),
        }
        cache_valid = (
            os.path.isfile(render_path)
            and os.path.isfile(deform_path)
            and _read_manifest(manifest_path) == desired_manifest
        )

        if not os.path.isfile(config_path):
            # An empty file fails yaml.safe_load/DotDict (returns None, not {}), so this must be
            # a valid empty YAML mapping for main.py's merge_configs(base_config, _config).
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                f.write("{}\n")
            log.append(f"Created minimal experiment config at {config_path} (defaults from configs/base.yaml)")

        if not cache_valid:
            if os.path.isfile(manifest_path):
                os.remove(manifest_path)
            out = run_python(
                os.path.join(config.MORPHGS_HOME, "src", "main.py"),
                [
                    "--config", f"demo/{experiment}.yaml",
                    f"--model.opt.iterations={iterations}",
                    f"--project.seed={seed}",
                ],
                timeout=None,
            )
            log.append(out)
        else:
            log.append(
                f"Rendered output and deform checkpoint already exist for iteration "
                f"{iterations}, skipping training"
            )

        for label, path in (("rendered video", render_path), ("deform checkpoint", deform_path)):
            if not os.path.isfile(path):
                raise RuntimeError(
                    f"Training finished but no {label} was produced at {path}. "
                    f"{_describe_iteration_artifacts(experiment)}\nFull log:\n" + "\n".join(log)
                )
        _write_manifest(manifest_path, desired_manifest)

        import folder_paths

        output_dir = folder_paths.get_output_directory()
        local_video_path = os.path.join(output_dir, "morphgs", f"{experiment}_{iterations}.mp4")
        os.makedirs(os.path.dirname(local_video_path), exist_ok=True)
        shutil.copy(render_path, local_video_path)
        log.append(f"Copied result to {local_video_path}")

        frames = self._load_video_as_tensor(local_video_path)
        return (local_video_path, frames, "\n".join(log), scene_name, character_name)

    @staticmethod
    def _load_video_as_tensor(video_path):
        import cv2
        import numpy as np
        import torch

        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32) / 255.0)
        cap.release()
        if not frames:
            raise RuntimeError(f"Could not decode any frames from {video_path}")
        return torch.from_numpy(np.stack(frames))


class MorphGSExportAnimatedMesh:
    """
    Exports MorphGS's trained per-scene motion as a real, standalone animated 3D mesh
    (glTF/GLB or FBX) -- not just a rendered video. Always starts with
    extract_pose_sequence.py, which replays the trained AnimationField/SimpleDeformNet
    checkpoint frame-by-frame to get absolute per-joint world-space transforms (the same FK
    code MorphGS's own training loop uses). Then bakes those transforms onto a skinned
    character mesh in one of two ways:
      - If the character has its original rigged source file on disk (written by MorphGS:
        Preprocess Character as _source.<ext>, or placed there directly), bake_animation.py
        keyframes the pose onto that file's own armature, applying the scale correction used
        when the character was first converted.
      - Otherwise (e.g. MorphGS's own bundled demo characters, which ship only as mesh.obj +
        a RigNet-format rig file with no original rigged file anywhere), the more general
        build_and_bake_animation.py builds a fresh skinned armature directly from mesh.obj +
        mesh_ori_rig.txt's own joint positions and per-vertex skin weights -- that file already
        contains everything needed, since MorphGS's rig format is a full RigNet rig, not just a
        skeleton. resolve_skinning_weights.py runs first in this case, since some characters'
        configs apply heat-diffusion smoothing (or heat-based recalculation) to the rig file's
        raw skin weights before training -- invisible at rest pose but causing severe mesh
        distortion under real motion if the raw weights are used unmodified.
    Requires MorphGS: Preprocess Character and MorphGS: Train & Render to have already
    been run for this character/scene pair. Wire scene_name/character_name from Train &
    Render's own passthrough outputs, NOT from the preprocess nodes: ComfyUI orders execution
    by data dependency, so taking them from the preprocess nodes leaves this node and Train &
    Render as unordered siblings and lets this one run first, against an output directory
    training has not written yet.

    Saves via the same folder_paths.get_save_image_path() convention ComfyUI's own built-in
    SaveGLB node uses, and returns the matching {"ui": {"3d": [...]}} payload -- so this node
    shows the result directly in ComfyUI's native interactive 3D viewer widget as soon as it
    finishes, with no separate downstream node needed. preview_path is ALSO returned as a real
    socket (the same output-dir-relative "subfolder/filename" string ComfyUI-Hunyuan3DWrapper's
    own Hy3DExportMesh returns) so this node can additionally be wired into ComfyUI's native
    "Preview 3D & Animation" (Preview3D) node when you want that separate, explicit node in the
    graph -- e.g. to view the result at a different point than right after export, or to record
    a fixed camera angle via Preview3D's optional camera_info input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_name": ("STRING", {"default": ""}),
                "character_name": ("STRING", {"default": ""}),
                "iterations": ("INT", {"default": 5000, "min": 100, "max": 100000, "step": 100}),
                "output_format": (["glb", "fbx"], {"default": "glb"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("mesh_path", "preview_path", "log")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, scene_name, character_name, iterations, output_format):
        try:
            experiment = (
                f"{_safe_stage_name(scene_name, 'scene_name')}_to_"
                f"{_safe_stage_name(character_name, 'character_name')}"
            )
            available = _available_iterations(experiment)
            requested_iteration = int(iterations)
            resolved_iteration = (
                requested_iteration if requested_iteration in available
                else available[-1] if available
                else requested_iteration
            )
            deform_path = os.path.join(
                _experiment_model_dir(experiment), "deform", f"iteration_{resolved_iteration}.pth"
            )
            return json.dumps(_path_signature(deform_path), sort_keys=True, separators=(",", ":"))
        except (OSError, ValueError):
            return float("NaN")

    def run(self, scene_name, character_name, iterations, output_format):
        log = []
        scene_name = _safe_stage_name(scene_name, "scene_name")
        character_name = _safe_stage_name(character_name, "character_name")
        experiment = f"{scene_name}_to_{character_name}"
        char_dir = os.path.join(config.MORPHGS_HOME, "demo", "characters", character_name)
        video_dir = os.path.join(config.MORPHGS_HOME, "demo", "videos", scene_name)
        rig_path = os.path.join(char_dir, "rigging", "mesh_ori_rig.txt")
        meta_path = os.path.join(char_dir, "rigging", "conversion_meta.json")
        mesh_obj_path = os.path.join(char_dir, "mesh.obj")
        # Resolve against what was actually trained rather than trusting this node's own
        # iterations widget, which is a separate value from Train & Render's and silently
        # disagrees the moment one of the two is changed. MorphGS's own render.py does exactly
        # this (_find_latest_iteration) when no iteration is given.
        available = _available_iterations(experiment)
        if iterations not in available:
            if not available:
                raise RuntimeError(
                    f"No deform checkpoint found for '{experiment}'. Run MorphGS: Preprocess "
                    f"Character and MorphGS: Train & Render first.\n"
                    f"{_describe_iteration_artifacts(experiment)}"
                )
            resolved = max(available)
            log.append(
                f"No deform checkpoint for iteration {iterations}; using the latest trained "
                f"iteration {resolved} instead (available: "
                f"{', '.join(str(i) for i in available)}). Set iterations to {resolved} on this "
                f"node to match Train & Render and silence this."
            )
            iterations = resolved

        ckpt_path = os.path.join(
            _experiment_model_dir(experiment), "deform", f"iteration_{iterations}.pth",
        )
        render_dir = os.path.join(_experiment_model_dir(experiment), "render")
        pose_npz_path = os.path.join(render_dir, f"pose_sequence_{iterations}.npz")
        exported_path = os.path.join(render_dir, f"animated_mesh_{iterations}.{output_format}")
        export_manifest_path = os.path.join(render_dir, f".morphgs_export_cache_{iterations}_{output_format}.json")

        for label, path in [("rig", rig_path), ("mesh", mesh_obj_path)]:
            if not os.path.isfile(path):
                raise RuntimeError(
                    f"Required {label} file not found at {path}. Run MorphGS: Preprocess Character "
                    f"and MorphGS: Train & Render for '{experiment}' first."
                )

        desired_export_manifest = {
            "schema": 1,
            "deform_checkpoint": _path_signature(ckpt_path),
            "mesh": _path_signature(mesh_obj_path),
            "rig": _path_signature(rig_path),
            "output_format": output_format,
        }
        export_cache_valid = (
            os.path.isfile(exported_path)
            and _read_manifest(export_manifest_path) == desired_export_manifest
        )

        # mesh_to_morphgs.py (run by MorphGS: Preprocess Character) copies the original rigged
        # source file into the pipeline as _source.<ext> when one exists, and characters
        # prepared via the "pre-prepared folder" path (or set up manually) may instead have it
        # sitting directly under char_dir under its own name -- either way, that's preferred
        # for baking since it carries the character's own original bone rest orientations. If
        # neither exists (e.g. MorphGS's own bundled demo characters), fall back to building a
        # fresh armature directly from mesh.obj + mesh_ori_rig.txt below.
        original_rigged_path = ""
        for pattern in (
            os.path.join(char_dir, "_source.*"),
            os.path.join(char_dir, "*.fbx"),
            os.path.join(char_dir, "*.glb"),
            os.path.join(char_dir, "*.gltf"),
        ):
            matches = glob.glob(pattern)
            if matches:
                original_rigged_path = matches[0]
                break

        meta_exists = os.path.isfile(meta_path)
        use_original_file = bool(original_rigged_path) and meta_exists
        if original_rigged_path and not meta_exists:
            log.append(
                f"Found {original_rigged_path} but no conversion_meta.json alongside it -- can't "
                f"apply the matching scale correction, so building a fresh armature from mesh.obj "
                f"+ mesh_ori_rig.txt instead."
            )

        # MorphGS normalizes per-frame time as frame_index / NF (main.py's
        # cam_t = frame_idx_by_cam[id(view)] / NF), where NF = len(cams_by_view[gt_views[0]])
        # -- the frame count of the SV4D-*processed* view sequence, NOT the raw input video's
        # frame count (SV4D's windowed multi-view synthesis can produce a different total, e.g.
        # 66 processed frames from a 70-frame source video). Extracting with the wrong NF would
        # silently desync every frame's time embedding from what was actually trained.
        processed_view0_dir = os.path.join(
            config.MORPHGS_HOME, "demo", "processed_videos", scene_name, "view_0", "color"
        )
        if not os.path.isdir(processed_view0_dir):
            raise RuntimeError(
                f"Could not find processed frames at {processed_view0_dir}. "
                f"Run MorphGS: Preprocess Video for '{scene_name}' first."
            )
        num_frames = len(os.listdir(processed_view0_dir))
        if num_frames <= 0:
            raise RuntimeError(f"No processed frames found at {processed_view0_dir}.")

        import cv2

        cap = cv2.VideoCapture(os.path.join(video_dir, "rgb.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        log.append(
            f"Using NF={num_frames} (from {processed_view0_dir}, matching main.py's training-time "
            f"normalization) at {fps:.3f} fps (from {video_dir}/rgb.mp4)"
        )

        if not export_cache_valid:
            if os.path.isfile(export_manifest_path):
                os.remove(export_manifest_path)
            os.makedirs(render_dir, exist_ok=True)
            morphgs_src_path = os.path.join(config.MORPHGS_HOME, "src")
            pythonpath_env = {
                "PYTHONPATH": morphgs_src_path + os.pathsep + os.environ.get("PYTHONPATH", "")
            }
            extract_out = run_python(
                node_script_path("extract_pose_sequence.py"),
                [rig_path, ckpt_path, num_frames, pose_npz_path],
                timeout=600,
                env=pythonpath_env,
            )
            log.append(extract_out)

            if use_original_file:
                log.append(f"Baking onto original rigged file: {original_rigged_path}")
                bake_out = run_blender_script(
                    node_script_path("bake_animation.py"),
                    [original_rigged_path, pose_npz_path, meta_path, fps, exported_path],
                    timeout=600,
                )
            else:
                log.append("No usable original rigged file -- building armature from mesh.obj + rig file")

                # Some characters' configs apply heat-diffusion smoothing (or heat-based
                # recalculation) to mesh_ori_rig.txt's raw skin weights before training --
                # invisible at rest pose but causing severe mesh distortion under real motion
                # if skipped (confirmed on MorphGS's own bundled chickenDC/moose1DOG demo
                # characters). Resolve the actual weights used before baking.
                exp_config_path = os.path.join(config.MORPHGS_HOME, "configs", "demo", f"{experiment}.yaml")
                base_config_path = os.path.join(config.MORPHGS_HOME, "configs", "base.yaml")
                resolved_weights_path = os.path.join(render_dir, "resolved_skinning_weights.npz")
                resolve_out = run_python(
                    node_script_path("resolve_skinning_weights.py"),
                    [mesh_obj_path, rig_path, exp_config_path, base_config_path, resolved_weights_path],
                    timeout=600,
                    env=pythonpath_env,
                )
                log.append(resolve_out)

                bake_out = run_blender_script(
                    node_script_path("build_and_bake_animation.py"),
                    [mesh_obj_path, rig_path, pose_npz_path, fps, exported_path, resolved_weights_path],
                    timeout=600,
                )
            log.append(bake_out)
            if not os.path.isfile(exported_path):
                raise RuntimeError(f"Animation export produced no file at {exported_path}")
            _write_manifest(export_manifest_path, desired_export_manifest)
        else:
            log.append("Training checkpoint and character inputs match the exported-mesh cache; skipping export")

        if not os.path.isfile(exported_path):
            raise RuntimeError(
                f"Expected animated mesh at {exported_path} but it was not produced. Full log:\n" + "\n".join(log)
            )

        import folder_paths

        # Save via the same folder_paths.get_save_image_path() convention ComfyUI's own
        # built-in SaveGLB node uses (despite the name, it's a generic numbered-output-path
        # helper, not image-specific) so the result lands somewhere the frontend can serve it,
        # and return the matching {"ui": {"3d": [...]}} payload so this node shows the animated
        # mesh directly in ComfyUI's native interactive 3D viewer -- the same mechanism SaveGLB
        # uses -- without needing a separate downstream Preview3D node.
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            f"morphgs/{experiment}", folder_paths.get_output_directory()
        )
        saved_filename = f"{filename}_{counter:05}_.{output_format}"
        local_mesh_path = os.path.join(full_output_folder, saved_filename)
        os.makedirs(full_output_folder, exist_ok=True)
        shutil.copy(exported_path, local_mesh_path)
        log.append(f"Copied result to {local_mesh_path}")

        # Same output-dir-relative shape ComfyUI-Hunyuan3DWrapper's Hy3DExportMesh returns
        # (str(Path(subfolder) / filename)) -- what Preview3D's plain-string input expects,
        # since it resolves the file via ComfyUI's own /view route, not an OS filesystem path.
        preview_path = f"{subfolder}/{saved_filename}" if subfolder else saved_filename

        ui = {"3d": [{"filename": saved_filename, "subfolder": subfolder, "type": "output"}]}
        return {"ui": ui, "result": (local_mesh_path, preview_path, "\n".join(log))}


NODE_CLASS_MAPPINGS = {
    "MorphGSPreprocessCharacter": MorphGSPreprocessCharacter,
    "MorphGSPreprocessVideo": MorphGSPreprocessVideo,
    "MorphGSTrainAndRender": MorphGSTrainAndRender,
    "MorphGSExportAnimatedMesh": MorphGSExportAnimatedMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MorphGSPreprocessCharacter": "MorphGS: Preprocess Character",
    "MorphGSPreprocessVideo": "MorphGS: Preprocess Video",
    "MorphGSTrainAndRender": "MorphGS: Train & Render",
    "MorphGSExportAnimatedMesh": "MorphGS: Export Animated Mesh",
}
