import glob
import os
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
from .process_utils import node_script_path, run_blender_script, run_python

CATEGORY = "MorphGS"


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


def _resolve_input_path(selection):
    """Resolve a value picked from _list_input_files/_list_input_prepared_folders's dropdown
    back into a real filesystem path under ComfyUI's input/ directory."""
    import folder_paths

    return os.path.join(folder_paths.get_input_directory(), *selection.split("/"))


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
        options = _list_input_files({".fbx", ".glb", ".gltf"}) + _list_input_prepared_folders()
        if not options:
            options = [""]
        return {
            "required": {
                "character_source_path": (options, {}),
                "character_name": ("STRING", {"default": "my_character"}),
                "target_height": ("FLOAT", {"default": 1.6, "min": 0.1, "max": 10.0, "step": 0.1}),
                "force_reprocess": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("character_name", "log")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True  # Lets this run (and its result be visible) standalone before it's
    # wired to anything downstream -- otherwise ComfyUI's execution graph would prune it out
    # entirely and queuing it alone would do nothing (confirmed via a real API test on an
    # OUTPUT_NODE-less node: "Prompt has no outputs").

    def run(self, character_source_path, character_name, target_height, force_reprocess):
        log = []
        character_source_path = _resolve_input_path(character_source_path)
        char_dir = os.path.join(config.MORPHGS_HOME, "demo", "characters", character_name)

        ext = os.path.splitext(character_source_path)[1].lower()
        is_mesh_file = ext in (".fbx", ".glb", ".gltf")

        if is_mesh_file:
            os.makedirs(char_dir, exist_ok=True)
            pipeline_src_path = os.path.join(char_dir, f"_source{ext}")
            shutil.copy(character_source_path, pipeline_src_path)
            log.append(f"Copied {ext} source into {pipeline_src_path}")

            mesh_path = os.path.join(char_dir, "mesh.obj")
            if force_reprocess or not os.path.isfile(mesh_path):
                out = run_blender_script(
                    node_script_path("mesh_to_morphgs.py"),
                    [pipeline_src_path, char_dir, target_height],
                    timeout=300,
                )
                log.append(out)
            else:
                log.append("mesh.obj already exists, skipping mesh conversion (force_reprocess=False)")
        else:
            # Treat character_source_path as a pre-prepared folder (mesh.obj + rigging/mesh_ori_rig.txt).
            # A user may point this directly at the character's own canonical location (e.g.
            # they already staged files there by hand, or re-ran with the same path) -- copying
            # a directory into itself would be wrong, so check via realpath first.
            if os.path.isdir(char_dir) and os.path.realpath(character_source_path) == os.path.realpath(char_dir):
                log.append(f"character_source_path is already {char_dir}, nothing to copy")
            else:
                os.makedirs(char_dir, exist_ok=True)
                shutil.copytree(character_source_path, char_dir, dirs_exist_ok=True)
                log.append(f"Copied prepared character folder into {char_dir}")

        mesh_path = os.path.join(char_dir, "mesh.obj")
        rig_path = os.path.join(char_dir, "rigging", "mesh_ori_rig.txt")
        if not (os.path.isfile(mesh_path) and os.path.isfile(rig_path)):
            # Blender can exit 0 (success) even when the --python script it ran hit an
            # uncaught exception partway through -- it doesn't set a non-zero exit code for
            # that on its own, so run_blender_script's own non-zero-exit check doesn't catch
            # it. Surface the collected log (Blender's actual stdout/stderr, including any
            # traceback) here instead of a contextless message, matching every other node's
            # final failure check in this file.
            raise RuntimeError(
                f"Character not ready after conversion: expected mesh.obj + rigging/mesh_ori_rig.txt "
                f"under {char_dir}. Full log:\n" + "\n".join(log)
            )

        feature_dir = os.path.join(char_dir, "feature")
        feat_ready = os.path.isdir(feature_dir) and len(os.listdir(feature_dir)) > 0
        if force_reprocess or not feat_ready:
            out = run_python(
                os.path.join(config.MORPHGS_HOME, "src", "preprocess", "preprocess_tgt.py"),
                [char_dir],
                timeout=1800,
            )
            log.append(out)
        else:
            log.append("Rendered views + features already exist, skipping preprocess_tgt.py")

        return (character_name, "\n".join(log))


_SV4D_CHECKPOINTS = {
    "sv4d": ("stabilityai/sv4d2.0", "sv4d2.safetensors"),
    "sv4d2_8views": ("stabilityai/sv4d2.0", "sv4d2_8views.safetensors"),
    "sp4d": ("stabilityai/sp4d", "sp4d.safetensors"),
}


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


def _sdpa_all_backends_work():
    """Whether PyTorch's own attention survives sgm's "enable everything, let torch pick"
    backend setting at SV4D's head dim (512). Probed in a subprocess, synchronized so an async
    CUDA fault surfaces here rather than later."""
    probe = (
        "import torch, torch.nn.functional as F\n"
        "from torch.backends.cuda import sdp_kernel\n"
        "assert torch.cuda.is_available()\n"
        "q = torch.zeros((1, 1, 64, 512), dtype=torch.float16, device='cuda')\n"
        "with sdp_kernel(enable_math=True, enable_flash=True, enable_mem_efficient=True):\n"
        "    F.scaled_dot_product_attention(q, q, q)\n"
        "torch.cuda.synchronize()\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, encoding="utf-8", errors="replace", cwd=config.MORPHGS_HOME,
    )
    return proc.returncode == 0


_SGM_SDPA_ALL_BACKENDS = (
    'None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},'
)
_SGM_SDPA_MATH_ONLY = (
    'None: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},'
)


def _force_math_sdpa_in_sgm():
    """Pin sgm's attention to PyTorch's math SDPA backend when the others can't serve it.

    sgm's CrossAttention defaults to backend=None, which it maps to "enable flash + mem
    efficient + math and let torch choose". At SV4D's head dim of 512 on a new GPU that picks a
    kernel that dies with "CUDA error: invalid argument" (confirmed on an RTX 5090, immediately
    after the xformers path was already routed around). The math backend has no such limits --
    slower and more memory, but it always computes.

    Same reasoning as the config rewrite: only applied when the probe shows the default
    genuinely fails here, so machines where flash/mem-efficient work keep them."""
    attention_path = os.path.join(
        config.MORPHGS_HOME, "src", "extlibs", "generative-models",
        "sgm", "modules", "attention.py",
    )
    if not os.path.isfile(attention_path):
        return f"No sgm attention.py at {attention_path} to check."

    with open(attention_path, encoding="utf-8") as f:
        original = f.read()
    if _SGM_SDPA_ALL_BACKENDS not in original:
        return "sgm attention already pinned to a single SDPA backend."
    if _sdpa_all_backends_work():
        return "sgm's default SDPA backends work on this GPU -- leaving them alone."

    with open(attention_path, "w", encoding="utf-8") as f:
        f.write(original.replace(_SGM_SDPA_ALL_BACKENDS, _SGM_SDPA_MATH_ONLY))
    return (
        "Pinned sgm's attention to PyTorch's math SDPA backend: the flash/mem-efficient "
        "kernels fail on this GPU at SV4D's attention head dim. Slower, but it computes."
    )


def _stage_sv4d_checkpoint(ckpt_path, filename):
    """Make the checkpoint reachable at the exact path SV4D actually loads it from.

    MorphGS's preprocess_src.py chdir's into src/extlibs/generative-models and loads the
    checkpoint by the relative path "checkpoints/<name>.safetensors" hardcoded in
    generative-models' own sampling configs. So wherever the user keeps the actual file
    (ComfyUI's models/sv4d, models/checkpoints, ...), it ALSO has to be reachable at that one
    relative location, or the load dies with a bare "FileNotFoundError: No such file or
    directory: checkpoints/sv4d2.safetensors" deep inside sgm -- confirmed in practice, with
    the file sitting correctly in models/sv4d the whole time.

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
    registers a dedicated models/sv4d folder for this (the same convention
    ComfyUI-SkinTokens's models/skintoken and ComfyUI-HY-Motion1's models/HY-Motion use), and
    also scans models/checkpoints and MorphGS's own generative-models checkout, so a checkpoint
    kept in any of those three shows up here. There is no node that downloads it for you:
    download the file yourself from
      - sv4d / sv4d2_8views: https://huggingface.co/stabilityai/sv4d2.0
      - sp4d: https://huggingface.co/stabilityai/sp4d
    and place it in your ComfyUI models/sv4d folder. Falls back to a plain list of mode names
    when folder_paths can't be listed (e.g. the Comfy Registry's isolated node scanner, which
    has no `folder_paths` module at all).
    """

    @classmethod
    def INPUT_TYPES(cls):
        sv4d_options = list(_SV4D_CHECKPOINTS.keys())
        try:
            import folder_paths

            known_filenames = {fname for _, fname in _SV4D_CHECKPOINTS.values()}
            available = [
                f for f in folder_paths.get_filename_list("morphgs_sv4d_checkpoints")
                if os.path.basename(f) in known_filenames
            ]
            if available:
                sv4d_options = available
        except Exception:
            pass  # No folder_paths available -- fall back to plain mode names.

        video_options = _list_input_files({".mp4", ".mov", ".avi", ".mkv", ".webm"}) or [""]

        return {
            "required": {
                "video_path": (video_options, {}),
                "scene_name": ("STRING", {"default": "my_scene"}),
                "already_masked": ("BOOLEAN", {"default": False}),
                "sv4d_mode": (sv4d_options, {"default": sv4d_options[0]}),
                "fastmode": ("BOOLEAN", {"default": True}),
                "force_reprocess": ("BOOLEAN", {"default": False}),
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

    def run(self, video_path, scene_name, already_masked, sv4d_mode, fastmode, force_reprocess):
        log = []
        video_path = _resolve_input_path(video_path)
        mode, filename = _resolve_sv4d_selection(sv4d_mode)

        import folder_paths

        ckpt_path = folder_paths.get_full_path("morphgs_sv4d_checkpoints", filename)
        if not ckpt_path:
            hf_repo, _ = _SV4D_CHECKPOINTS[mode]
            raise RuntimeError(
                f"SV4D checkpoint '{filename}' not found. Download it from "
                f"https://huggingface.co/{hf_repo} and place it in your ComfyUI "
                f"models/sv4d folder (models/checkpoints also works)."
            )
        log.append(_stage_sv4d_checkpoint(ckpt_path, filename))
        log.append(_disable_xformers_in_sv4d_config(filename))
        log.append(_force_math_sdpa_in_sgm())

        scene_dir = os.path.join(config.MORPHGS_HOME, "demo", "videos", scene_name)
        rgb_path = os.path.join(scene_dir, "rgb.mp4")

        if force_reprocess or not os.path.isfile(rgb_path):
            os.makedirs(scene_dir, exist_ok=True)
            pipeline_src_video = os.path.join(scene_dir, f"_source{os.path.splitext(video_path)[1]}")
            shutil.copy(video_path, pipeline_src_video)

            mask_args = [pipeline_src_video, rgb_path]
            if already_masked:
                mask_args.append("--skip-mask")
            out = run_python(node_script_path("mask_video.py"), mask_args, timeout=1800)
            log.append(out)
        else:
            log.append(f"{rgb_path} already exists, skipping masking step")

        processed_dir = os.path.join(config.MORPHGS_HOME, "demo", "processed_videos", scene_name)
        if force_reprocess or not os.path.isdir(processed_dir):
            args = [rgb_path, "--mode", mode]
            if fastmode:
                args.append("--fastmode")
            out = run_python(
                os.path.join(config.MORPHGS_HOME, "src", "preprocess", "preprocess_src.py"),
                args,
                timeout=3600,
            )
            log.append(out)
        else:
            log.append(f"processed_videos/{scene_name} already exists, skipping preprocess_src.py")

        return (scene_name, "\n".join(log))


class MorphGSTrainAndRender:
    """
    Registers/trains the <scene>_to_<character> experiment and returns the rendered
    output video, both as a file path and as an IMAGE batch for in-graph preview.

    seed controls MorphGS's own training-time randomness (Gaussian initialization, sampling --
    threaded through to main.py as --project.seed, the same config field MorphGS's own configs
    set to 43 by default) and has the standard ComfyUI seed widget next to it
    (fixed/increment/decrement/randomize) so you can get a different training result the usual
    way. One real caveat: MorphGS's own output filenames are keyed by iterations only, not
    seed (rendered_video_<iterations>.mp4) -- so changing just the seed at the same iterations
    does NOT by itself invalidate the on-disk cache below. If you want a fresh run at a new
    seed but the same iterations, turn on force_retrain too.

    force_retrain exists because the check below (skip training if the render already exists)
    is deliberately a real, on-disk check, not ComfyUI's own in-memory result cache -- training
    can take hours, and ComfyUI's own cache doesn't survive a restart, so relying on it alone
    would mean losing hours of finished training the moment ComfyUI restarts. This disk check
    is what actually lets you safely restart ComfyUI (or re-queue the same node while building
    out the rest of the graph) without retraining from scratch. force_retrain=False is the
    normal state; if it looks like every run is retraining anyway, check whether scene_name/
    character_name/iterations actually stayed identical between runs -- any of those changing
    points at a different output path that (correctly) doesn't exist yet.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_name": ("STRING", {"default": ""}),
                "character_name": ("STRING", {"default": ""}),
                "iterations": ("INT", {"default": 5000, "min": 100, "max": 100000, "step": 100}),
                "seed": ("INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "force_retrain": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("video_path", "frames", "log")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True  # Lets this run (and its result be visible) standalone before it's
    # wired to anything downstream -- otherwise ComfyUI's execution graph would prune it out
    # entirely and queuing it alone would do nothing (confirmed via a real API test on an
    # OUTPUT_NODE-less node: "Prompt has no outputs").

    def run(self, scene_name, character_name, iterations, seed, force_retrain):
        log = []
        experiment = f"{scene_name}_to_{character_name}"
        config_path = os.path.join(config.MORPHGS_HOME, "configs", "demo", f"{experiment}.yaml")
        render_path = os.path.join(
            config.MORPHGS_HOME, "output", experiment, "model", "morphgs", "render",
            f"rendered_video_{iterations}.mp4",
        )

        if not os.path.isfile(config_path):
            # An empty file fails yaml.safe_load/DotDict (returns None, not {}), so this must be
            # a valid empty YAML mapping for main.py's merge_configs(base_config, _config).
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                f.write("{}\n")
            log.append(f"Created minimal experiment config at {config_path} (defaults from configs/base.yaml)")

        if force_retrain or not os.path.isfile(render_path):
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
            log.append(f"Rendered output already exists at {render_path}, skipping training")

        if not os.path.isfile(render_path):
            raise RuntimeError(
                f"Expected rendered video at {render_path} but it was not produced. Full log:\n" + "\n".join(log)
            )

        import folder_paths

        output_dir = folder_paths.get_output_directory()
        local_video_path = os.path.join(output_dir, "morphgs", f"{experiment}_{iterations}.mp4")
        os.makedirs(os.path.dirname(local_video_path), exist_ok=True)
        shutil.copy(render_path, local_video_path)
        log.append(f"Copied result to {local_video_path}")

        frames = self._load_video_as_tensor(local_video_path)
        return (local_video_path, frames, "\n".join(log))

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
    been run for this character/scene pair.

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
                "force_reexport": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("mesh_path", "preview_path", "log")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def run(self, scene_name, character_name, iterations, output_format, force_reexport):
        log = []
        experiment = f"{scene_name}_to_{character_name}"
        char_dir = os.path.join(config.MORPHGS_HOME, "demo", "characters", character_name)
        video_dir = os.path.join(config.MORPHGS_HOME, "demo", "videos", scene_name)
        rig_path = os.path.join(char_dir, "rigging", "mesh_ori_rig.txt")
        meta_path = os.path.join(char_dir, "rigging", "conversion_meta.json")
        mesh_obj_path = os.path.join(char_dir, "mesh.obj")
        ckpt_path = os.path.join(
            config.MORPHGS_HOME, "output", experiment, "model", "morphgs", "deform",
            f"iteration_{iterations}.pth",
        )
        render_dir = os.path.join(config.MORPHGS_HOME, "output", experiment, "model", "morphgs", "render")
        pose_npz_path = os.path.join(render_dir, f"pose_sequence_{iterations}.npz")
        exported_path = os.path.join(render_dir, f"animated_mesh_{iterations}.{output_format}")

        for label, path in [
            ("rig", rig_path),
            ("mesh", mesh_obj_path),
            ("deform checkpoint", ckpt_path),
        ]:
            if not os.path.isfile(path):
                raise RuntimeError(
                    f"Required {label} file not found at {path}. Run MorphGS: Preprocess Character "
                    f"and MorphGS: Train & Render for '{experiment}' first."
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

        if force_reexport or not os.path.isfile(exported_path):
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
        else:
            log.append(f"Animated mesh already exists at {exported_path}, skipping (force_reexport=False)")

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
