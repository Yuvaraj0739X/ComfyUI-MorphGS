import glob
import os
import shutil

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


class MorphGSPreprocessCharacter:
    """
    Prepares a rigged character for MorphGS. Accepts either:
      - a rigged .fbx (e.g. Mixamo) or .glb (e.g. SkinTokens/TokenRig output) -- anything
        Blender can import with an armature + skinned mesh -- auto-converted to mesh.obj
        + a RigNet-format rig, or
      - an already-prepared folder containing mesh.obj + rigging/mesh_ori_rig.txt
    then runs MorphGS's own preprocess_tgt.py (360-view render + feature extraction).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_source_path": ("STRING", {"default": "", "multiline": False}),
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
            raise RuntimeError(
                f"Character not ready after conversion: expected mesh.obj + rigging/mesh_ori_rig.txt "
                f"under {char_dir}."
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

        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
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
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_name": ("STRING", {"default": ""}),
                "character_name": ("STRING", {"default": ""}),
                "iterations": ("INT", {"default": 5000, "min": 100, "max": 100000, "step": 100}),
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

    def run(self, scene_name, character_name, iterations, force_retrain):
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
                ["--config", f"demo/{experiment}.yaml", f"--model.opt.iterations={iterations}"],
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
    shows the result directly in ComfyUI's native interactive 3D viewer widget (the same one
    SaveGLB/Preview3D use) as soon as it finishes, with no separate downstream node needed.
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

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("mesh_path", "log")
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

        ui = {"3d": [{"filename": saved_filename, "subfolder": subfolder, "type": "output"}]}
        return {"ui": ui, "result": (local_mesh_path, "\n".join(log))}


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
