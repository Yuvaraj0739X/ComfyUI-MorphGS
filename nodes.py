import os

# numpy/torch/folder_paths are deliberately NOT imported at module scope: folder_paths only
# exists inside a running ComfyUI process, so importing it here breaks the Comfy Registry's
# isolated node scanner (which inspects this file without a full ComfyUI installation
# alongside it) with ModuleNotFoundError, even though it works fine once ComfyUI itself
# loads this node. They're imported lazily inside the one method that actually needs them.

from . import config
from .process_utils import (
    copy_local_file_into_pipeline,
    copy_pipeline_file_to_local,
    node_script_path,
    run_bash,
    run_blender_script,
    to_pipeline_path,
)

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

    def run(self, character_source_path, character_name, target_height, force_reprocess):
        log = []
        char_dir = f"{config.MORPHGS_HOME}/demo/characters/{character_name}"

        ext = os.path.splitext(character_source_path)[1].lower()
        is_mesh_file = ext in (".fbx", ".glb", ".gltf")

        if is_mesh_file:
            pipeline_src_path = f"{char_dir}/_source{ext}"
            copy_local_file_into_pipeline(character_source_path, pipeline_src_path)
            log.append(f"Copied {ext} source into pipeline env: {pipeline_src_path}")

            mesh_exists = "EXISTS" in run_bash(f"[ -f '{char_dir}/mesh.obj' ] && echo EXISTS || echo MISSING")
            if force_reprocess or not mesh_exists:
                out = run_blender_script(
                    node_script_path("mesh_to_morphgs.py"),
                    [pipeline_src_path, char_dir, str(target_height)],
                    timeout=300,
                )
                log.append(out)
            else:
                log.append("mesh.obj already exists, skipping mesh conversion (force_reprocess=False)")
        else:
            # Treat character_source_path as a pre-prepared folder (mesh.obj + rigging/mesh_ori_rig.txt).
            src_dir = to_pipeline_path(character_source_path)
            run_bash(f"mkdir -p '{char_dir}' && cp -r '{src_dir}/.' '{char_dir}/'")
            log.append(f"Copied prepared character folder into {char_dir}")

        rig_check = run_bash(
            f"[ -f '{char_dir}/mesh.obj' ] && [ -f '{char_dir}/rigging/mesh_ori_rig.txt' ] "
            f"&& echo OK || echo MISSING"
        )
        if "OK" not in rig_check:
            raise RuntimeError(
                f"Character not ready after conversion: expected mesh.obj + rigging/mesh_ori_rig.txt "
                f"under {char_dir}. Got: {rig_check}"
            )

        feat_ready = "EXISTS" in run_bash(
            f"[ -d '{char_dir}/feature' ] && [ \"$(ls -A '{char_dir}/feature' 2>/dev/null)\" ] "
            f"&& echo EXISTS || echo MISSING"
        )
        if force_reprocess or not feat_ready:
            out = run_bash(f"python src/preprocess/preprocess_tgt.py {char_dir}", timeout=1800)
            log.append(out)
        else:
            log.append("Rendered views + features already exist, skipping preprocess_tgt.py")

        return (character_name, "\n".join(log))


class MorphGSPreprocessVideo:
    """
    Prepares a source video for MorphGS: segments + composites onto a white square
    background if needed, then runs SV4D/SP4D multi-view synthesis + feature extraction.
    Requires the SV4D/SP4D extlibs and the relevant checkpoint to already be present in the
    MorphGS environment -- if they aren't, the underlying error surfaces verbatim here
    rather than being caught or hidden.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "scene_name": ("STRING", {"default": "my_scene"}),
                "already_masked": ("BOOLEAN", {"default": False}),
                "sv4d_mode": (["sv4d", "sp4d", "sv4d2_8views"], {"default": "sv4d"}),
                "fastmode": ("BOOLEAN", {"default": True}),
                "force_reprocess": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("scene_name", "log")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, video_path, scene_name, already_masked, sv4d_mode, fastmode, force_reprocess):
        log = []
        scene_dir = f"{config.MORPHGS_HOME}/demo/videos/{scene_name}"
        rgb_path = f"{scene_dir}/rgb.mp4"

        rgb_exists = "EXISTS" in run_bash(f"[ -f '{rgb_path}' ] && echo EXISTS || echo MISSING")
        if force_reprocess or not rgb_exists:
            pipeline_src_video = f"{scene_dir}/_source{os.path.splitext(video_path)[1]}"
            copy_local_file_into_pipeline(video_path, pipeline_src_video)

            mask_cmd = f"python '{node_script_path('mask_video.py')}' '{pipeline_src_video}' '{rgb_path}'"
            if already_masked:
                mask_cmd += " --skip-mask"
            out = run_bash(mask_cmd, timeout=1800)
            log.append(out)
        else:
            log.append(f"{rgb_path} already exists, skipping masking step")

        processed_exists = "EXISTS" in run_bash(
            f"[ -d '{config.MORPHGS_HOME}/demo/processed_videos/{scene_name}' ] && echo EXISTS || echo MISSING"
        )
        if force_reprocess or not processed_exists:
            cmd = f"python src/preprocess/preprocess_src.py {rgb_path} --mode {sv4d_mode}"
            if fastmode:
                cmd += " --fastmode"
            out = run_bash(cmd, timeout=3600)
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

    def run(self, scene_name, character_name, iterations, force_retrain):
        log = []
        experiment = f"{scene_name}_to_{character_name}"
        config_path = f"{config.MORPHGS_HOME}/configs/demo/{experiment}.yaml"
        render_path = (
            f"{config.MORPHGS_HOME}/output/{experiment}/model/morphgs/render/rendered_video_{iterations}.mp4"
        )

        config_exists = "EXISTS" in run_bash(f"[ -f '{config_path}' ] && echo EXISTS || echo MISSING")
        if not config_exists:
            # An empty file fails yaml.safe_load/DotDict (returns None, not {}), so this must be
            # a valid empty YAML mapping for main.py's merge_configs(base_config, _config).
            run_bash(f"mkdir -p '{config.MORPHGS_HOME}/configs/demo' && printf '{{}}\\n' > '{config_path}'")
            log.append(f"Created minimal experiment config at {config_path} (defaults from configs/base.yaml)")

        render_exists = "EXISTS" in run_bash(f"[ -f '{render_path}' ] && echo EXISTS || echo MISSING")
        if force_retrain or not render_exists:
            out = run_bash(
                f"python src/main.py --config demo/{experiment}.yaml --model.opt.iterations={iterations}",
                timeout=None,
            )
            log.append(out)
        else:
            log.append(f"Rendered output already exists at {render_path}, skipping training")

        final_check = run_bash(f"[ -f '{render_path}' ] && echo EXISTS || echo MISSING")
        if "EXISTS" not in final_check:
            raise RuntimeError(
                f"Expected rendered video at {render_path} but it was not produced. Full log:\n" + "\n".join(log)
            )

        import folder_paths

        output_dir = folder_paths.get_output_directory()
        local_video_path = os.path.join(output_dir, "morphgs", f"{experiment}_{iterations}.mp4")
        copy_pipeline_file_to_local(render_path, local_video_path)
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


_SV4D_CHECKPOINTS = {
    "sv4d": ("stabilityai/sv4d2.0", "sv4d2.safetensors"),
    "sv4d2_8views": ("stabilityai/sv4d2.0", "sv4d2_8views.safetensors"),
    "sp4d": ("stabilityai/sp4d", "sp4d.safetensors"),
}


class MorphGSSetupSV4D:
    """
    One-time environment setup for the SV4D/SP4D multi-view synthesis step used by
    MorphGS: Preprocess Video. Not needed for DINOv2 target-character features (those
    download automatically via torch.hub on first use) or for SkinTokens (handled by
    ComfyUI-SkinTokens's own node).

    Clones Stability AI's generative-models repo (sp4d branch, which carries both the
    SP4D and SV4D2.0 code paths) into the MorphGS environment, installs its Python
    dependencies, and downloads the checkpoint for the requested mode. All downloads are
    anonymous HTTPS -- neither the repo clone nor the Hugging Face checkpoints require any
    login or token.

    generative-models' own requirements/pt2.txt pins numpy==2.1, which silently breaks
    torch.from_numpy/pytorch3d in the MorphGS environment (the same regression class
    encountered earlier from an unrelated pip install in this environment) -- this node
    re-pins numpy<2 immediately after installing SV4D's requirements and verifies both
    torch.from_numpy and pytorch3d still import cleanly before reporting success, rather
    than silently leaving the environment in a broken state.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sv4d_mode": (["sv4d", "sp4d", "sv4d2_8views"], {"default": "sv4d"}),
                "force_reinstall": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, sv4d_mode, force_reinstall):
        log = []
        gm_dir = f"{config.MORPHGS_HOME}/src/extlibs/generative-models"

        repo_exists = "EXISTS" in run_bash(f"[ -d '{gm_dir}/.git' ] && echo EXISTS || echo MISSING")
        if force_reinstall or not repo_exists:
            out = run_bash(
                f"rm -rf '{gm_dir}' && "
                f"git clone --branch sp4d --depth 1 https://github.com/Stability-AI/generative-models.git '{gm_dir}'",
                timeout=600,
            )
            log.append(out)

            install_out = run_bash(
                f"cd '{gm_dir}' && "
                f"pip install -r requirements/pt2.txt && "
                f"pip install -e . && "
                f"pip install -e 'git+https://github.com/Stability-AI/datapipelines.git@main#egg=sdata'",
                timeout=1800,
            )
            log.append(install_out)

            # generative-models' requirements/pt2.txt pins numpy==2.1, which breaks
            # torch.from_numpy/pytorch3d in this environment -- re-pin and verify before
            # declaring success.
            fix_out = run_bash(
                'pip install "numpy<2" && '
                "python -c \"import torch, numpy as np; assert torch.from_numpy(np.zeros(3)) is not None\" && "
                'python -c "import pytorch3d; from pytorch3d.renderer import look_at_view_transform" && '
                'echo NUMPY_FIX_VERIFIED',
                timeout=300,
            )
            log.append(fix_out)
            if "NUMPY_FIX_VERIFIED" not in fix_out:
                raise RuntimeError(
                    "SV4D dependency install completed but the post-install numpy/torch/pytorch3d "
                    "verification did not pass -- environment may be left in a broken state. "
                    f"Full log:\n" + "\n".join(log)
                )
        else:
            log.append(f"{gm_dir} already exists, skipping clone/install (force_reinstall=False)")

        hf_repo, filename = _SV4D_CHECKPOINTS[sv4d_mode]
        ckpt_dir = f"{gm_dir}/checkpoints"
        ckpt_path = f"{ckpt_dir}/{filename}"
        ckpt_exists = "EXISTS" in run_bash(f"[ -f '{ckpt_path}' ] && echo EXISTS || echo MISSING")
        if force_reinstall or not ckpt_exists:
            url = f"https://huggingface.co/{hf_repo}/resolve/main/{filename}"
            out = run_bash(
                f"mkdir -p '{ckpt_dir}' && curl -L -o '{ckpt_path}' '{url}'",
                timeout=None,
            )
            log.append(out)
        else:
            log.append(f"{ckpt_path} already exists, skipping download")

        return ("\n".join(log),)


NODE_CLASS_MAPPINGS = {
    "MorphGSPreprocessCharacter": MorphGSPreprocessCharacter,
    "MorphGSPreprocessVideo": MorphGSPreprocessVideo,
    "MorphGSTrainAndRender": MorphGSTrainAndRender,
    "MorphGSSetupSV4D": MorphGSSetupSV4D,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MorphGSPreprocessCharacter": "MorphGS: Preprocess Character",
    "MorphGSPreprocessVideo": "MorphGS: Preprocess Video",
    "MorphGSTrainAndRender": "MorphGS: Train & Render",
    "MorphGSSetupSV4D": "MorphGS: Setup SV4D",
}
