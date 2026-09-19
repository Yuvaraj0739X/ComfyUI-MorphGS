from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


def _register_model_folders():
    """
    Register the folders MorphGS: Preprocess Video and MorphGS: Train & Render read
    checkpoints from as proper ComfyUI model folders.

    ComfyUI's standard models/diffusion_models folder is scanned first. A dedicated models/sv4d
    fallback is auto-created here too, matching the per-package folder convention used by
    ComfyUI-SkinTokens and ComfyUI-HY-Motion1. models/checkpoints and MorphGS's own
    generative-models checkout are additional fallbacks. There is no node that downloads the
    SV4D/SP4D checkpoint; download it from Hugging Face into models/diffusion_models and it
    appears in MorphGS: Preprocess Video's sv4d_mode dropdown.

    morphgs_deform_checkpoints points at MorphGS's own per-experiment output directory, where
    MorphGS: Train & Render writes each trained deform-network checkpoint.

    Skipped silently if folder_paths isn't importable (the Comfy Registry's isolated node
    scanner, which has no such module) -- a normal situation, not an error.
    """
    try:
        import os

        import folder_paths

        from . import config

        sv4d_dir = os.path.join(folder_paths.models_dir, "sv4d")
        if not os.path.isdir(sv4d_dir):
            os.makedirs(sv4d_dir)

        gm_ckpt_dir = os.path.join(config.MORPHGS_HOME, "src", "extlibs", "generative-models", "checkpoints")
        # diffusion_models first: ComfyUI's own standard home for a single diffusion checkpoint
        # (what UNETLoader reads from) is the natural place to keep an SV4D/SP4D file, ahead of
        # the dedicated sv4d folder and the generic checkpoints bucket.
        sv4d_paths = [
            *folder_paths.get_folder_paths("diffusion_models"),
            sv4d_dir,
            *folder_paths.get_folder_paths("checkpoints"),
        ]
        if os.path.isdir(gm_ckpt_dir):
            sv4d_paths.append(gm_ckpt_dir)
        folder_paths.folder_names_and_paths["morphgs_sv4d_checkpoints"] = (sv4d_paths, {".safetensors", ".ckpt"})

        deform_dir = os.path.join(config.MORPHGS_HOME, "output")
        if os.path.isdir(deform_dir):
            folder_paths.folder_names_and_paths["morphgs_deform_checkpoints"] = ([deform_dir], {".pth"})
    except Exception:
        pass  # Discoverability-only; never let this break node-pack loading.


_register_model_folders()


def _register_input_routes():
    """Refreshable input-file lists for ComfyUI frontends that support remote combo options."""
    try:
        from aiohttp import web
        from server import PromptServer

        from .nodes import _character_input_options, _sv4d_checkpoint_options, _video_input_options

        server = PromptServer.instance
        marker = "_morphgs_input_routes_registered"
        if getattr(server, marker, False):
            return

        @server.routes.get("/morphgs/input/characters")
        async def morphgs_character_inputs(_request):
            return web.json_response(_character_input_options())

        @server.routes.get("/morphgs/input/videos")
        async def morphgs_video_inputs(_request):
            return web.json_response(_video_input_options())

        @server.routes.get("/morphgs/models/sv4d")
        async def morphgs_sv4d_checkpoints(_request):
            return web.json_response(_sv4d_checkpoint_options())

        setattr(server, marker, True)
    except Exception:
        pass  # Comfy Registry scanner and older ComfyUI builds do not expose PromptServer.


_register_input_routes()
