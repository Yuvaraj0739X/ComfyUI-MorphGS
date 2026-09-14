from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


def _register_model_folders():
    """
    Register the folders MorphGS: Preprocess Video and MorphGS: Train & Render read
    checkpoints from as proper ComfyUI model folders.

    sv4d gets its own dedicated ComfyUI models/sv4d folder, auto-created here at load time --
    the same convention ComfyUI-SkinTokens (models/skintoken, created the same way in that
    package's own __init__.py) and ComfyUI-HY-Motion1 (models/HY-Motion) already use in this
    same ComfyUI install, rather than dropping an unrelated safetensors file into the generic,
    shared models/checkpoints bucket. There is no node that downloads the SV4D/SP4D checkpoint
    for you; download it by hand from Hugging Face and drop it in models/sv4d, and it shows up
    in MorphGS: Preprocess Video's sv4d_mode dropdown with no extra step. models/checkpoints and
    MorphGS's own generative-models checkout are also scanned, so a file kept in either of those
    instead still works.

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
        sv4d_paths = [sv4d_dir, *folder_paths.get_folder_paths("checkpoints")]
        if os.path.isdir(gm_ckpt_dir):
            sv4d_paths.append(gm_ckpt_dir)
        folder_paths.folder_names_and_paths["morphgs_sv4d_checkpoints"] = (sv4d_paths, {".safetensors", ".ckpt"})

        deform_dir = os.path.join(config.MORPHGS_HOME, "output")
        if os.path.isdir(deform_dir):
            folder_paths.folder_names_and_paths["morphgs_deform_checkpoints"] = ([deform_dir], {".pth"})
    except Exception:
        pass  # Discoverability-only; never let this break node-pack loading.


_register_model_folders()
