from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


def _register_model_folders():
    """
    Register the folders MorphGS: Preprocess Video and MorphGS: Train & Render read
    checkpoints from as proper ComfyUI model folders.

    morphgs_sv4d_checkpoints is deliberately backed by BOTH this ComfyUI install's own
    models/checkpoints folder(s) (folder_paths.get_folder_paths("checkpoints"), which always
    exists) and MorphGS's own generative-models checkout -- there is no node that downloads the
    SV4D/SP4D checkpoint for you; the user downloads it by hand from Hugging Face and drops it
    into models/checkpoints exactly like any other ComfyUI checkpoint, and it shows up in
    MorphGS: Preprocess Video's sv4d_mode dropdown from there with no extra step. The
    generative-models path is also included in case someone already has a checkout with a
    checkpoint staged there from before.

    morphgs_deform_checkpoints points at MorphGS's own per-experiment output directory, where
    MorphGS: Train & Render writes each trained deform-network checkpoint.

    Skipped silently if folder_paths isn't importable (the Comfy Registry's isolated node
    scanner, which has no such module) -- a normal situation, not an error.
    """
    try:
        import os

        import folder_paths

        from . import config

        gm_ckpt_dir = os.path.join(config.MORPHGS_HOME, "src", "extlibs", "generative-models", "checkpoints")
        sv4d_paths = list(folder_paths.get_folder_paths("checkpoints"))
        if os.path.isdir(gm_ckpt_dir):
            sv4d_paths.append(gm_ckpt_dir)
        folder_paths.folder_names_and_paths["morphgs_sv4d_checkpoints"] = (sv4d_paths, {".safetensors", ".ckpt"})

        deform_dir = os.path.join(config.MORPHGS_HOME, "output")
        if os.path.isdir(deform_dir):
            folder_paths.folder_names_and_paths["morphgs_deform_checkpoints"] = ([deform_dir], {".pth"})
    except Exception:
        pass  # Discoverability-only; never let this break node-pack loading.


_register_model_folders()
