from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


def _register_model_folders():
    """
    Register MorphGS's own checkpoint directories -- the SV4D checkpoint MorphGS: Setup SV4D
    downloads, and the per-experiment trained deform-network checkpoints MorphGS: Train &
    Render produces -- as proper ComfyUI model folders. This is now the single, authoritative
    location for these files (MorphGS's source ships bundled in this same package, so there's
    no separate pipeline environment to bridge to), and it's what MorphGS: Preprocess Video's
    sv4d_mode dropdown reads from.

    Skipped silently if folder_paths isn't importable (the Comfy Registry's isolated node
    scanner, which has no such module) or if MORPHGS_HOME doesn't exist yet (install.py hasn't
    run yet) -- both normal situations, not errors.
    """
    try:
        import os

        import folder_paths

        from . import config

        categories = {
            "morphgs_sv4d_checkpoints": (
                os.path.join(config.MORPHGS_HOME, "src", "extlibs", "generative-models", "checkpoints"),
                {".safetensors", ".ckpt"},
            ),
            "morphgs_deform_checkpoints": (
                os.path.join(config.MORPHGS_HOME, "output"),
                {".pth"},
            ),
        }
        for category, (path, extensions) in categories.items():
            if os.path.isdir(path):
                folder_paths.add_model_folder_path(category, path)
                folder_paths.folder_names_and_paths[category] = ([path], extensions)
    except Exception:
        pass  # Discoverability-only; never let this break node-pack loading.


_register_model_folders()
