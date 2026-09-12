from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


def _register_model_folders():
    """
    Register MorphGS's own checkpoint directories -- the SV4D checkpoint MorphGS: Setup SV4D
    downloads, and the per-experiment trained deform-network checkpoints MorphGS: Train &
    Render produces -- as proper ComfyUI model folders, purely for discoverability: so they
    show up in ComfyUI's own model folder listings, and so extra_model_paths.yaml can point at
    them the standard way on setups where MorphGS lives on a different drive.

    This does NOT change how any node actually loads these files (they're still read from
    inside the MorphGS environment via the existing subprocess-based pipeline, since that
    environment can be a different machine entirely in a real deployment) -- registration is
    skipped silently whenever the resolved path isn't reachable from wherever this ComfyUI
    process itself runs (e.g. a genuinely remote MorphGS deployment with no local mount, or
    the Comfy Registry's isolated node scanner, which has no `folder_paths` module at all),
    since both are normal, expected situations here, not errors.
    """
    try:
        import os

        import folder_paths

        from . import config
        from .process_utils import to_local_path

        categories = {
            "morphgs_sv4d_checkpoints": (
                f"{config.MORPHGS_HOME}/src/extlibs/generative-models/checkpoints",
                {".safetensors", ".ckpt"},
            ),
            "morphgs_deform_checkpoints": (
                f"{config.MORPHGS_HOME}/output",
                {".pth"},
            ),
        }
        for category, (pipeline_path, extensions) in categories.items():
            local_path = to_local_path(pipeline_path)
            if os.path.isdir(local_path):
                folder_paths.add_model_folder_path(category, local_path)
                folder_paths.folder_names_and_paths[category] = ([local_path], extensions)
    except Exception:
        pass  # Discoverability-only; never let this break node-pack loading.


_register_model_folders()
