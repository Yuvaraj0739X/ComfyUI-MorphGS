"""
Configuration for ComfyUI-MorphGS.

MorphGS's own dependencies (pinned torch/CUDA build, compiled CUDA extensions) are installed
directly into ComfyUI's own Python environment by install.py, and MorphGS's source ships
bundled in this package's own morphgs_src/ directory -- there is no separate environment to
point at, so unlike earlier versions of this package there's no backend/conda/WSL selection here.

Environment variables (all optional, sensible defaults below):
    MORPHGS_HOME         path to the MorphGS source root
                          default: <this package's directory>/morphgs_src -- override only for
                          an advanced/manual setup pointing at a MorphGS checkout that lives
                          somewhere else.
    MORPHGS_BLENDER_BIN  path to (or bare name of) the Blender executable
                          default: "blender" (expects it on PATH, same requirement as
                          ComfyUI-SkinTokens's headless Blender server, so one Blender
                          install serves both node packs). Blender is not a MorphGS
                          dependency -- it's only used by this package's own mesh/rig
                          conversion and export scripts.
"""
import os

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_HOME = os.path.join(_PACKAGE_DIR, "morphgs_src")

MORPHGS_HOME = os.environ.get("MORPHGS_HOME", _DEFAULT_HOME)
BLENDER_BIN = os.environ.get("MORPHGS_BLENDER_BIN", "blender")
