"""
Configuration for ComfyUI-MorphGS, resolved from environment variables so the same node
package works both against a native Linux MorphGS install (e.g. on a cloud GPU box) and,
for local development on Windows, against a MorphGS install running inside WSL.

Environment variables (all optional, sensible defaults below):
    MORPHGS_BACKEND      "linux" (default) or "wsl"
    MORPHGS_HOME         path to the MorphGS repo root
                          default: /workspace/MorphGS (linux) or /home/MorphGS (wsl)
    MORPHGS_CONDA_ENV    name of the conda/venv environment MorphGS was installed into
                          default: "morphgs"
    MORPHGS_CONDA_BASE   path to the conda installation's activate script
                          default: /opt/conda (linux) or /home/miniconda3 (wsl, best-effort)
    MORPHGS_BLENDER_BIN  path to (or bare name of) the Blender executable
                          default: "blender" (expects it on PATH, same requirement as
                          ComfyUI-SkinTokens's headless Blender server, so one Blender
                          install serves both node packs)
    MORPHGS_WSL_DISTRO   WSL distro name, only used when MORPHGS_BACKEND=wsl
                          default: "Ubuntu-22.04"
"""
import os

BACKEND = os.environ.get("MORPHGS_BACKEND", "linux").lower()
if BACKEND not in ("linux", "wsl"):
    raise ValueError(f"MORPHGS_BACKEND must be 'linux' or 'wsl', got: {BACKEND!r}")

_DEFAULT_HOME = "/workspace/MorphGS" if BACKEND == "linux" else "/home/MorphGS"
_DEFAULT_CONDA_BASE = "/opt/conda" if BACKEND == "linux" else "/home/miniconda3"

MORPHGS_HOME = os.environ.get("MORPHGS_HOME", _DEFAULT_HOME)
CONDA_ENV = os.environ.get("MORPHGS_CONDA_ENV", "morphgs")
CONDA_BASE = os.environ.get("MORPHGS_CONDA_BASE", _DEFAULT_CONDA_BASE)
BLENDER_BIN = os.environ.get("MORPHGS_BLENDER_BIN", "blender")
WSL_DISTRO = os.environ.get("MORPHGS_WSL_DISTRO", "Ubuntu-22.04")

CONDA_ACTIVATE = f"source {CONDA_BASE}/bin/activate {CONDA_ENV}"
