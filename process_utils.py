"""
Process-invocation helpers for running MorphGS pipeline steps.

MorphGS's dependencies are installed directly into ComfyUI's own Python environment by
install.py (see that file's docstring for why, and the tradeoffs), so its scripts run via
`sys.executable` in the same environment ComfyUI itself runs in -- no separate environment,
no shell-quoting of an activation command, no cross-filesystem path translation. Blender is
invoked as a subprocess too (it's a standalone application, not a Python-importable module,
and was never part of the separate-environment problem to begin with), using a plain argv
list rather than a shell string, so paths containing spaces or quotes need no manual escaping.
"""
import os
import subprocess
import sys

from . import config


def subprocess_env(env: dict = None) -> dict:
    """Build the environment for a MorphGS subprocess: ComfyUI's own, minus anything that only
    makes sense inside ComfyUI's process, plus our defaults and the caller's overrides.

    XFORMERS_DISABLED=1 is set by default (the caller's own env dict can still override it).
    xFormers' fused attention kernels -- used by both DINOv2's feature extractor and SV4D's own
    diffusion attention blocks -- don't have compiled kernels for newer GPU architectures.
    Confirmed in practice on an RTX 5090: "requires device with capability <= (9, 0) but your
    GPU has capability (12, 0) (too new)". DINOv2 specifically checks this exact env var to
    skip xFormers and fall back to plain PyTorch attention instead, which works on any
    hardware (just slower) -- set proactively here so the same failure doesn't have to be
    hit again at the next pipeline stage that happens to use xFormers internally.

    LD_PRELOAD is dropped. Managed ComfyUI images (Vast.ai's among them) start ComfyUI with
    comfy-aimdo preloaded, which installs CUDA *driver-level function hooks* to implement its
    DynamicVRAM feature -- "cuda-funchooks.c: hooks successfully installed" in the startup log.
    A child process inherits that preload but none of aimdo's per-process setup, so it ends up
    running with hooked CUDA entry points that were never initialised for it. Our subprocesses
    manage their own VRAM and gain nothing from aimdo, so they run without the preload entirely.
    """
    full_env = os.environ.copy()
    full_env.setdefault("XFORMERS_DISABLED", "1")
    full_env.pop("LD_PRELOAD", None)
    if env:
        full_env.update(env)
    return full_env


def run_python(script_path: str, args: list, timeout: int = None, env: dict = None) -> str:
    """
    Run a MorphGS script with ComfyUI's own Python interpreter, from MORPHGS_HOME. Raises
    RuntimeError with the full captured stdout+stderr on any non-zero exit -- errors are
    surfaced verbatim, never swallowed, so failures in the underlying MorphGS/SV4D pipeline
    are visible directly in the ComfyUI node error rather than silently producing a wrong
    result.
    """
    full_env = subprocess_env(env)
    proc = subprocess.run(
        [sys.executable, script_path, *[str(a) for a in args]],
        cwd=config.MORPHGS_HOME,
        capture_output=True, encoding="utf-8", errors="replace", timeout=timeout,
        env=full_env,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(
            f"MorphGS pipeline step failed (exit {proc.returncode}): "
            f"{script_path} {' '.join(str(a) for a in args)}\n\n--- output ---\n{output}"
        )
    return output


def run_blender_script(script_path: str, args: list, timeout: int = None) -> str:
    proc = subprocess.run(
        [config.BLENDER_BIN, "--background", "--python", script_path, "--", *[str(a) for a in args]],
        capture_output=True, encoding="utf-8", errors="replace", timeout=timeout,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(
            f"Blender step failed (exit {proc.returncode}): {script_path} "
            f"{' '.join(str(a) for a in args)}\n\n--- output ---\n{output}"
        )
    return output


def node_script_path(filename: str) -> str:
    """Resolve a script bundled inside this node package's scripts/ directory."""
    return os.path.join(os.path.dirname(__file__), "scripts", filename)
