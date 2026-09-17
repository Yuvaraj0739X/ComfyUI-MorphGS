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


def run_python(script_path: str, args: list, timeout: int = None, env: dict = None) -> str:
    """
    Run a MorphGS script with ComfyUI's own Python interpreter, from MORPHGS_HOME. Raises
    RuntimeError with the full captured stdout+stderr on any non-zero exit -- errors are
    surfaced verbatim, never swallowed, so failures in the underlying MorphGS/SV4D pipeline
    are visible directly in the ComfyUI node error rather than silently producing a wrong
    result.

    XFORMERS_DISABLED=1 is set by default (the caller's own env dict can still override it).
    xFormers' fused attention kernels -- used by both DINOv2's feature extractor and SV4D's own
    diffusion attention blocks -- don't have compiled kernels for newer GPU architectures.
    Confirmed in practice on an RTX 5090: "requires device with capability <= (9, 0) but your
    GPU has capability (12, 0) (too new)". DINOv2 specifically checks this exact env var to
    skip xFormers and fall back to plain PyTorch attention instead, which works on any
    hardware (just slower) -- set proactively here so the same failure doesn't have to be
    hit again at the next pipeline stage that happens to use xFormers internally.
    """
    full_env = os.environ.copy()
    full_env.setdefault("XFORMERS_DISABLED", "1")
    if env:
        full_env.update(env)
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
