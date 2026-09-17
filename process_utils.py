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
import shutil
import subprocess
import sys

from . import config


def _nvidia_pip_cuda_dirs():
    """include/ and lib/ directories contributed by pip-installed nvidia-* wheels.

    Mirrors install.py's helper of the same name (kept separate because install.py runs as a
    standalone script and can't import from this package). `nvidia` is a PEP 420 namespace
    package, so it has no meaningful __file__ -- only __path__, one entry per contributing
    wheel."""
    try:
        import nvidia
    except ImportError:
        return [], []
    include_dirs, lib_dirs = [], []
    for base in list(getattr(nvidia, "__path__", []) or []):
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            for kind, collected in (("include", include_dirs), ("lib", lib_dirs)):
                candidate = os.path.join(base, name, kind)
                if os.path.isdir(candidate):
                    collected.append(candidate)
    return include_dirs, lib_dirs


def _cuda_toolkit_root():
    """A directory whose include/cuda.h exists, or None."""
    for var in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(var)
        if root and os.path.isfile(os.path.join(root, "include", "cuda.h")):
            return root
    nvcc = shutil.which("nvcc")
    candidates = []
    if nvcc:
        candidates.append(os.path.dirname(os.path.dirname(os.path.realpath(nvcc))))
    candidates.append("/usr/local/cuda")
    for root in candidates:
        if os.path.isfile(os.path.join(root, "include", "cuda.h")):
            return root
    return None


def _apply_cuda_env(env: dict) -> None:
    """Let libraries that JIT-compile CUDA at runtime find the toolkit headers and libs.

    MorphGS's ParametricModel imports pykeops, which compiles its own CUDA kernels the first
    time it runs -- inside this subprocess, long after install.py finished. Without this it
    fails with "fatal error: cuda.h: No such file or directory" and "CUDA include path not
    found. Please set the CUDA_PATH or CUDA_HOME environment variable", then silently falls
    back or breaks later.

    CPATH matters as much as CUDA_PATH here: on a modern torch install the CUDA headers come
    from several separate pip wheels (cuda.h and nvrtc.h can live in different directories), so
    there may be no single root that contains them all -- but the compiler searches every
    CPATH entry, so listing them all works where one CUDA_PATH cannot."""
    include_dirs, lib_dirs = _nvidia_pip_cuda_dirs()
    root = _cuda_toolkit_root()
    if root:
        env.setdefault("CUDA_PATH", root)
        env.setdefault("CUDA_HOME", root)
        include_dirs = [os.path.join(root, "include"), *include_dirs]
        lib_dirs = [os.path.join(root, "lib64"), *lib_dirs]
    if include_dirs:
        env["CPATH"] = os.pathsep.join(
            [*include_dirs, env.get("CPATH", "")]
        ).rstrip(os.pathsep)
    if lib_dirs:
        for var in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
            env[var] = os.pathsep.join([*lib_dirs, env.get(var, "")]).rstrip(os.pathsep)


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
    _apply_cuda_env(full_env)
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
