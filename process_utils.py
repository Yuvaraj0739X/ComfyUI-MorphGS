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
import threading

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


def _cuda_toolkit_include_dirs(root: str) -> list:
    """Header directories belonging to a toolkit root, in search order.

    CUDA 12 keeps them in include/; CUDA 13 moved them under targets/<arch>/include (with
    include/ usually, but not always, a symlink to it) -- look in both rather than assuming.
    """
    dirs = []
    targets = os.path.join(root, "targets")
    candidates = [os.path.join(root, "include")]
    if os.path.isdir(targets):
        candidates += [os.path.join(targets, a, "include") for a in sorted(os.listdir(targets))]
    for candidate in candidates:
        if os.path.isdir(candidate) and candidate not in dirs:
            dirs.append(candidate)
    return dirs


def _cuda_toolkit_root():
    """Root of the CUDA toolkit whose nvcc will actually compile any runtime CUDA extension.

    A root is accepted on the strength of bin/nvcc alone, not just a findable cuda.h. That
    matters because torch's cpp_extension falls back to /usr/local/cuda and compiles with its
    nvcc whether or not we recognise it -- so failing to recognise a toolkit here doesn't avoid
    using it, it only means _apply_cuda_env would wrap a *different* CUDA's headers around it.
    """
    def is_toolkit(root):
        if not root:
            return False
        if os.path.isfile(os.path.join(root, "bin", "nvcc")):
            return True
        return any(os.path.isfile(os.path.join(d, "cuda.h"))
                   for d in _cuda_toolkit_include_dirs(root))

    for var in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(var)
        if is_toolkit(root):
            return root
    nvcc = shutil.which("nvcc")
    candidates = []
    if nvcc:
        candidates.append(os.path.dirname(os.path.dirname(os.path.realpath(nvcc))))
    candidates.append("/usr/local/cuda")
    for root in candidates:
        if is_toolkit(root):
            return root
    return None


def _prepend_paths(env: dict, variables, dirs) -> None:
    """Put `dirs` at the front of each path-list environment variable, keeping what's there."""
    if not dirs:
        return
    for var in variables:
        env[var] = os.pathsep.join([*dirs, env.get(var, "")]).rstrip(os.pathsep)


def _apply_cuda_env(env: dict) -> None:
    """Let libraries that JIT-compile CUDA at runtime find the toolkit headers and libs.

    On the normal install path nothing compiles at runtime any more: install.py installs
    gsplat as a prebuilt wheel that ships its kernels (gsplat/csrc.so), and gsplat only falls
    back to JIT-compiling its rasterizer when that module is absent. This environment exists
    for that fallback -- a torch/CUDA pair the prebuilt index does not cover, where install.py
    built gsplat from the PyPI sdist and the kernels are compiled on first use inside this
    subprocess. (pykeops used to JIT the same way from MorphGS's ParametricModel; its ARAP
    nearest neighbours are plain torch now.)

    When a real toolkit exists we deliberately DO NOT touch CPATH. CPATH is searched like -I,
    i.e. *before* the `-isystem $CUDA_HOME/include` that torch's own build passes, so adding the
    CUDA headers bundled in pip's nvidia-* wheels there lets them outrank the toolkit's own --
    and those wheels track torch's build (CUDA 13.0 for torch 2.10.0+cu130), not the installed
    toolkit. Mixing the two is not a missing header, it's two CUDA versions in one translation
    unit, and CCCL detects it: "CUDA compiler and CUDA toolkit headers are incompatible, please
    check your include paths", which killed a gsplat_cuda build 32 minutes into a training run.
    nvcc finds its own headers unaided, so contributing nothing here is both correct and safe.

    The pip wheel include dirs are still used when there is no toolkit at all: then they're the
    only CUDA headers on the machine, there is no second version to conflict with, and they must
    all be listed because cuda.h and nvrtc.h can live in separate wheels -- no single CUDA_PATH
    covers them, but the compiler searches every CPATH entry.

    Library directories are additive in both cases: shared libraries resolve by soname, so an
    extra search path can only help a build find libnvrtc, never silently change a version.
    """
    pip_include_dirs, pip_lib_dirs = _nvidia_pip_cuda_dirs()
    root = _cuda_toolkit_root()
    if root:
        # Assigned, not setdefault: _cuda_toolkit_root already returns CUDA_HOME/CUDA_PATH
        # when either names a real toolkit, so this only overwrites a value that pointed
        # somewhere unusable -- which torch would otherwise take at face value.
        env["CUDA_PATH"] = root
        env["CUDA_HOME"] = root
        toolkit_libs = [d for d in (os.path.join(root, "lib64"), os.path.join(root, "lib"))
                        if os.path.isdir(d)]
        _prepend_paths(env, ("LIBRARY_PATH", "LD_LIBRARY_PATH"), toolkit_libs + pip_lib_dirs)
        # Not adding the pip wheel headers isn't enough on its own: an inherited CPATH that
        # already lists them shadows the toolkit just the same, whoever exported it. Drop
        # exactly those entries -- nothing else -- so "nothing we hand the compiler can
        # outrank nvcc's own headers" holds regardless of the environment we started from.
        inherited = [d for d in (env.get("CPATH") or "").split(os.pathsep) if d]
        kept = [d for d in inherited if d not in set(pip_include_dirs)]
        if kept != inherited:
            env["CPATH"] = os.pathsep.join(kept)
            if not kept:
                env.pop("CPATH")
        return
    _prepend_paths(env, ("CPATH",), pip_include_dirs)
    _prepend_paths(env, ("LIBRARY_PATH", "LD_LIBRARY_PATH"), pip_lib_dirs)


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


def _cuda_env_summary(env: dict) -> str:
    """One line naming the CUDA build environment a failed step actually ran with.

    Runtime CUDA compilation (gsplat's rasterizer) fails on *which* headers the compiler found, not on the pipeline arguments -- so when a step dies, the header search
    paths are the first thing worth seeing rather than something to reconstruct afterwards.
    """
    keys = ("CUDA_HOME", "CPATH", "LD_LIBRARY_PATH")
    return "CUDA build env: " + ", ".join(f"{k}={env.get(k) or '<unset>'}" for k in keys)


def run_python(script_path: str, args: list, timeout: int = None, env: dict = None) -> str:
    """
    Run a MorphGS script with ComfyUI's own Python interpreter, from MORPHGS_HOME. Raises
    RuntimeError with the full captured stdout+stderr on any non-zero exit -- errors are
    surfaced verbatim, never swallowed, so failures in the underlying MorphGS/SV4D pipeline
    are visible directly in the ComfyUI node error rather than silently producing a wrong
    result.
    """
    full_env = subprocess_env(env)
    proc = subprocess.Popen(
        [sys.executable, "-u", script_path, *[str(a) for a in args]],
        cwd=config.MORPHGS_HOME,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        encoding="utf-8", errors="replace",
        env=full_env,
    )
    lines = []

    def forward_output():
        for line in proc.stdout:
            lines.append(line)
            print(line, end="", flush=True)

    reader = threading.Thread(target=forward_output, daemon=True)
    reader.start()
    try:
        proc.wait(timeout=timeout)
    except BaseException:
        proc.kill()
        proc.wait()
        reader.join(timeout=5)
        if not reader.is_alive():
            proc.stdout.close()
        raise
    reader.join()
    proc.stdout.close()
    output = "".join(lines)
    if proc.returncode != 0:
        raise RuntimeError(
            f"MorphGS pipeline step failed (exit {proc.returncode}): "
            f"{script_path} {' '.join(str(a) for a in args)}\n"
            f"{_cuda_env_summary(full_env)}\n\n--- output ---\n{output}"
        )
    return output


def run_blender_script(script_path: str, args: list, timeout: int = None) -> str:
    blender = config.find_blender()
    if blender is None or (not os.path.isfile(blender) and shutil.which(blender) is None):
        raise RuntimeError(
            "Blender not found. MorphGS: Preprocess Character and MorphGS: Export Animated Mesh "
            "run Blender 4.2+ headlessly (--background). Install it and either put `blender` "
            "on PATH or set MORPHGS_BLENDER_BIN to the executable "
            "(e.g. C:/Program Files/Blender Foundation/Blender 4.5/blender.exe on "
            "Windows, /usr/bin/blender after `apt-get install blender` on Linux)."
        )
    proc = subprocess.run(
        [blender, "--background", "--python", script_path, "--", *[str(a) for a in args]],
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
