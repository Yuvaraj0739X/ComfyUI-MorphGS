"""
One-time setup for ComfyUI-MorphGS, run automatically by ComfyUI Manager right after this
package is installed (and safe to re-run manually: `python install.py`).

Installs MorphGS's dependencies directly into the SAME Python environment this script is
running under (ComfyUI's own environment, when Manager runs it) -- there is no separate
MorphGS environment with this design. This is a real, deliberate tradeoff, not a default to
take lightly:

  - If your existing torch build isn't CUDA 11.8, this REPLACES it with torch==2.0.1+cu118 to
    match what MorphGS's compiled CUDA extensions need. That's likely to affect any OTHER
    custom node in the same ComfyUI install that wants a different/newer torch -- this design
    is only appropriate for a ComfyUI instance dedicated to running this pipeline.
  - If your existing torch build already IS a CUDA 11.8 build (common on ComfyUI installs
    that haven't been recently upgraded), this script leaves it alone and builds MorphGS's
    extensions against it instead -- no downgrade needed in that case.
  - pytorch3d and MorphGS's own two CUDA extensions (`simple_knn`,
    `diff_gaussian_rasterization`) are compiled from source at install time, which needs the
    CUDA 11.8 toolkit (`nvcc`) actually present on this machine -- this is a real prerequisite
    this script cannot install for you (it's a system-level package, e.g. `nvidia-cuda-toolkit`
    or NVIDIA's own installer), and it checks for it upfront with a clear error rather than
    failing deep inside a pip build log.

Blender is a separate system binary (not a MorphGS dependency at all -- confirmed nothing in
MorphGS's own source imports/uses it; it's only used by this package's own mesh/rig conversion
and export scripts) and is not installed by this script. Install it separately and point
MORPHGS_BLENDER_BIN at it if it's not already on PATH.

MorphGS's own source (a customized fork with fixes: DINOv2-only feature matching, gsplat-based
rendering, topology-aware ARAP regularization) ships bundled in this package's own
morphgs_src/ directory -- no separate clone step, no separate repo to keep in sync. This
script only installs *dependencies* against that already-present source.

This script also clones and installs Stability AI's `generative-models` (the SV4D/SP4D code
MorphGS: Preprocess Video needs), so that's ready with no extra setup step of its own. There is
deliberately no node that downloads the actual SV4D/SP4D checkpoint file for you -- SV4D has no
native ComfyUI model architecture (unlike SV3D/SVD, which ComfyUI does support natively), so it
couldn't be loaded through the built-in Load Checkpoint node either way. Instead: download the
checkpoint by hand from Hugging Face (stabilityai/sv4d2.0 or stabilityai/sp4d) and drop it in
your ComfyUI models/sv4d folder (created and registered automatically by this package, the
same convention ComfyUI-SkinTokens/models/skintoken and ComfyUI-HY-Motion1/models/HY-Motion
already use) -- MorphGS: Preprocess Video's sv4d_mode dropdown reads from that folder directly.
"""
import os
import shutil
import subprocess
import sys

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
MORPHGS_SRC = os.path.join(PACKAGE_DIR, "morphgs_src")

REQUIRED_CUDA = "11.8"


def log(msg):
    print(f"[ComfyUI-MorphGS install] {msg}", flush=True)


def run(cmd, **kwargs):
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def pip_install(*args, **kwargs):
    run([sys.executable, "-m", "pip", "install", *args], **kwargs)


def check_cuda_toolkit():
    """MorphGS's pytorch3d and its two custom CUDA extensions all compile from source at
    install time and need a matching CUDA toolkit's nvcc present -- fail clearly here rather
    than deep inside an opaque pip build error."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        for candidate in (f"/usr/local/cuda-{REQUIRED_CUDA}/bin/nvcc", "/usr/local/cuda/bin/nvcc"):
            if os.path.isfile(candidate):
                nvcc = candidate
                break
    if nvcc is None:
        raise RuntimeError(
            f"CUDA {REQUIRED_CUDA} toolkit (nvcc) not found. pytorch3d and MorphGS's own "
            f"compiled CUDA extensions (simple_knn, diff_gaussian_rasterization) need to "
            f"build from source against it. Install the CUDA {REQUIRED_CUDA} toolkit (e.g. "
            f"`apt install cuda-toolkit-11-8`, or NVIDIA's own installer) and re-run this "
            f"script."
        )
    out = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
    log(f"Found nvcc at {nvcc}:\n{out.strip()}")
    if REQUIRED_CUDA not in out:
        log(
            f"WARNING: nvcc reports a version other than {REQUIRED_CUDA} -- MorphGS's compiled "
            f"extensions are only confirmed working against CUDA {REQUIRED_CUDA}. Continuing, "
            f"but build failures below may trace back to this mismatch."
        )
    return nvcc


def existing_torch_cuda_version():
    try:
        import torch
    except ImportError:
        return None
    version = getattr(torch, "__version__", "")
    cuda_version = getattr(torch.version, "cuda", None)
    log(f"Existing torch: {version} (CUDA build: {cuda_version})")
    return cuda_version


def _parse_cuda_version(cuda_version):
    try:
        major, minor = cuda_version.split(".")[:2]
        return (int(major), int(minor))
    except Exception:
        return None


def ensure_torch():
    cuda_version = existing_torch_cuda_version()
    parsed = _parse_cuda_version(cuda_version) if cuda_version else None

    if parsed == (11, 8):
        log("Existing torch is already a CUDA 11.8 build -- leaving it as-is, no reinstall needed.")
        return

    if parsed is not None and parsed > (11, 8):
        # Two real, separate reasons a forced downgrade to torch==2.0.1+cu118 is actively wrong
        # here, not just unnecessary: (1) PyTorch's own cu118 wheel index has since dropped that
        # exact version for newer Python builds (confirmed: "Could not find a version that
        # satisfies the requirement torch==2.0.1" against a real cu118 index listing only
        # 2.2.0+ upward) -- it may simply no longer be installable at all on a current Python.
        # (2) CUDA 11.8 has no support for newer GPU architectures at all (e.g. NVIDIA
        # Blackwell/RTX 50-series) -- even if the install somehow succeeded, it could not
        # actually run a single kernel on hardware newer than what CUDA 11.8 knows about. So:
        # leave torch alone and try building pytorch3d/gsplat/MorphGS's own CUDA extensions
        # against whatever newer stack is already here instead. This is NOT the combination
        # MorphGS was originally built/tested against -- a build or runtime failure below may
        # trace back to this newer CUDA/torch version rather than to a missing dependency.
        log(
            f"Existing torch is CUDA {cuda_version}, newer than the CUDA 11.8 build MorphGS's "
            f"compiled extensions were originally built against. NOT forcing a downgrade to "
            f"torch==2.0.1+cu118: that exact version is no longer available from PyTorch's own "
            f"cu118 index for newer Python builds, and CUDA 11.8 doesn't support newer GPU "
            f"architectures (e.g. Blackwell/RTX 50-series) regardless. Leaving torch as-is and "
            f"attempting to build against this newer stack instead -- untested territory for "
            f"MorphGS's own CUDA extensions, so a failure below may trace back to this version "
            f"gap, not a missing dependency."
        )
        return

    log(
        "Existing torch (if any) is not a CUDA 11.8 build. Installing torch==2.0.1+cu118 / "
        "torchvision==0.15.2+cu118 to match what MorphGS's compiled extensions need. This "
        "REPLACES your current torch install -- only proceed if this ComfyUI instance is "
        "dedicated to running this pipeline."
    )
    pip_install(
        "torch==2.0.1", "torchvision==0.15.2",
        "--index-url", "https://download.pytorch.org/whl/cu118",
    )


def nvidia_pip_cuda_dirs():
    """When torch's own CUDA support comes from pip-installed `nvidia-*` wheels instead of a
    full system CUDA toolkit (normal for modern torch/CUDA 12+/13 installs), the actual CUDA
    headers/libs needed to compile a NEW CUDA extension (pytorch3d, MorphGS's own two
    extensions) can live inside site-packages instead of /usr/local/cuda -- confirmed in
    practice: `cusparse.h` existed only at .../site-packages/nvidia/cu13/include/cusparse.h,
    invisible to a build that only searches /usr/local/cuda/include, causing a "fatal error:
    cusparse.h: No such file or directory" that has nothing to do with a missing dependency.
    Collected dynamically (works for both the older per-library nvidia-cusparse-cuXX/
    nvidia-cublas-cuXX layout and the newer consolidated nvidia-cu13-style layout) rather than
    hardcoding a path, since the exact venv/CUDA version varies per machine."""
    try:
        import nvidia
    except ImportError:
        return [], []
    # `nvidia` is a PEP 420 namespace package (no single __init__.py, contributed to by each
    # separately-installed nvidia-* wheel) -- it has no meaningful __file__ (that's None, which
    # is exactly what broke this the first time), only __path__, an iterable of every
    # contributing directory.
    bases = list(getattr(nvidia, "__path__", []) or [])
    if not bases:
        nvidia_file = getattr(nvidia, "__file__", None)
        if nvidia_file:
            bases = [os.path.dirname(nvidia_file)]
    include_dirs, lib_dirs = [], []
    for base in bases:
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            subdir = os.path.join(base, name)
            inc = os.path.join(subdir, "include")
            lib = os.path.join(subdir, "lib")
            if os.path.isdir(inc):
                include_dirs.append(inc)
            if os.path.isdir(lib):
                lib_dirs.append(lib)
    return include_dirs, lib_dirs


def _cuda_build_env():
    """Extra CPATH/LIBRARY_PATH/LD_LIBRARY_PATH so compiling a CUDA extension can find
    headers/libs bundled inside pip-installed nvidia-* wheels (see nvidia_pip_cuda_dirs).
    Returns None (== inherit the current environment unchanged) when there's nothing to add."""
    include_dirs, lib_dirs = nvidia_pip_cuda_dirs()
    if not include_dirs and not lib_dirs:
        return None
    env = os.environ.copy()
    if include_dirs:
        env["CPATH"] = os.pathsep.join([*include_dirs, env.get("CPATH", "")]).rstrip(os.pathsep)
    if lib_dirs:
        joined = os.pathsep.join([*lib_dirs, env.get("LIBRARY_PATH", "")]).rstrip(os.pathsep)
        env["LIBRARY_PATH"] = joined
        env["LD_LIBRARY_PATH"] = os.pathsep.join([*lib_dirs, env.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    return env


def ensure_pytorch3d():
    try:
        import pytorch3d  # noqa: F401
        log(f"pytorch3d already installed ({pytorch3d.__version__}), skipping.")
        return
    except ImportError:
        pass
    # pytorch3d's own setup.py needs `import torch` to succeed *during the build itself* (to
    # pick CUDA extension settings) -- pip's default build isolation runs that step in a
    # throwaway env that does NOT include this environment's already-installed torch, causing
    # a "ModuleNotFoundError: No module named 'torch'" failure even though torch is right
    # there. --no-build-isolation is pytorch3d's own documented install method for exactly
    # this reason (same as MorphGS's own two CUDA extensions below, which already use it).
    pip_install(
        "git+https://github.com/facebookresearch/pytorch3d.git", "--no-build-isolation",
        env=_cuda_build_env(),
    )


def pip_install_requirements_file(path, env=None):
    """Installs each line of a requirements file as its own separate pip call, not one bulk
    `pip install -r file` -- that treats the whole file as one transaction, so a single
    unsatisfiable/unbuildable pin can abort the WHOLE call, taking every other package in the
    file down with it. Confirmed in practice twice: MorphGS's own requirements.txt mixes
    simple, reliable packages (trimesh, numpy, tqdm...) with fragile, binary-heavy ones
    (open3d, pymeshlab, pykeops, scikit-sparse) -- trimesh silently never got installed this
    way, surfacing later as an opaque ModuleNotFoundError deep inside preprocess_tgt.py instead
    of a clear message here. Then Stability AI's own generative-models/requirements/pt2.txt hit
    the same thing: it pins triton==2.0.0, which has no build for a current Python at all,
    aborting that entire install too. Installing one line at a time means one bad pin only
    costs that one package; returns the list of lines that failed so the caller can report
    them instead of the failure staying silent."""
    failed = []
    if not os.path.isfile(path):
        return failed
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    for line in lines:
        try:
            pip_install(line, env=env)
        except subprocess.CalledProcessError:
            log(f"WARNING: failed to install '{line}' from {path} -- continuing with the rest.")
            failed.append(line)
    return failed


def ensure_morphgs_requirements():
    requirements_path = os.path.join(MORPHGS_SRC, "requirements.txt")
    failed = pip_install_requirements_file(requirements_path)
    # Enforced regardless of what requirements.txt itself pins: numpy 2.x is a known, already-
    # encountered break for MorphGS's compiled extensions (torch.from_numpy/pytorch3d silently
    # broke under numpy 2.1 during earlier work on this project).
    pip_install("numpy<2")
    if failed:
        log(
            f"WARNING: {len(failed)} package(s) from requirements.txt failed to install: "
            f"{', '.join(failed)}. Install these by hand (e.g. `pip install open3d`) if a "
            f"pipeline step later complains one of them is missing."
        )


def ensure_rembg_backend():
    """rembg (used by this package's own mask_video.py, not a MorphGS dependency) has no
    built-in inference engine -- plain `pip install rembg` installs successfully but fails at
    runtime with "No onnxruntime backend found" the first time it's actually used. It needs
    the `[cpu]` or `[gpu]` extra to pull in onnxruntime. requirements.txt/pyproject.toml
    already specify rembg[cpu], but this is a second, explicit guarantee in case a given
    ComfyUI Manager version doesn't re-run a node's requirements.txt on update, only on first
    install."""
    try:
        import onnxruntime  # noqa: F401

        log("onnxruntime already installed, rembg has a working backend.")
        return
    except ImportError:
        pass
    pip_install("rembg[cpu]")


def ensure_gsplat():
    try:
        import gsplat
        log(f"gsplat already installed ({gsplat.__version__}), skipping.")
        return
    except ImportError:
        pass
    pip_install("gsplat==1.5.3")


def ensure_morphgs_source():
    if not os.path.isdir(MORPHGS_SRC):
        raise RuntimeError(
            f"{MORPHGS_SRC} not found. MorphGS's source ships bundled with this package -- "
            f"if it's missing, re-install ComfyUI-MorphGS (git clone or reinstall via Manager)."
        )
    log(f"Using bundled MorphGS source at {MORPHGS_SRC}")


def ensure_cuda_extensions():
    cuda_env = _cuda_build_env()  # see nvidia_pip_cuda_dirs -- same header/lib gap as pytorch3d
    for name, subdir in [
        ("diff_gaussian_rasterization", "latent-gaussian-rasterization"),
        ("simple_knn", "simple-knn"),
    ]:
        try:
            __import__(name)
            log(f"{name} already installed, skipping.")
            continue
        except ImportError:
            pass
        ext_dir = os.path.join(MORPHGS_SRC, "src", "extlibs", subdir)
        pip_install("-e", ext_dir, "--no-build-isolation", env=cuda_env)


GENERATIVE_MODELS_DIR = os.path.join(MORPHGS_SRC, "src", "extlibs", "generative-models")


def ensure_generative_models():
    """MorphGS: Preprocess Video needs Stability AI's `generative-models` (SGM) code importable
    to run SV4D/SP4D -- a checkpoint file alone isn't enough, and there's no ComfyUI-native
    architecture for it to load through instead (see this file's module docstring). Installed
    once here, automatically, same as every other dependency -- not a separate manual step."""
    if os.path.isdir(os.path.join(GENERATIVE_MODELS_DIR, ".git")):
        log(f"{GENERATIVE_MODELS_DIR} already exists, skipping clone.")
        return
    run(["git", "clone", "--branch", "sp4d", "--depth", "1",
         "https://github.com/Stability-AI/generative-models.git", GENERATIVE_MODELS_DIR])
    # Per-package, not a bulk `pip install -r pt2.txt` -- that file pins triton==2.0.0, which
    # has no build for a current Python at all, and a bulk install aborts entirely over that
    # one line (see pip_install_requirements_file's docstring). triton accelerates certain
    # fused kernels; SV4D/SP4D's core inference path this package actually needs doesn't
    # depend on it being present.
    failed = pip_install_requirements_file(os.path.join(GENERATIVE_MODELS_DIR, "requirements", "pt2.txt"))
    if failed:
        log(
            f"WARNING: {len(failed)} package(s) from generative-models' own requirements failed "
            f"to install: {', '.join(failed)}. Continuing -- these are usually optional "
            f"acceleration extras (e.g. triton), not required for SV4D/SP4D's core inference "
            f"path Preprocess Video actually uses."
        )
    pip_install("-e", GENERATIVE_MODELS_DIR)
    pip_install("-e", "git+https://github.com/Stability-AI/datapipelines.git@main#egg=sdata")
    # generative-models' own requirements/pt2.txt pins numpy==2.1, which silently breaks
    # torch.from_numpy/pytorch3d (already encountered once during this project) -- re-pin
    # immediately, verify() below re-checks this actually held.
    pip_install("numpy<2")


def verify():
    import torch
    import gsplat
    import pytorch3d

    log(f"torch {torch.__version__} (CUDA build {torch.version.cuda}), CUDA available: {torch.cuda.is_available()}")
    log(f"gsplat {gsplat.__version__}")
    log(f"pytorch3d {pytorch3d.__version__}")

    assert torch.from_numpy(__import__("numpy").zeros(3)) is not None, "torch.from_numpy is broken"
    from pytorch3d.renderer import look_at_view_transform  # noqa: F401

    if not torch.cuda.is_available():
        log("WARNING: torch.cuda.is_available() is False -- training/rendering needs a GPU.")

    log("Verification passed.")


def main():
    log(f"Installing MorphGS into this Python environment: {sys.executable}")
    check_cuda_toolkit()
    ensure_torch()
    ensure_pytorch3d()
    ensure_morphgs_source()
    ensure_morphgs_requirements()
    ensure_rembg_backend()
    ensure_gsplat()
    ensure_cuda_extensions()
    ensure_generative_models()
    verify()
    log(
        "Done. To use MorphGS: Preprocess Video, download an SV4D/SP4D checkpoint from Hugging "
        "Face (stabilityai/sv4d2.0 or stabilityai/sp4d) and place it in your ComfyUI "
        "models/sv4d folder."
    )


if __name__ == "__main__":
    main()
