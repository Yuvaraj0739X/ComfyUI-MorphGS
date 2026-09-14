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
your ComfyUI models/checkpoints folder, exactly the same way as any other checkpoint --
MorphGS: Preprocess Video's sv4d_mode dropdown reads from that folder directly.
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


def existing_torch_is_cu118():
    try:
        import torch
    except ImportError:
        return False
    version = getattr(torch, "__version__", "")
    cuda_version = getattr(torch.version, "cuda", None)
    log(f"Existing torch: {version} (CUDA build: {cuda_version})")
    return cuda_version is not None and cuda_version.startswith("11.8")


def ensure_torch():
    if existing_torch_is_cu118():
        log("Existing torch is already a CUDA 11.8 build -- leaving it as-is, no reinstall needed.")
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


def ensure_pytorch3d():
    try:
        import pytorch3d  # noqa: F401
        log(f"pytorch3d already installed ({pytorch3d.__version__}), skipping.")
        return
    except ImportError:
        pass
    pip_install("git+https://github.com/facebookresearch/pytorch3d.git")


def ensure_morphgs_requirements():
    requirements_path = os.path.join(MORPHGS_SRC, "requirements.txt")
    if os.path.isfile(requirements_path):
        pip_install("-r", requirements_path)
    # Enforced regardless of what requirements.txt itself pins: numpy 2.x is a known, already-
    # encountered break for MorphGS's compiled extensions (torch.from_numpy/pytorch3d silently
    # broke under numpy 2.1 during earlier work on this project).
    pip_install("numpy<2")


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
        pip_install("-e", ext_dir, "--no-build-isolation")


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
    pip_install("-r", os.path.join(GENERATIVE_MODELS_DIR, "requirements", "pt2.txt"))
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
    ensure_gsplat()
    ensure_cuda_extensions()
    ensure_generative_models()
    verify()
    log(
        "Done. To use MorphGS: Preprocess Video, download an SV4D/SP4D checkpoint from Hugging "
        "Face (stabilityai/sv4d2.0 or stabilityai/sp4d) and place it in your ComfyUI "
        "models/checkpoints folder, the same way as any other checkpoint."
    )


if __name__ == "__main__":
    main()
