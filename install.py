"""
One-time setup for ComfyUI-MorphGS, run automatically by ComfyUI Manager (and comfy-cli) right
after `requirements.txt` has been installed. Safe to re-run by hand: `python install.py`.

This follows the same install contract every other ComfyUI custom node uses:

  1. Manager clones the repo,
  2. installs `requirements.txt` (every pure-Python dependency of this package lives there),
  3. runs this script.

Nothing here touches torch. ComfyUI owns torch; Manager refuses to install it from a node's
requirements anyway (torch/torchvision/torchaudio are on its pip blacklist), and the Comfy
Registry standards forbid a node from interfering with the shared environment. Whatever torch
build ComfyUI is running on is the one MorphGS runs on.

What this script actually does is the one thing `requirements.txt` cannot express: pick the
two compiled CUDA packages MorphGS needs -- `pytorch3d` (mesh rendering, KNN, chamfer loss)
and `gsplat` (the Gaussian-splat rasterizer) -- in the build that matches the running
Python / torch / CUDA combination. Both come as PREBUILT wheels from the cuda-wheels index
(https://github.com/PozzettiAndrea/cuda-wheels, the same index the comfy-env tooling behind
ComfyUI-TRELLIS2 and ComfyUI-3D-Pack resolves against), which covers Linux and Windows,
CPython 3.10-3.14, torch 2.4-2.13 and CUDA 12.4-13.2, with kernels for every GPU generation
from Turing (sm_75) through Blackwell (sm_120). On the normal path no compiler and no CUDA
toolkit are needed -- exactly like installing any other node.

Only when the running torch/CUDA pair has no wheel in that index does this fall back to
building from source, and only then does it need `nvcc`. That path is reported loudly.

Stability AI's `generative-models` (the SV4D/SP4D code behind MorphGS: Preprocess Video) is
cloned into the bundled MorphGS tree because MorphGS imports its `scripts/` helpers from a
checkout, not from an installed package. There is deliberately no node that downloads the
SV4D/SP4D checkpoint: download it from Hugging Face (stabilityai/sv4d2.0 or stabilityai/sp4d)
into ComfyUI's models/sv4d folder, the same way every other checkpoint is handled. The example
workflow declares those files in its node metadata, so ComfyUI's own missing-models dialog
offers the download link when the workflow is loaded.

Blender is a separate system binary (used only by this package's own rig conversion and export
scripts, never by MorphGS itself) and is not installed here. Put `blender` 4.2+ on PATH or set
MORPHGS_BLENDER_BIN.
"""
import os
import shutil
import subprocess
import sys

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
MORPHGS_SRC = os.path.join(PACKAGE_DIR, "morphgs_src")
REQUIREMENTS = os.path.join(PACKAGE_DIR, "requirements.txt")

# PEP 503 index of prebuilt CUDA wheels. Wheels are tagged with a local version such as
# `1.5.3+cu130torch2.10`, so an exact `==` pin selects the build for one torch/CUDA pair.
CUDA_WHEEL_INDEX = "https://pozzettiandrea.github.io/cuda-wheels/v2/"

# Versions this package's vendored MorphGS source is written against. gsplat's rasterization
# API and pytorch3d's renderer are both used directly by morphgs_src; bump deliberately.
PYTORCH3D_VERSION = "0.7.9"
GSPLAT_VERSION = "1.5.3"

GENERATIVE_MODELS_DIR = os.path.join(MORPHGS_SRC, "src", "extlibs", "generative-models")


def log(msg):
    print(f"[ComfyUI-MorphGS install] {msg}", flush=True)


def run(cmd, **kwargs):
    log("$ " + " ".join(cmd))
    # Pin cwd to this package's own directory: pip calls os.getcwd() on startup and crashes
    # with a bare FileNotFoundError if the caller's cwd has vanished underneath it (seen when
    # Manager rewrites this directory while install.py runs from a shell inside it).
    kwargs.setdefault("cwd", PACKAGE_DIR)
    subprocess.run(cmd, check=True, **kwargs)


def pip_install(*args, **kwargs):
    run([sys.executable, "-m", "pip", "install", *args], **kwargs)


def _python_subprocess(code, env=None):
    """(returncode, combined output) of running `code` in a fresh interpreter.

    A broken native package can kill the interpreter outright rather than raise (seen in
    practice: "Intel oneMKL FATAL ERROR: Cannot load libtorch_cpu.so"), so every probe in this
    script runs out-of-process where that cannot take the installer down with it."""
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=PACKAGE_DIR, env=env,
        capture_output=True, encoding="utf-8", errors="replace",
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def installed_version(package):
    """Installed version string from package metadata, or None. Never imports the package."""
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return None


def _torch_release(raw):
    """(major, minor) from "2.10.0+cu130" -> (2, 10). Tuples, never strings: "2.10" < "2.3"
    lexically."""
    if not raw:
        return None
    parts = raw.split("+", 1)[0].split(".")
    try:
        return (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return None


def _cuda_release(raw):
    """(major, minor) from torch.version.cuda: "13.0" -> (13, 0). None for CPU/ROCm builds."""
    if not raw:
        return None
    try:
        major, minor = raw.split(".")[:2]
        return (int(major), int(minor))
    except ValueError:
        return None


def probe_torch():
    """(torch_version, cuda_version) as reported by the torch ComfyUI is actually running on.

    torch is a hard prerequisite owned by ComfyUI itself, so a missing or broken torch is
    reported and left alone -- this script never installs or replaces it."""
    code = (
        "import torch\n"
        "print('TORCH_VERSION=' + torch.__version__)\n"
        "print('CUDA_VERSION=' + str(torch.version.cuda))\n"
    )
    rc, output = _python_subprocess(code)
    if rc != 0:
        raise RuntimeError(
            "torch is not importable in this environment:\n" + output + "\n\n"
            "ComfyUI provides torch; this package never installs it. Fix ComfyUI's own torch "
            "install first (a numpy-major-version mismatch is the usual cause), then re-run "
            "install.py."
        )
    values = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    torch_version = values.get("TORCH_VERSION")
    cuda_version = values.get("CUDA_VERSION")
    if cuda_version in ("None", ""):
        cuda_version = None
    log(f"torch {torch_version}, CUDA build {cuda_version or 'none (CPU/ROCm build)'}")
    return torch_version, cuda_version


def cuda_wheel_tag(torch_version, cuda_version):
    """Local-version tag the cuda-wheels index uses for this torch/CUDA pair, e.g.
    "cu130torch2.10". None when there is no CUDA build to match."""
    release = _torch_release(torch_version)
    cuda = _cuda_release(cuda_version)
    if release is None or cuda is None:
        return None
    return f"cu{cuda[0]}{cuda[1]}torch{release[0]}.{release[1]}"


# ---------------------------------------------------------------------------------------------
# numpy
# ---------------------------------------------------------------------------------------------

# torch builds from 2.3 onward are compiled against the numpy 2.x ABI; older ones need numpy<2.
_NUMPY2_MIN_TORCH = (2, 3)


def ensure_numpy_matching_torch():
    """Keep numpy on the major version the installed torch was built against.

    Both directions have bitten in practice: numpy 2.x under torch 2.0 broke
    torch.from_numpy/pytorch3d, and numpy<2 under torch 2.10 left `import torch` dying with
    "Intel oneMKL FATAL ERROR: Cannot load libtorch_cpu.so". Some transitive requirements
    (generative-models pins numpy==2.1) move numpy as a side effect, so this runs after them."""
    raw = installed_version("torch")
    release = _torch_release(raw)
    if release is not None and release >= _NUMPY2_MIN_TORCH:
        pip_install("numpy>=2")
    else:
        log(f"torch {raw} predates the numpy 2 ABI -- pinning numpy<2 to match it.")
        pip_install("numpy<2")


# ---------------------------------------------------------------------------------------------
# Pure-Python requirements
# ---------------------------------------------------------------------------------------------

def pip_install_requirements_file(path, env=None):
    """Install a requirements file one line at a time and return the lines that failed.

    A bulk `pip install -r` is one transaction: a single unbuildable pin aborts every other
    package in the file. One line at a time, one bad pin costs one package, and the caller can
    say which."""
    failed = []
    if not os.path.isfile(path):
        return failed
    with open(path) as f:
        lines = [line.split("#", 1)[0].strip() for line in f]
    for line in lines:
        if not line or line.startswith("-"):
            continue
        try:
            pip_install(line, env=env)
        except subprocess.CalledProcessError:
            log(f"WARNING: failed to install '{line}' from {os.path.basename(path)} -- continuing.")
            failed.append(line)
    return failed


def ensure_requirements():
    """Manager already installed requirements.txt before running this script; re-running it
    here only matters for a manual `git clone` install, and costs nothing when everything is
    already satisfied."""
    failed = pip_install_requirements_file(REQUIREMENTS)
    if failed:
        log(
            f"WARNING: {len(failed)} package(s) from requirements.txt did not install: "
            f"{', '.join(failed)}. A pipeline step that needs one of them will say so."
        )


def ensure_rembg_backend():
    """rembg (used by this package's own mask_video.py) needs an onnxruntime backend; a bare
    `pip install rembg` succeeds and then fails at first use with "No onnxruntime backend
    found". requirements.txt asks for rembg[cpu]; this is the explicit guarantee."""
    rc, _ = _python_subprocess("import onnxruntime")
    if rc == 0:
        return
    pip_install("rembg[cpu]")


def ensure_setuptools():
    """generative-models' pytorch_lightning dependency still imports the legacy
    `pkg_resources`, which setuptools removed in v81. Pin back only when it is missing."""
    rc, _ = _python_subprocess("import pkg_resources")
    if rc == 0:
        return
    log("pkg_resources is missing (setuptools >= 81 no longer ships it); pinning setuptools<81.")
    pip_install("setuptools<81")
    rc, _ = _python_subprocess("import pkg_resources")
    if rc != 0:
        raise RuntimeError(
            "pkg_resources is still missing after installing setuptools<81. "
            "MorphGS: Preprocess Video cannot import generative-models without it."
        )


# ---------------------------------------------------------------------------------------------
# Compiled CUDA packages: prebuilt wheels first, source build only as a fallback
# ---------------------------------------------------------------------------------------------

def _install_prebuilt(package, version, tag):
    """Install `package==version+tag` from the cuda-wheels index. False when no such wheel
    exists for this Python/torch/CUDA/OS combination."""
    target = f"{version}+{tag}"
    if installed_version(package) == target:
        log(f"{package} {target} already installed.")
        return True
    try:
        pip_install(f"{package}=={target}", "--extra-index-url", CUDA_WHEEL_INDEX)
    except subprocess.CalledProcessError:
        return False
    return installed_version(package) == target


def _import_ok(module):
    rc, _ = _python_subprocess(f"import {module}")
    return rc == 0


def nvidia_pip_cuda_dirs():
    """include/ and lib/ directories contributed by pip-installed nvidia-* wheels. Only relevant
    to the source-build fallback: on a pip-CUDA torch install, e.g. cusparse.h exists only
    inside site-packages/nvidia/. `nvidia` is a PEP 420 namespace package -- __path__ only."""
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


def cuda_toolkit_include_dirs():
    """Header directories of the toolkit whose nvcc will compile a source build, listed AHEAD
    of any pip-wheel headers so cuda.h and the CCCL tree come from the same toolkit as nvcc.
    CUDA 12 keeps them in include/; CUDA 13 moved them under targets/<arch>/include."""
    nvcc = shutil.which("nvcc")
    roots = [r for r in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")) if r]
    if nvcc:
        roots.append(os.path.dirname(os.path.dirname(os.path.realpath(nvcc))))
    roots.append("/usr/local/cuda")
    dirs = []
    for root in roots:
        targets = os.path.join(root, "targets")
        candidates = [os.path.join(root, "include")]
        if os.path.isdir(targets):
            candidates += [os.path.join(targets, a, "include") for a in sorted(os.listdir(targets))]
        for candidate in candidates:
            if os.path.isdir(candidate) and candidate not in dirs:
                dirs.append(candidate)
    return dirs


def _cuda_build_env():
    """Environment for compiling a CUDA extension from source (fallback path only).

    CPATH is searched like -I, i.e. before the `-isystem $CUDA_HOME/include` torch's build
    passes, so the toolkit's headers go first and pip-wheel headers (which track torch's CUDA,
    not the toolkit's) stay behind them. Mixing the two is what CCCL rejects with "CUDA
    compiler and CUDA toolkit headers are incompatible"."""
    pip_include_dirs, lib_dirs = nvidia_pip_cuda_dirs()
    include_dirs = cuda_toolkit_include_dirs() + pip_include_dirs
    env = os.environ.copy()
    if include_dirs:
        env["CPATH"] = os.pathsep.join([*include_dirs, env.get("CPATH", "")]).rstrip(os.pathsep)
    if lib_dirs:
        env["LIBRARY_PATH"] = os.pathsep.join([*lib_dirs, env.get("LIBRARY_PATH", "")]).rstrip(os.pathsep)
        env["LD_LIBRARY_PATH"] = os.pathsep.join([*lib_dirs, env.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    return env


def require_nvcc():
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        for candidate in ("/usr/local/cuda/bin/nvcc",):
            if os.path.isfile(candidate):
                nvcc = candidate
    if nvcc is None:
        raise RuntimeError(
            "No prebuilt pytorch3d/gsplat wheel exists for this torch/CUDA combination, and "
            "building from source needs the CUDA toolkit's nvcc, which is not installed. "
            "Either install a CUDA toolkit matching torch's CUDA version, or run ComfyUI on a "
            f"torch build the prebuilt index covers ({CUDA_WHEEL_INDEX}: torch 2.4-2.13 with "
            "CUDA 12.4-13.2)."
        )
    out = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
    log(f"Source build: using nvcc at {nvcc}\n{out.strip()}")


def _build_pytorch3d_from_source():
    # pytorch3d's setup.py imports torch during the build itself; pip's isolated build env
    # would not contain it, hence --no-build-isolation (pytorch3d's own documented method).
    pip_install(
        "git+https://github.com/facebookresearch/pytorch3d.git", "--no-build-isolation",
        env=_cuda_build_env(),
    )


def _build_gsplat_from_source():
    # The PyPI sdist ships no kernels; gsplat JIT-compiles them on first import of its CUDA
    # backend. Doing that import here turns a mid-training build failure into an install-time
    # one and leaves the compiled extension cached.
    pip_install(f"gsplat=={GSPLAT_VERSION}")
    log("Compiling gsplat's CUDA kernels (this takes several minutes)...")
    rc, output = _python_subprocess("from gsplat.cuda._backend import _C", env=_cuda_build_env())
    if rc != 0:
        raise RuntimeError("gsplat's CUDA kernels failed to build:\n" + output)


def ensure_compiled_packages(torch_version, cuda_version):
    tag = cuda_wheel_tag(torch_version, cuda_version)
    wanted = [("pytorch3d", PYTORCH3D_VERSION, _build_pytorch3d_from_source),
              ("gsplat", GSPLAT_VERSION, _build_gsplat_from_source)]
    needs_source = []

    if tag is None:
        log("torch has no CUDA build here; MorphGS needs CUDA for training and rendering.")
        needs_source = wanted
    else:
        log(f"Resolving prebuilt CUDA wheels tagged +{tag} from {CUDA_WHEEL_INDEX}")
        for package, version, build in wanted:
            if _install_prebuilt(package, version, tag):
                continue
            if _import_ok(package):
                log(
                    f"No prebuilt {package} wheel for +{tag}; keeping the already-installed "
                    f"{package} {installed_version(package)} since it imports."
                )
                continue
            needs_source.append((package, version, build))

    if not needs_source:
        return
    log(
        "Falling back to building from source for: "
        + ", ".join(p for p, _, _ in needs_source)
        + ". This is the slow path and needs a CUDA toolkit."
    )
    require_nvcc()
    for package, _, build in needs_source:
        build()


# ---------------------------------------------------------------------------------------------
# MorphGS source + SV4D code
# ---------------------------------------------------------------------------------------------

def ensure_morphgs_source():
    if not os.path.isdir(MORPHGS_SRC):
        raise RuntimeError(
            f"{MORPHGS_SRC} not found. MorphGS's source ships inside this package; re-install "
            f"ComfyUI-MorphGS (git clone or reinstall via Manager)."
        )
    log(f"Bundled MorphGS source: {MORPHGS_SRC}")


# What SV4D/SP4D inference actually imports from Stability's `sgm` package and its sv4d demo
# helpers (checked against the sp4d branch: sgm.models, sgm.util, sgm.modules.encoders,
# scripts/demo/sv4d_helpers.py, scripts/util/detection). generative-models' own
# requirements/pt2.txt is NOT used: it is a frozen dev-box snapshot that pins torch,
# torchvision, xformers, an ancient transformers==4.19.1, opencv-python==4.6 and numpy==2.1,
# and installing it into a shared ComfyUI environment breaks other nodes. Unpinned on purpose
# (Manager's guidance: never more restrictive than needed).
SGM_RUNTIME_REQUIREMENTS = [
    "pytorch-lightning",
    "open-clip-torch",
    "transformers",
    "kornia",
    "safetensors",
    "fsspec",
    "packaging",
    "clip @ git+https://github.com/openai/CLIP.git",
]


def ensure_generative_models():
    """Stability AI's generative-models, branch sp4d, as a plain checkout inside the bundled
    MorphGS tree. MorphGS's preprocess_src.py puts that directory on sys.path itself and
    imports both `sgm` and the repo's `scripts/demo/sv4d_helpers` from it (the latter is not
    part of any installable package), so the checkout is the install -- nothing is pip
    installed from it, only the runtime dependencies above."""
    if not os.path.isdir(os.path.join(GENERATIVE_MODELS_DIR, ".git")):
        run(["git", "clone", "--branch", "sp4d", "--depth", "1",
             "https://github.com/Stability-AI/generative-models.git", GENERATIVE_MODELS_DIR])
    else:
        log(f"generative-models already present at {GENERATIVE_MODELS_DIR}")
    failed = []
    for spec in SGM_RUNTIME_REQUIREMENTS:
        try:
            pip_install(spec)
        except subprocess.CalledProcessError:
            failed.append(spec)
    if failed:
        log(
            f"WARNING: could not install {', '.join(failed)}. MorphGS: Preprocess Video imports "
            f"these; install them by hand before running it."
        )


# ---------------------------------------------------------------------------------------------
# Repair + verify
# ---------------------------------------------------------------------------------------------

# Packages whose compiled extensions link against numpy's C ABI. If numpy's major version moves
# underneath them they fail at import with "module compiled against ABI version ... but this
# version of numpy is ..." (seen with opencv-python).
_NUMPY_ABI_SENSITIVE = [
    ("cv2", "opencv-python"),
    ("scipy", "scipy"),
    ("skimage", "scikit-image"),
    ("matplotlib", "matplotlib"),
]


def repair_numpy_abi_mismatches():
    for module, package in _NUMPY_ABI_SENSITIVE:
        rc, output = _python_subprocess(f"import {module}")
        if rc == 0 or "numpy" not in output.lower():
            continue
        log(f"{module} fails to import against the installed numpy; reinstalling {package}.")
        try:
            pip_install("--force-reinstall", "--no-cache-dir", package)
        except subprocess.CalledProcessError:
            log(f"WARNING: could not reinstall {package}.")


def verify():
    code = (
        "import numpy, torch, cv2, trimesh, gsplat, pytorch3d\n"
        "from pytorch3d.renderer import look_at_view_transform\n"
        "from gsplat.cuda._backend import _C\n"
        "assert torch.from_numpy(numpy.zeros(3)) is not None\n"
        "print(f'numpy {numpy.__version__}')\n"
        "print(f'torch {torch.__version__} (CUDA {torch.version.cuda}), "
        "cuda available: {torch.cuda.is_available()}')\n"
        "print(f'pytorch3d {pytorch3d.__version__}')\n"
        "print(f'gsplat {gsplat.__version__}')\n"
        "print('CUDA_AVAILABLE=' + str(torch.cuda.is_available()))\n"
    )
    rc, output = _python_subprocess(code)
    log(output)
    if rc != 0:
        raise RuntimeError(
            "Verification failed -- see the output above for which import broke. The nodes "
            "will not run until this is resolved."
        )
    if "CUDA_AVAILABLE=True" not in output:
        log("WARNING: torch.cuda.is_available() is False -- training and rendering need a GPU.")
    if shutil.which(os.environ.get("MORPHGS_BLENDER_BIN", "blender")) is None:
        log(
            "NOTE: no `blender` on PATH. MorphGS: Preprocess Character and Export Animated Mesh "
            "need Blender 4.2+ (set MORPHGS_BLENDER_BIN if it lives elsewhere)."
        )
    log("Verification passed.")


def main():
    log(f"Environment: {sys.executable}")
    ensure_morphgs_source()
    ensure_setuptools()
    torch_version, cuda_version = probe_torch()
    ensure_requirements()
    ensure_rembg_backend()
    ensure_compiled_packages(torch_version, cuda_version)
    ensure_generative_models()
    ensure_numpy_matching_torch()
    repair_numpy_abi_mismatches()
    verify()
    log(
        "Done. For MorphGS: Preprocess Video, download an SV4D/SP4D checkpoint from Hugging "
        "Face (stabilityai/sv4d2.0 or stabilityai/sp4d) into ComfyUI's models/sv4d folder."
    )


if __name__ == "__main__":
    main()
