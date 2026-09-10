"""
Process-invocation helpers for running MorphGS pipeline steps.

MorphGS needs its own pinned Python environment (specific torch/CUDA build, compiled
extensions) that is expected to differ from ComfyUI's own environment, so every node in
this package shells out to that environment as a subprocess rather than importing MorphGS
in-process. On a native Linux deployment (the primary target -- e.g. a cloud GPU box) this
is a plain bash subprocess; MORPHGS_BACKEND=wsl additionally supports developing/testing this
node package on Windows against a MorphGS install running inside WSL.
"""
import os
import subprocess

from . import config


def _wrap_bash(script: str) -> list:
    full_script = f"set -e\n{config.CONDA_ACTIVATE}\ncd {config.MORPHGS_HOME}\n{script}"
    if config.BACKEND == "wsl":
        return ["wsl.exe", "-d", config.WSL_DISTRO, "--", "bash", "-c", full_script]
    return ["bash", "-c", full_script]


def run_bash(script: str, timeout: int = None) -> str:
    """
    Run a bash script in the MorphGS environment. Raises RuntimeError with the full
    captured stdout+stderr on any non-zero exit -- errors are surfaced verbatim, never
    swallowed, so failures in the underlying MorphGS/SV4D/Blender pipeline are visible
    directly in the ComfyUI node error rather than silently producing a wrong result.
    """
    proc = subprocess.run(
        _wrap_bash(script),
        capture_output=True, encoding="utf-8", errors="replace", timeout=timeout,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"MorphGS pipeline step failed (exit {proc.returncode}):\n{script}\n\n--- output ---\n{output}")
    return output


def run_blender_script(script_path: str, args: list, timeout: int = None) -> str:
    args_str = " ".join(f"'{a}'" for a in args)
    cmd = f"{config.BLENDER_BIN} --background --python '{script_path}' -- {args_str}"
    proc = subprocess.run(
        _wrap_bash(cmd),
        capture_output=True, encoding="utf-8", errors="replace", timeout=timeout,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"Blender step failed (exit {proc.returncode}):\n{cmd}\n\n--- output ---\n{output}")
    return output


def to_pipeline_path(local_path: str) -> str:
    """
    Translate a path from ComfyUI's own filesystem view into the path the MorphGS
    environment will see it at. On a native Linux deployment both processes share one
    filesystem, so this is a no-op; MORPHGS_BACKEND=wsl additionally maps Windows paths
    (both plain drive paths and \\\\wsl$\\<distro>\\... paths) onto their WSL equivalents.
    """
    if config.BACKEND != "wsl":
        return local_path

    path = local_path
    normalized = path.replace("/", "\\")
    if path.startswith("/"):
        return path
    if normalized.startswith("\\\\wsl$\\") or normalized.startswith("\\\\wsl.localhost\\"):
        prefix = "\\\\wsl$\\" if normalized.startswith("\\\\wsl$\\") else "\\\\wsl.localhost\\"
        _, _, rest = normalized[len(prefix):].partition("\\")
        return "/" + rest.replace("\\", "/")

    win_path = os.path.abspath(path)
    drive, rest = os.path.splitdrive(win_path)
    return f"/mnt/{drive.rstrip(':').lower()}{rest.replace(chr(92), '/')}"


def copy_local_file_into_pipeline(local_src_path: str, pipeline_dest_path: str):
    """Copy a file from ComfyUI's filesystem into the MorphGS environment's filesystem."""
    if config.BACKEND != "wsl":
        os.makedirs(os.path.dirname(pipeline_dest_path), exist_ok=True)
        import shutil
        shutil.copy(local_src_path, pipeline_dest_path)
        return
    src = to_pipeline_path(local_src_path)
    run_bash(f"mkdir -p \"$(dirname '{pipeline_dest_path}')\" && cp '{src}' '{pipeline_dest_path}'")


def copy_pipeline_file_to_local(pipeline_src_path: str, local_dest_path: str):
    """Copy a file from the MorphGS environment's filesystem into ComfyUI's filesystem."""
    if config.BACKEND != "wsl":
        os.makedirs(os.path.dirname(local_dest_path), exist_ok=True)
        import shutil
        shutil.copy(pipeline_src_path, local_dest_path)
        return
    os.makedirs(os.path.dirname(local_dest_path), exist_ok=True)
    dest = to_pipeline_path(local_dest_path)
    run_bash(f"mkdir -p \"$(dirname '{dest}')\" && cp '{pipeline_src_path}' '{dest}'")


def node_script_path(filename: str) -> str:
    """Resolve a script bundled inside this node package to the path the pipeline environment sees."""
    local_path = os.path.join(os.path.dirname(__file__), "scripts", filename)
    return to_pipeline_path(local_path)
