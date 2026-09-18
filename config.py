"""
Configuration for ComfyUI-MorphGS.

MorphGS's dependencies are installed into ComfyUI's own Python environment (requirements.txt
plus install.py, run by ComfyUI Manager), and MorphGS's source ships bundled in this package's
own morphgs_src/ directory -- there is no separate environment to point at.

Environment variables (all optional, sensible defaults below):
    MORPHGS_HOME         path to the MorphGS source root
                          default: <this package's directory>/morphgs_src -- override only for
                          an advanced/manual setup pointing at a MorphGS checkout that lives
                          somewhere else.
    MORPHGS_BLENDER_BIN  path to (or bare name of) the Blender executable
                          default: `blender` on PATH, else the newest Blender found in the
                          usual per-OS install locations (see find_blender). Blender is not a
                          MorphGS dependency -- it's only used by this package's own mesh/rig
                          conversion and export scripts, always in --background mode.

Only the standard library is imported here: this module is loaded by the Comfy Registry's
isolated node scanner, which has neither ComfyUI's modules nor this package's dependencies.
"""
import glob
import os
import shutil
import sys

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_HOME = os.path.join(_PACKAGE_DIR, "morphgs_src")

MORPHGS_HOME = os.environ.get("MORPHGS_HOME", _DEFAULT_HOME)


def _blender_candidates():
    """Where a plain Blender install lands on each OS when nobody added it to PATH.

    Windows' installer and macOS' .app bundle never touch PATH, and neither does unpacking the
    Linux tarball into /opt or the home directory -- only distro packages and snap do."""
    home = os.path.expanduser("~")
    if sys.platform.startswith("win"):
        roots = [os.environ.get("ProgramFiles", r"C:\Program Files"),
                 os.environ.get("ProgramW6432", r"C:\Program Files"),
                 os.environ.get("LOCALAPPDATA", os.path.join(home, "AppData", "Local"))]
        patterns = [os.path.join(r, "Blender Foundation", "Blender*", "blender.exe") for r in roots]
        patterns += [os.path.join(r, "Blender*", "blender.exe") for r in roots]
        # Steam and Microsoft Store installs.
        patterns.append(r"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe")
        patterns.append(os.path.join(home, "AppData", "Local", "Microsoft", "WindowsApps", "blender.exe"))
    elif sys.platform == "darwin":
        patterns = ["/Applications/Blender*.app/Contents/MacOS/Blender",
                    os.path.join(home, "Applications", "Blender*.app", "Contents", "MacOS", "Blender")]
    else:
        patterns = ["/usr/bin/blender", "/usr/local/bin/blender", "/snap/bin/blender",
                    "/opt/blender*/blender", "/opt/blender*/*/blender",
                    os.path.join(home, "blender*", "blender"),
                    os.path.join(home, "Applications", "blender*", "blender"),
                    "/workspace/blender*/blender", "/usr/local/blender*/blender"]
    return patterns


def find_blender():
    """Absolute path of the Blender executable to use, or None if none can be found.

    Order: MORPHGS_BLENDER_BIN (a path, or a name looked up on PATH), then `blender` on PATH,
    then the newest-looking install in the usual per-OS locations. Version ordering falls out
    of sorting the paths ("Blender 4.2" < "Blender 4.5"), which is good enough for picking
    the newest of several side-by-side installs."""
    configured = os.environ.get("MORPHGS_BLENDER_BIN")
    if configured:
        if os.path.isfile(configured):
            return configured
        return shutil.which(configured) or configured
    on_path = shutil.which("blender")
    if on_path:
        return on_path
    found = []
    for pattern in _blender_candidates():
        found.extend(p for p in glob.glob(pattern) if os.path.isfile(p))
    if not found:
        return None
    return sorted(found, key=lambda p: (len(p), p))[-1] if len(found) == 1 else sorted(found)[-1]


BLENDER_BIN = find_blender() or "blender"
