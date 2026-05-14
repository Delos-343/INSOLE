"""
Build script — packages the desktop app into a single distributable
executable using PyInstaller.

Usage:
    python app/build_exe.py
The result lands in `dist/InsoleFootClassification[.exe]`.

Notes
-----
- On Windows it produces a `.exe`. On macOS a `.app` bundle. On Linux a
  single ELF binary.
- We bundle the torch backend BUT NOT the data folder. The first launch
  will look for a checkpoint in ./backend/model/checkpoints/best.pt.
- A warm bootstrap with torch+timm+PySide6 yields a 250–400 MB binary;
  --onefile is slow to start, so we ship in --onedir mode by default.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"
ICON_PNG = ROOT / "app" / "assets" / "icons" / "app.png"

APP_NAME = "InsoleFootClassification"


def main() -> int:
    if shutil.which("pyinstaller") is None:
        print("PyInstaller not found. Run: pip install pyinstaller", file=sys.stderr)
        return 1

    # Clean previous builds.
    for d in (DIST, BUILD):
        if d.exists():
            shutil.rmtree(d)

    cmd = [
        "pyinstaller",
        "--name", APP_NAME,
        "--noconfirm",
        "--clean",
        "--windowed",                      # no console window on Win/macOS
        "--noconsole" if platform.system() == "Windows" else "",
        "--noupx",
        "--onedir",                        # faster startup than --onefile
        "--collect-all", "PySide6",
        "--collect-all", "timm",
        "--collect-submodules", "albumentations",
        "--collect-data", "torch",
        "--collect-data", "torchvision",
        "--hidden-import", "backend.model",
        "--hidden-import", "backend.model.architectures.classifier",
        "--hidden-import", "scipy.special._cdflib",
        "--add-data", f"{ROOT / 'app' / 'assets'}{os.pathsep}app/assets",
        "--add-data", f"{ROOT / 'backend' / 'model' / 'checkpoints'}{os.pathsep}backend/model/checkpoints",
        str(ROOT / "app" / "main.py"),
    ]
    if ICON_PNG.exists():
        cmd[1:1] = ["--icon", str(ICON_PNG)]
    cmd = [c for c in cmd if c]  # drop empties

    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        return proc.returncode

    print()
    print(f"✓ Build complete.")
    print(f"  Output:  {DIST / APP_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
