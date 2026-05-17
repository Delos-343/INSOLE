"""
Build script — packages the desktop app into a distributable executable
using PyInstaller. HARDENED VERSION.

Usage (from the project root, in the activated venv):
    python app/build_exe.py

Result:
    dist/InsoleFootClassification/InsoleFootClassification(.exe)

What this hardened version fixes vs. the original
-------------------------------------------------
1. The original inserted an empty-string argument ("" on non-Windows),
   which PyInstaller rejects as an invalid argument. Arguments are now
   assembled in a list with no empty entries.
2. Adds a post-build self-check: confirms the expected binary exists and
   is non-trivial in size, and prints the exact launch path.
3. Fails loudly with actionable messages instead of a stack trace.
4. Verifies a trained checkpoint exists before building, so you don't
   ship an app that will raise NoTrainedModelError on first launch.

Notes
-----
- --onedir (folder) build: faster startup than --onefile and easier to
  diagnose. The whole dist/InsoleFootClassification/ folder is the
  deliverable; zip it for handoff.
- The data/ folder is intentionally NOT bundled. The app expects a
  checkpoint at backend/model/checkpoints/best.pt at runtime; ship that
  file inside the dist folder (the script copies it for you).
"""

from __future__ import annotations

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
CKPT = ROOT / "backend" / "model" / "checkpoints" / "best.pt"


def _fail(msg: str) -> int:
    print("\n[BUILD FAILED] " + msg + "\n", file=sys.stderr)
    return 1


def main() -> int:
    print("=" * 64)
    print("Insole Foot Classification — executable build")
    print("=" * 64)

    if shutil.which("pyinstaller") is None:
        return _fail("PyInstaller not installed. Run:  pip install pyinstaller")

    if not CKPT.exists():
        print(
            f"[WARN] No trained checkpoint at {CKPT}.\n"
            f"       The built app will run but classification needs a model.\n"
            f"       Make sure best.pt exists before handoff."
        )
    else:
        print(f"[OK] Found checkpoint: {CKPT} ({CKPT.stat().st_size/1e6:.0f} MB)")

    # Clean previous builds.
    for d in (DIST, BUILD):
        if d.exists():
            shutil.rmtree(d)
            print(f"[OK] Cleaned {d}")

    is_windows = platform.system() == "Windows"

    # Assemble args as a list — NO empty strings (the original bug).
    cmd: list[str] = [
        "pyinstaller",
        "--name", APP_NAME,
        "--noconfirm",
        "--clean",
        "--windowed",
        "--noupx",
        "--onedir",
        "--collect-all", "PySide6",
        "--collect-all", "timm",
        "--collect-submodules", "albumentations",
        "--collect-data", "torch",
        "--collect-data", "torchvision",
        "--hidden-import", "backend.model",
        "--hidden-import", "backend.model.inference.predictor",
        "--hidden-import", "backend.model.architectures.classifier",
    ]
    if is_windows and ICON_PNG.with_suffix(".ico").exists():
        cmd += ["--icon", str(ICON_PNG.with_suffix(".ico"))]

    cmd += [str(ROOT / "app" / "main.py")]

    print("\n[RUN] " + " ".join(cmd) + "\n")
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        return _fail(
            f"PyInstaller exited with code {proc.returncode}. "
            f"Scroll up for its error output."
        )

    # ---- Post-build verification ----
    exe_name = APP_NAME + (".exe" if is_windows else "")
    out_dir = DIST / APP_NAME
    exe_path = out_dir / exe_name

    if not exe_path.exists():
        return _fail(
            f"Build reported success but {exe_path} was not produced. "
            f"Check the PyInstaller warnings above."
        )
    size_mb = exe_path.stat().st_size / 1e6
    if size_mb < 0.05:
        return _fail(f"{exe_path} is suspiciously small ({size_mb:.2f} MB).")

    # Ship the checkpoint alongside the binary so first launch works.
    if CKPT.exists():
        target = out_dir / "backend" / "model" / "checkpoints"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CKPT, target / "best.pt")
        print(f"[OK] Bundled checkpoint into {target / 'best.pt'}")

    print("\n" + "=" * 64)
    print("BUILD SUCCESSFUL")
    print("=" * 64)
    print(f"Deliverable folder : {out_dir}")
    print(f"Launch the app     : {exe_path}")
    print("\nNext: double-click the executable, then run Test A")
    print("(P014, arch 4.69 -> expect Normal Foot, 100%, green banner).")
    print("Then zip the whole folder above for handoff.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
