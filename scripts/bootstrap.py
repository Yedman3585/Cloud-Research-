"""Recreate the environment using Python 3.10+ and a JDK (tested with 17)."""
import json
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.version_info < (3, 10):
        raise SystemExit("Use Python 3.10 or newer (tested with Python 3.12).")
    lock = json.loads((ROOT / "configs/ifogsim.lock.json").read_text())
    destination = ROOT / "vendor/ifogsim"
    if not destination.exists():
        destination.parent.mkdir(exist_ok=True)
        subprocess.run(["git", "clone", lock["repository"], str(destination)], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", lock["commit"]], check=True)
    else:
        actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
        if actual != lock["commit"]:
            raise SystemExit("Existing checkout differs from lock; refusing to replace it automatically.")
    environment = ROOT / ".venv"
    if not environment.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    # Windows venvs put executables in Scripts\, POSIX venvs in bin/.
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "install", "--no-cache-dir", "-e", str(ROOT)], check=True)
    subprocess.run([str(python), str(ROOT / "scripts/ifogsim.py"), "build"], check=True)
    subprocess.run([str(python), "-m", "fogids.cli", "doctor"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
