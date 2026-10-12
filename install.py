"""Install GLBFaceRetexture's private Python 3.12 CPU worker, never Comfy dependencies."""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request

from worker_runtime import (
    MODELS,
    installer_command,
    private_venv_directory,
    probe_python,
    remove_private_venv,
    verify_worker,
    worker_environment,
)


PACKAGE_DIR = Path(__file__).resolve().parent


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(target: Path, url: str, expected_sha256: str) -> None:
    """Keep verified files; replace others atomically only after a verified download."""
    if target.is_file() and file_sha256(target) == expected_sha256:
        print(f"Verified {target.name}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "wb") as destination:
            with urllib.request.urlopen(url, timeout=60) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    destination.write(chunk)
                    digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise RuntimeError(f"Model checksum mismatch for {target.name}: expected {expected_sha256}, received {actual}. Existing model was not replaced.")
        temporary.replace(target)
        print(f"Downloaded and verified {target.name}")
    finally:
        temporary.unlink(missing_ok=True)


def executable_path(value: str, description: str) -> str:
    executable = shutil.which(os.path.expanduser(value))
    if executable is None:
        raise RuntimeError(f"{description} executable was not found: {value}")
    return str(Path(executable).resolve())


def select_python(explicit: str | None) -> list[str] | None:
    """Use an explicit 3.12 first, then a working local interpreter or Windows launcher."""
    if explicit:
        command = [executable_path(explicit, "Python 3.12")]
        if probe_python(command).get("version") != [3, 12]:
            raise RuntimeError(f"--python/PYTHON312 must select Python 3.12, not {explicit}.")
        return command
    candidates = []
    if sys.version_info[:2] == (3, 12):
        candidates.append([sys.executable])
    python312 = shutil.which("python3.12")
    if python312:
        candidates.append([str(Path(python312).resolve())])
    if os.name == "nt":
        launcher = shutil.which("py")
        if launcher:
            candidates.append([str(Path(launcher).resolve()), "-3.12"])
    for command in candidates:
        try:
            if probe_python(command).get("version") == [3, 12]:
                return command
        except (OSError, subprocess.CalledProcessError, ValueError):
            continue
    return None


def run(command: list[str]) -> None:
    subprocess.run(command, env=worker_environment(), check=True, cwd=PACKAGE_DIR)


def install_worker(python_option: str | None, uv_option: str | None, recreate: bool = False) -> None:
    uv = executable_path(uv_option, "uv") if uv_option else shutil.which("uv")
    if uv:
        uv = str(Path(uv).resolve())
    # Reject even an explicitly requested wrong-version Python before touching packages.
    python = select_python(python_option)
    if recreate:
        if not python and not uv:
            raise RuntimeError("Refusing to remove .venv: neither Python 3.12 nor uv is available to rebuild it.")
        print("Removing the private worker .venv for a clean rebuild.")
        remove_private_venv(PACKAGE_DIR)
    venv = private_venv_directory(PACKAGE_DIR)
    if venv.exists():
        try:
            verify_worker(PACKAGE_DIR)
        except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
            raise RuntimeError(f"The existing private worker {venv} is unusable ({exc}). "
                               f"Rebuild it with: {installer_command(PACKAGE_DIR, recreate=True)}") from exc
    else:
        if python:
            run([*python, "-I", "-m", "venv", str(venv)])
        elif uv:
            run([uv, "--no-config", "venv", "--python", "3.12", str(venv)])
        else:
            raise RuntimeError("Python 3.12 or uv is required. Install Python 3.12 and run python install.py --python /path/to/python3.12, or install uv and run python install.py --uv /path/to/uv.")
    private_python = verify_worker(PACKAGE_DIR)
    requirements = str(PACKAGE_DIR / "requirements-worker.txt")
    if uv:
        run([uv, "--no-config", "pip", "install", "--python", str(private_python), "-r", requirements])
    else:
        run([str(private_python), "-I", "-m", "pip", "--require-virtualenv", "--disable-pip-version-check", "install", "-r", requirements])
    # MediaPipe eagerly imports optional AudioRecord. This vision-only worker must
    # not initialize PortAudio/PulseAudio on headless hosts. Never touch Comfy's pip.
    private_python = verify_worker(PACKAGE_DIR)
    if uv:
        run([uv, "--no-config", "pip", "uninstall", "--python", str(private_python), "sounddevice"])
    else:
        run([str(private_python), "-I", "-m", "pip", "--require-virtualenv", "--disable-pip-version-check", "uninstall", "-y", "sounddevice"])
    for name, url, expected in MODELS:
        download_model(PACKAGE_DIR / "models" / name, url, expected)
    print(f"GLBFaceRetexture CPU worker installed in {venv}. ComfyUI dependencies were not changed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", dest="python_option", default=os.environ.get("PYTHON312"), help="Existing Python 3.12 executable (also PYTHON312).")
    parser.add_argument("--uv", dest="uv_option", default=os.environ.get("UV_BIN"), help="uv executable (also UV_BIN, otherwise discovered on PATH).")
    parser.add_argument("--recreate", action="store_true", help="Delete and rebuild this package's private .venv (nothing else is removed).")
    arguments = parser.parse_args()
    try:
        install_worker(arguments.python_option, arguments.uv_option, arguments.recreate)
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"GLBFaceRetexture installation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
