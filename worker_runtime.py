"""Stdlib-only paths and safety checks shared by the node, worker, and installer."""

import json
import os
from pathlib import Path
import shlex
import subprocess


MODELS = (
    (
        "face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
    (
        "selfie_multiclass.tflite",
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite",
        "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0",
    ),
)


def installer_command(package_dir: Path) -> str:
    command = ["python", str(package_dir / "install.py")]
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def private_venv_directory(package_dir: Path) -> Path:
    package_dir = package_dir.resolve()
    venv = package_dir / ".venv"
    if venv.resolve() != venv:
        raise RuntimeError("Refusing a worker .venv that points outside this package.")
    return venv


def worker_python(package_dir: Path) -> Path:
    """Return only the private platform interpreter; never fall back to Comfy Python."""
    repair = installer_command(package_dir)
    try:
        venv = private_venv_directory(package_dir)
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.is_file():
            raise RuntimeError(f"Private CPU worker is missing: {python}")
        config = (venv / "pyvenv.cfg").read_text(encoding="utf-8")
        settings = {}
        for line in config.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                settings[key.strip().lower()] = value.strip().lower()
        if settings.get("include-system-site-packages") != "false":
            raise RuntimeError("Worker .venv must exclude system site-packages.")
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"GLBFaceRetexture private worker unavailable: {exc}. Run: {repair}") from exc
    return python


def worker_environment() -> dict[str, str]:
    """Drop inherited Comfy Python paths and package-manager installation overrides."""
    environment = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if upper.startswith(("PYTHON", "PIP_", "UV_")) or upper in {"VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "__PYVENV_LAUNCHER__"}:
            continue
        environment[name] = value
    # Disable even globally configured pip targets; PATH/proxies/certificates stay usable.
    environment["PIP_CONFIG_FILE"] = os.devnull
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def validate_worker_identity(identity: dict, venv: Path) -> None:
    """Refuse wrong-version, system, and unrelated-venv interpreters before any pip work."""
    if identity.get("version") != [3, 12]:
        raise RuntimeError("GLBFaceRetexture CPU worker requires exactly Python 3.12.")
    prefix = Path(identity["prefix"]).resolve()
    if prefix != venv.resolve() or prefix == Path(identity["base_prefix"]).resolve():
        raise RuntimeError("Refusing an interpreter outside this package's private .venv.")


def probe_python(command: list[str]) -> dict:
    """Ask the actual interpreter, ignoring Python startup paths and user site-packages."""
    code = "import json,sys; print(json.dumps({'version':list(sys.version_info[:2]),'prefix':sys.prefix,'base_prefix':sys.base_prefix}))"
    completed = subprocess.run(
        [*command, "-I", "-c", code], env=worker_environment(),
        capture_output=True, text=True, check=True,
    )
    return json.loads(completed.stdout)


def verify_worker(package_dir: Path) -> Path:
    python = worker_python(package_dir)
    validate_worker_identity(probe_python([str(python)]), private_venv_directory(package_dir))
    return python
