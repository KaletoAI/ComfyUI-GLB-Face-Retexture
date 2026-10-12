"""Stdlib-only paths and safety checks shared by the node, worker, and installer."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time


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


# Package-index, certificate, proxy and cache settings stay usable for mirrors
# and corporate networks. Install-target overrides (PIP_TARGET, PIP_PREFIX,
# PIP_USER, UV_SYSTEM_PYTHON, UV_PROJECT_ENVIRONMENT, ...) are still dropped.
SOURCE_SETTINGS = frozenset({
    "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_TRUSTED_HOST", "PIP_CERT", "PIP_CLIENT_CERT",
    "PIP_PROXY", "PIP_TIMEOUT", "PIP_RETRIES", "PIP_CACHE_DIR",
    "UV_INDEX_URL", "UV_EXTRA_INDEX_URL", "UV_DEFAULT_INDEX", "UV_INDEX", "UV_INDEX_STRATEGY",
    "UV_INSECURE_HOST", "UV_NATIVE_TLS", "UV_SYSTEM_CERTS", "UV_CACHE_DIR", "UV_HTTP_TIMEOUT",
    "UV_PYTHON_INSTALL_DIR", "UV_PYTHON_INSTALL_MIRROR", "UV_OFFLINE",
})


def installer_command(package_dir: Path, recreate: bool = False) -> str:
    command = ["python", str(package_dir / "install.py"), *(["--recreate"] if recreate else [])]
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def remove_private_venv(package_dir: Path) -> None:
    """Delete only this package's .venv; a symlink is unlinked, never followed."""
    venv = package_dir.resolve() / ".venv"
    if venv.is_symlink() or venv.is_file():
        venv.unlink()
    elif venv.exists():
        shutil.rmtree(venv)


def private_venv_directory(package_dir: Path) -> Path:
    package_dir = package_dir.resolve()
    venv = package_dir / ".venv"
    if venv.resolve() != venv:
        raise RuntimeError("Refusing a worker .venv that points outside this package; "
                           "remove that link or run install.py --recreate.")
    return venv


def worker_python(package_dir: Path) -> Path:
    """Return only the private platform interpreter; never fall back to Comfy Python."""
    existing = package_dir.resolve() / ".venv"
    # Re-running the plain installer cannot fix an existing broken .venv.
    repair = installer_command(package_dir, recreate=existing.exists() or existing.is_symlink())
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
        if upper in SOURCE_SETTINGS:
            environment[name] = value
            continue
        if upper.startswith(("PYTHON", "PIP_", "UV_")) or upper in {"VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "__PYVENV_LAUNCHER__"}:
            continue
        environment[name] = value
    # Disable config-file pip targets; PATH, proxies, SSL_CERT_FILE and the
    # SOURCE_SETTINGS above stay usable.
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


def run_worker(command: list[str], *, cwd: Path, env: dict, log_dir: Path,
               timeout: float | None = None, check=None, poll: float = 0.25) -> tuple[int, str, str]:
    """Run the worker with logs on disk; kill it on timeout or when `check` raises.

    Logs go to files instead of pipes so a chatty worker can never block, and
    are decoded as UTF-8 with replacement so a stray byte never hides an error.
    """
    stdout_path, stderr_path = log_dir / "worker.stdout.log", log_dir / "worker.stderr.log"
    with open(stdout_path, "wb") as stdout, open(stderr_path, "wb") as stderr:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            while True:
                try:
                    returncode = process.wait(timeout=poll)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if check is not None:
                    check()
                if deadline is not None and time.monotonic() > deadline:
                    raise subprocess.TimeoutExpired(command, timeout)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    read = lambda path: path.read_text(encoding="utf-8", errors="replace")
    return returncode, read(stdout_path), read(stderr_path)
