"""Guard private-worker isolation, cross-platform selection, and atomic model integrity."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import install
import worker_runtime


class RuntimeTests(unittest.TestCase):
    def make_environment(self, package):
        venv = package / ".venv"
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        python.parent.mkdir(parents=True)
        python.write_bytes(b"private interpreter fixture")
        (venv / "pyvenv.cfg").write_text("include-system-site-packages = false\n", encoding="utf-8")
        return python

    def test_missing_install_refuses_interpreter_fallback(self):
        """A missing worker must fail with a usable installer rather than run in Comfy."""
        with tempfile.TemporaryDirectory(prefix="face retexture ") as directory:
            with self.assertRaises(RuntimeError):
                worker_runtime.worker_python(Path(directory))

    def test_system_packages_environment_is_refused(self):
        """An inherited site-packages venv defeats isolation and must never run the worker."""
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            self.make_environment(package)
            (package / ".venv/pyvenv.cfg").write_text("include-system-site-packages = true\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                worker_runtime.worker_python(package)

    @unittest.skipIf(os.name == "nt", "Creating symlinks requires privileges on Windows")
    def test_external_venv_is_refused(self):
        """A .venv symlink into Comfy's environment must not become an install target."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "node"
            package.mkdir()
            external_package = root / "comfy"
            self.make_environment(external_package)
            (package / ".venv").symlink_to(external_package / ".venv", target_is_directory=True)
            with self.assertRaises(RuntimeError):
                worker_runtime.worker_python(package)

    def test_actual_interpreter_identity_is_required(self):
        """A mislabeled executable must not permit pip against a different Python prefix."""
        with tempfile.TemporaryDirectory() as directory:
            venv = Path(directory) / ".venv"
            valid = {"version": [3, 12], "prefix": str(venv), "base_prefix": str(Path(directory) / "base")}
            worker_runtime.validate_worker_identity(valid, venv)
            for change in ({"version": [3, 13]}, {"prefix": valid["base_prefix"]}, {"base_prefix": str(venv)}):
                with self.subTest(change=change), self.assertRaises(RuntimeError):
                    worker_runtime.validate_worker_identity(dict(valid, **change), venv)

    def test_inherited_python_and_package_targets_are_removed(self):
        """Comfy launch variables and pip/uv target overrides must not escape the worker."""
        inherited = {"PYTHONHOME": "/comfy", "PYTHONPATH": "/comfy/site-packages", "VIRTUAL_ENV": "/comfy", "PIP_TARGET": "/comfy/site-packages", "PIP_PREFIX": "/comfy", "UV_SYSTEM_PYTHON": "1", "PATH": "preserved", "HTTPS_PROXY": "preserved-proxy"}
        with patch.dict(os.environ, inherited, clear=True):
            environment = worker_runtime.worker_environment()
        for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "PIP_TARGET", "PIP_PREFIX", "UV_SYSTEM_PYTHON"):
            self.assertNotIn(name, environment)
        self.assertEqual(environment["PIP_CONFIG_FILE"], os.devnull)
        self.assertEqual(environment["PATH"], "preserved")
        self.assertEqual(environment["HTTPS_PROXY"], "preserved-proxy")

    def test_mirror_and_certificate_settings_reach_pip_and_uv(self):
        """Corporate indexes and CA bundles must survive the install-target scrub."""
        kept = {"PIP_INDEX_URL": "https://mirror/simple", "PIP_CERT": "/ca.pem", "UV_INDEX_URL": "https://mirror/simple",
                "UV_CACHE_DIR": "/cache", "UV_SYSTEM_CERTS": "1", "SSL_CERT_FILE": "/ca.pem"}
        dropped = {"PIP_USER": "1", "PIP_TARGET": "/t", "UV_PROJECT_ENVIRONMENT": "/comfy", "UV_PYTHON": "/comfy/python"}
        with patch.dict(os.environ, {**kept, **dropped}, clear=True):
            environment = worker_runtime.worker_environment()
        for name, value in kept.items():
            self.assertEqual(environment[name], value)
        for name in dropped:
            self.assertNotIn(name, environment)

    def test_existing_broken_venv_points_to_recreate(self):
        """Re-running the plain installer cannot repair a broken .venv; say how to rebuild it."""
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            self.make_environment(package)
            (package / ".venv/pyvenv.cfg").write_text("include-system-site-packages = true\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "--recreate"):
                worker_runtime.worker_python(package)
            with patch.object(install, "PACKAGE_DIR", package.resolve()), \
                    patch.object(install, "select_python", return_value=None), \
                    patch.object(install.shutil, "which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "--recreate"):
                    install.install_worker(None, None)
                with self.assertRaisesRegex(RuntimeError, "Refusing to remove"):
                    install.install_worker(None, None, recreate=True)
            self.assertTrue((package / ".venv").is_dir())

    def test_remove_private_venv_never_follows_links(self):
        """--recreate deletes the package's own .venv, never a linked foreign environment."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "node"
            package.mkdir()
            self.make_environment(package)
            worker_runtime.remove_private_venv(package)
            self.assertFalse((package / ".venv").exists())
            if os.name != "nt":
                foreign = root / "comfy"
                self.make_environment(foreign)
                (package / ".venv").symlink_to(foreign / ".venv", target_is_directory=True)
                worker_runtime.remove_private_venv(package)
                self.assertFalse((package / ".venv").is_symlink())
                self.assertTrue((foreign / ".venv/pyvenv.cfg").is_file())

    def test_worker_runner_reports_utf8_and_exit_code(self):
        """Worker output is captured from files and decoded without masking errors."""
        code = "import sys; sys.stdout.write('ok'); sys.stderr.buffer.write('Fehler \u00e4'.encode() + b'\\xff'); sys.exit(3)"
        with tempfile.TemporaryDirectory() as directory:
            returncode, stdout, stderr = worker_runtime.run_worker(
                [sys.executable, "-c", code], cwd=Path(directory), env=dict(os.environ), log_dir=Path(directory))
        self.assertEqual((returncode, stdout), (3, "ok"))
        self.assertTrue(stderr.startswith("Fehler \u00e4"))

    def test_worker_runner_kills_on_timeout_and_interrupt(self):
        """A stuck or cancelled worker must not block the ComfyUI queue."""
        class Interrupted(BaseException):
            pass

        def interrupt():
            raise Interrupted()

        sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
        with tempfile.TemporaryDirectory() as directory:
            started = time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):
                worker_runtime.run_worker(sleeper, cwd=Path(directory), env=dict(os.environ),
                                          log_dir=Path(directory), timeout=.5, poll=.05)
            with self.assertRaises(Interrupted):
                worker_runtime.run_worker(sleeper, cwd=Path(directory), env=dict(os.environ),
                                          log_dir=Path(directory), check=interrupt, poll=.05)
            self.assertLess(time.monotonic() - started, 20)

    def test_bad_model_download_keeps_existing_bytes(self):
        """A truncated or untrusted download must not destroy the previously installed model."""
        with tempfile.TemporaryDirectory(prefix="model paths ") as directory:
            root = Path(directory)
            target = root / "model.task"
            target.write_bytes(b"old model")
            download = root / "download.fixture"
            download.write_bytes(b"bad new model")
            expected = hashlib.sha256(b"correct new model").hexdigest()
            with self.assertRaises(RuntimeError):
                install.download_model(target, download.as_uri(), expected)
            self.assertEqual(target.read_bytes(), b"old model")
            self.assertEqual(set(root.iterdir()), {target, download})

    def test_verified_model_does_not_need_network_or_replacement(self):
        """Offline reinstalls preserve verified weights instead of redownloading them."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.task"
            target.write_bytes(b"verified model")
            expected = hashlib.sha256(b"verified model").hexdigest()
            original = target.stat().st_mtime_ns
            install.download_model(target, "invalid://not-a-download", expected)
            self.assertEqual(target.read_bytes(), b"verified model")
            self.assertEqual(target.stat().st_mtime_ns, original)

    def test_verified_download_replaces_corrupt_model(self):
        """Only complete checksum-verified weights may replace a corrupt installation."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "model.task"
            target.write_bytes(b"corrupt")
            download = root / "download.fixture"
            download.write_bytes(b"correct model")
            install.download_model(target, download.as_uri(), hashlib.sha256(b"correct model").hexdigest())
            self.assertEqual(target.read_bytes(), b"correct model")
            self.assertEqual(set(root.iterdir()), {target, download})


if __name__ == "__main__":
    unittest.main()
