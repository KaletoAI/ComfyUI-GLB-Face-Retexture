"""Guard private-worker isolation, cross-platform selection, and atomic model integrity."""

import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import install
import worker_runtime


class RuntimeTests(unittest.TestCase):
    def make_environment(self, package, platform=None):
        venv = package / ".venv"
        platform = os.name if platform is None else platform
        python = venv / ("Scripts/python.exe" if platform == "nt" else "bin/python")
        python.parent.mkdir(parents=True)
        python.write_bytes(b"private interpreter fixture")
        (venv / "pyvenv.cfg").write_text("include-system-site-packages = false\n", encoding="utf-8")
        return python

    def test_private_python_uses_platform_layout_in_paths_with_spaces(self):
        """Windows consumers must not accidentally use a POSIX worker or Comfy Python."""
        with tempfile.TemporaryDirectory(prefix="face retexture ") as directory:
            package = Path(directory)
            windows = self.make_environment(package, "nt")
            with patch.object(worker_runtime.os, "name", "nt"):
                self.assertEqual(worker_runtime.worker_python(package), windows)
            with patch.object(worker_runtime.os, "name", "posix"):
                with self.assertRaises(RuntimeError):
                    worker_runtime.worker_python(package)

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
