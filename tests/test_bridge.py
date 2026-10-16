"""Guard the ComfyUI node bridge with stub Comfy/torch modules and a fake worker."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from tests.test_face_core import WORKER_IMPORT_ERROR

if WORKER_IMPORT_ERROR is None:
    import numpy as np
    from PIL import Image

PACKAGE_DIR = Path(__file__).resolve().parents[1]


class FakeTensor:
    """The small slice of torch.Tensor the bridge touches."""

    def __init__(self, array):
        self.array = np.asarray(array)
        self.ndim, self.shape = self.array.ndim, self.array.shape

    def __getitem__(self, index):
        return FakeTensor(self.array[index])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def unsqueeze(self, axis):
        return FakeTensor(np.expand_dims(self.array, axis))


class File3D:
    def __init__(self, source, file_format=""):
        self.source = source
        self.format = file_format or Path(str(source)).suffix.lstrip(".").lower()

    def get_source(self):
        return self.source

    def save_to(self, path):
        Path(path).write_bytes(Path(self.source).read_bytes())
        return path


def load_bridge(temp_directory):
    """Import the package __init__ as a package with stubbed Comfy dependencies."""
    comfy = types.ModuleType("comfy")
    management = types.ModuleType("comfy.model_management")
    management.throw_exception_if_processing_interrupted = lambda: None
    comfy.model_management = management
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_temp_directory = lambda: str(temp_directory)
    torch = types.ModuleType("torch")
    torch.from_numpy = FakeTensor
    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.Types = types.SimpleNamespace(File3D=File3D)
    stubs = {"comfy": comfy, "comfy.model_management": management, "folder_paths": folder_paths,
             "torch": torch, "comfy_api": comfy_api, "comfy_api.latest": latest}
    with patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location(
            "glb_face_retexture_bridge", PACKAGE_DIR / "__init__.py", submodule_search_locations=[str(PACKAGE_DIR)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)
            sys.modules.pop(spec.name + ".worker_runtime", None)
    return module


@unittest.skipIf(WORKER_IMPORT_ERROR is not None, "Standalone worker dependencies unavailable")
class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.bridge = load_bridge(self.root / "temp")
        self.source = self.root / "source.glb"
        self.source.write_bytes(b"glTF fixture")
        self.image = FakeTensor(np.full((1, 8, 8, 3), .5, dtype=np.float32))

    def tearDown(self):
        self.directory.cleanup()

    def run_node(self, worker, image=None, source=None, **controls):
        node = self.bridge.GLBFaceRetexture()
        with patch.object(self.bridge, "worker_python", return_value=Path(sys.executable)), \
                patch.object(self.bridge, "MODELS", ()), patch.object(self.bridge, "run_worker", worker):
            return node.retexture(source or File3D(str(self.source)), image or self.image, 1.0, .18, 256, 0.0, **controls)

    def test_success_returns_native_outputs_from_worker_files(self):
        """A completed worker yields FILE_3D, a 1xHxWx3 IMAGE and the proof text."""
        def worker(command, **kwargs):
            arguments = dict(zip(command, command[1:]))
            self.assertEqual(command[1:5], ["-E", "-s", "-X", "utf8"])
            self.assertIsNotNone(kwargs["check"])
            self.assertGreater(kwargs["timeout"], 0)
            Path(arguments["--output-glb"]).write_bytes(b"glTF output")
            Image.new("RGB", (6, 4), (10, 20, 30)).save(arguments["--comparison-png"])
            Path(arguments["--proof-json"]).write_text(json.dumps({"status": "face_texture_only"}), encoding="utf-8")
            return 0, "", ""
        result = self.run_node(worker)
        glb, comparison, proof = result["result"]
        self.assertEqual(Path(glb.get_source()).read_bytes(), b"glTF output")
        self.assertEqual(comparison.shape, (1, 4, 6, 3))
        self.assertEqual(json.loads(proof)["status"], "face_texture_only")
        self.assertEqual(result["ui"]["text"], [proof])

    def test_optional_controls_forwarded_and_workflow_order(self):
        schema = self.bridge.GLBFaceRetexture.INPUT_TYPES()
        controls = dict(light_balance=.5, tone_match=.75, brightness=.9, skin_recolor=.6)
        self.assertEqual(list(schema["optional"]), list(controls))
        def worker(command, **kwargs):
            for name, value in controls.items():
                self.assertEqual(float(command[command.index("--" + name.replace("_", "-"))+1]), value)
            return 1, "", "checked controls"
        with self.assertRaisesRegex(RuntimeError, "checked controls"):
            self.run_node(worker, **controls)
        workflow = json.loads((PACKAGE_DIR / "workflows/glb-face-retexture.json").read_text())
        node = next(n for n in workflow["nodes"] if n["type"] == "GLBFaceRetexture")
        widgets = {**schema["required"], **schema["optional"]}
        defaults = [definition[1]["default"] for definition in widgets.values() if "default" in definition[1]]
        self.assertEqual(node["widgets_values"], defaults)

    def test_worker_failure_surfaces_its_error(self):
        """The worker's stderr is the user's error message; no output is returned."""
        def worker(command, **kwargs):
            return 1, "", "FaceTextureError: no unique usable face detected"
        with self.assertRaisesRegex(RuntimeError, "no unique usable face"):
            self.run_node(worker)

    def test_timeout_is_reported_with_remedy(self):
        def worker(command, **kwargs):
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        with self.assertRaisesRegex(RuntimeError, "GLB_FACE_RETEXTURE_TIMEOUT"):
            self.run_node(worker)

    def test_timeout_setting_is_validated(self):
        with patch.dict("os.environ", {"GLB_FACE_RETEXTURE_TIMEOUT": "90"}):
            self.assertEqual(self.bridge.worker_timeout(), 90)
        for value in ("0", "-5", "nan", "inf", "soon"):
            with self.subTest(value=value), patch.dict("os.environ", {"GLB_FACE_RETEXTURE_TIMEOUT": value}):
                with self.assertRaises(ValueError):
                    self.bridge.worker_timeout()

    def test_batches_and_non_glb_inputs_are_refused_before_the_worker(self):
        def worker(command, **kwargs):
            self.fail("worker must not start")
        with self.assertRaisesRegex(ValueError, "exactly one RGB"):
            self.run_node(worker, image=FakeTensor(np.zeros((2, 8, 8, 3), dtype=np.float32)))
        with self.assertRaisesRegex(ValueError, "static textured GLB"):
            self.run_node(worker, source=File3D(str(self.root / "model.obj")))


if __name__ == "__main__":
    unittest.main()
