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


def worker_arguments(command):
    """Parse a bridge command with worker.py's own parser, as the worker does."""
    sys.path.insert(0, str(PACKAGE_DIR))
    try:
        spec = importlib.util.spec_from_file_location("glb_face_retexture_worker", PACKAGE_DIR / "worker.py")
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
    finally:
        sys.path.remove(str(PACKAGE_DIR))
    return worker.argument_parser().parse_args(command[6:])


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
        """A completed worker yields FILE_3D, a 1xHxWx3 IMAGE, the proof text and the texel mask."""
        def worker(command, **kwargs):
            arguments = dict(zip(command, command[1:]))
            self.assertEqual(command[1:5], ["-E", "-s", "-X", "utf8"])
            self.assertIsNotNone(kwargs["check"])
            self.assertGreater(kwargs["timeout"], 0)
            Path(arguments["--output-glb"]).write_bytes(b"glTF output")
            Image.new("RGB", (6, 4), (10, 20, 30)).save(arguments["--comparison-png"])
            Path(arguments["--proof-json"]).write_text(json.dumps({"status": "face_texture_only"}), encoding="utf-8")
            self.assertEqual(arguments["--seam-blend"], "true")
            Image.fromarray(np.array([[0, 128, 255]], dtype=np.uint8)).save(arguments["--projection-mask-png"])
            return 0, "", ""
        result = self.run_node(worker)
        glb, comparison, proof, mask = result["result"]
        self.assertEqual(mask.shape, (1, 1, 3, 3))
        np.testing.assert_allclose(mask.array[0, 0, :, 0], [0, 128/255, 1], rtol=1e-6)
        np.testing.assert_array_equal(mask.array[..., 0], mask.array[..., 2])
        self.assertEqual(Path(glb.get_source()).read_bytes(), b"glTF output")
        self.assertEqual(comparison.shape, (1, 4, 6, 3))
        self.assertEqual(json.loads(proof)["status"], "face_texture_only")
        self.assertEqual(result["ui"]["text"], [proof])

    def test_optional_controls_forwarded_and_workflow_order(self):
        schema = self.bridge.GLBFaceRetexture.INPUT_TYPES()
        controls = dict(light_balance=.5, tone_match=.75, brightness=.9, skin_recolor=.6, seam_blend=False)
        self.assertEqual(list(schema["optional"]), list(controls))
        self.assertIs(schema["optional"]["seam_blend"][1]["default"], True)
        def worker(command, **kwargs):
            arguments = worker_arguments(command)
            for name, value in controls.items():
                argument = getattr(arguments, name)
                if isinstance(value, bool):
                    self.assertEqual(argument, "true" if value else "false")
                else:
                    self.assertEqual(argument, value)
            return 1, "", "checked controls"
        with self.assertRaisesRegex(RuntimeError, "checked controls"):
            self.run_node(worker, **controls)
        workflow = json.loads((PACKAGE_DIR / "workflows/glb-face-retexture.json").read_text())
        node = next(n for n in workflow["nodes"] if n["type"] == "GLBFaceRetexture")
        widgets = {**schema["required"], **schema["optional"]}
        defaults = [definition[1]["default"] for definition in widgets.values() if "default" in definition[1]]
        self.assertEqual(node["widgets_values"], defaults)

    def test_missing_mask_is_an_error(self):
        """A worker that skips the mask did not finish; the node must not return partial outputs."""
        def worker(command, **kwargs):
            arguments = dict(zip(command, command[1:]))
            Path(arguments["--output-glb"]).write_bytes(b"glTF output")
            Image.new("RGB", (6, 4)).save(arguments["--comparison-png"])
            Path(arguments["--proof-json"]).write_text("{}", encoding="utf-8")
            return 0, "", ""
        with self.assertRaisesRegex(RuntimeError, "projection_mask.png"):
            self.run_node(worker)

    def test_regions_node_runs_the_worker_in_locate_mode(self):
        """GLBFaceRegions needs no photo and returns FILE_3D, preview IMAGE and proof."""
        def worker(command, **kwargs):
            self.assertIn("--locate", command)
            self.assertNotIn("--reference-image", command)
            self.assertEqual(worker_arguments(command).depth_window, .4)
            arguments = dict(zip(command, command[1:]))
            Path(arguments["--output-glb"]).write_bytes(b"glTF regions")
            Image.new("RGB", (5, 3)).save(arguments["--preview-png"])
            Path(arguments["--proof-json"]).write_text(json.dumps({"status": "face_regions_only"}), encoding="utf-8")
            return 0, "", ""
        node = self.bridge.GLBFaceRegions()
        self.assertEqual(self.bridge.NODE_CLASS_MAPPINGS["GLBFaceRegions"], self.bridge.GLBFaceRegions)
        with patch.object(self.bridge, "worker_python", return_value=Path(sys.executable)), \
                patch.object(self.bridge, "MODELS", ()), patch.object(self.bridge, "run_worker", worker):
            result = node.locate(File3D(str(self.source)), 256, 0.0, depth_window=.4)
            glb, preview, proof = result["result"]
            self.assertEqual(Path(glb.get_source()).read_bytes(), b"glTF regions")
            self.assertEqual(preview.shape, (1, 3, 5, 3))
            self.assertEqual(json.loads(proof)["status"], "face_regions_only")
            def silent(command, **kwargs):
                arguments = dict(zip(command, command[1:]))
                Path(arguments["--output-glb"]).write_bytes(b"glTF regions")
                Path(arguments["--proof-json"]).write_text("{}", encoding="utf-8")
                return 0, "", ""
            with patch.object(self.bridge, "run_worker", silent), self.assertRaisesRegex(RuntimeError, "regions.png"):
                node.locate(File3D(str(self.source)), 256, 0.0)
            with self.assertRaisesRegex(ValueError, "static textured GLB"):
                node.locate(File3D(str(self.root / "model.obj")), 256, 0.0)

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

    def work_dirs(self):
        return sorted((self.root / "temp").glob("glb_face_retexture_*"))

    def succeed(self, command, **kwargs):
        arguments = dict(zip(command, command[1:]))
        Path(arguments["--output-glb"]).write_bytes(b"glTF output")
        Image.new("RGB", (6, 4)).save(arguments["--comparison-png"])
        Path(arguments["--proof-json"]).write_text("{}", encoding="utf-8")
        Image.new("L", (3, 1)).save(arguments["--projection-mask-png"])
        (kwargs["log_dir"] / "worker.stderr.log").write_text("log", encoding="utf-8")
        return 0, "", ""

    def test_negative_exponent_values_reach_the_worker(self):
        """str(-1e-05) starts with "-" and an exponent; argparse took it for an option."""
        def worker(command, **kwargs):
            arguments = worker_arguments(command)
            self.assertEqual((arguments.yaw, arguments.light_balance), (-1e-05, 1e-07))
            return 1, "", "parsed"
        node = self.bridge.GLBFaceRetexture()
        with patch.object(self.bridge, "worker_python", return_value=Path(sys.executable)), \
                patch.object(self.bridge, "MODELS", ()), patch.object(self.bridge, "run_worker", worker), \
                self.assertRaisesRegex(RuntimeError, "parsed"):
            node.retexture(File3D(str(self.source)), self.image, 1.0, .18, 256, -1e-05, light_balance=1e-07)

    def test_success_keeps_only_the_returned_glb(self):
        glb = self.run_node(self.succeed, source=File3D(str(self.source)))["result"][0]
        (work_dir,) = self.work_dirs()
        self.assertEqual([path.name for path in work_dir.iterdir()], ["GLBFaceRetexture.glb"])
        self.assertEqual(Path(glb.get_source()).read_bytes(), b"glTF output")
        self.assertTrue(self.source.is_file())

    def test_failure_and_cancel_remove_the_work_directory(self):
        def cancelled(command, **kwargs):
            raise KeyboardInterrupt
        with self.assertRaises(RuntimeError):
            self.run_node(lambda command, **kwargs: (1, "", "boom"))
        with self.assertRaises(KeyboardInterrupt):
            self.run_node(cancelled)
        self.assertEqual(self.work_dirs(), [])

    def test_input_error_shows_only_its_message(self):
        stderr = ("W0000 00:00:1763241000.1 1 inference_feedback_manager.cc:114] Feedback manager requires a model\n"
                  "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR\n"
                  "FaceTextureError: Source mesh render: no usable face detected\n")
        with self.assertRaises(RuntimeError) as caught:
            self.run_node(lambda command, **kwargs: (3, "", stderr))
        self.assertEqual(str(caught.exception), "GLBFaceRetexture: Source mesh render: no usable face detected")

    def test_crash_details_drop_mediapipe_log_noise(self):
        stderr = ("I0000 00:00:1763241000.1 1 gl_context.cc:369] GPU suport is not available\n"
                  "/venv/image_segmenter.py:158: UserWarning: MessageFactory class is deprecated.\n"
                  "  graph_config = self._runner.get_graph_config()\n"
                  "Traceback (most recent call last):\n  File \"face_core.py\", line 1\nMemoryError\n")
        with self.assertRaises(RuntimeError) as caught:
            self.run_node(lambda command, **kwargs: (1, "", stderr))
        message = str(caught.exception)
        self.assertIn("Traceback (most recent call last):\n  File \"face_core.py\", line 1\nMemoryError", message)
        for noise in ("GPU suport", "UserWarning", "graph_config"):
            self.assertNotIn(noise, message)

    def test_worker_refuses_a_replaced_model(self):
        spec = importlib.util.spec_from_file_location("glb_face_retexture_runtime", PACKAGE_DIR / "worker_runtime.py")
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        models = self.root / "models"
        models.mkdir()
        runtime.verify_models(models)  # Missing files are reported elsewhere.
        (models / runtime.MODELS[0][0]).write_bytes(b"not the pinned model")
        with self.assertRaisesRegex(RuntimeError, "pinned SHA-256"):
            runtime.verify_models(models)


if __name__ == "__main__":
    unittest.main()
