"""Guard refusal of unusable photos with the installed CPU vision models."""
from pathlib import Path
import tempfile
import unittest

from tests.test_face_core import WORKER_IMPORT_ERROR

if WORKER_IMPORT_ERROR is None:
    from PIL import Image
    from tests.test_face_core import core, fixture

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_AVAILABLE = all((MODEL_DIR / name).is_file() for name in
                       ("face_landmarker.task", "selfie_multiclass.tflite"))


@unittest.skipIf(WORKER_IMPORT_ERROR is not None or not MODELS_AVAILABLE,
                 "Run install.py to provide the private worker and vision models")
class VisionInputTests(unittest.TestCase):
    def test_faceless_reference_refuses_export_and_preserves_source(self):
        """A faceless input must fail, not emit a passthrough mesh as a successful edit."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.glb"
            photo = root / "reference.png"
            fixture(source)
            original = source.read_bytes()
            core.read_source(source)  # The mesh must pass the supported-asset gate.
            Image.new("RGB", (256, 256), (0, 0, 0)).save(photo)
            output = root / "corrected.glb"
            comparison = root / "comparison.png"
            proof = root / "proof.json"
            with self.assertRaises(core.FaceTextureError):
                core.run_retexture(source, photo, output, comparison, proof,
                                   MODEL_DIR, render_resolution=256)
            self.assertEqual(source.read_bytes(), original)
            for artifact in (output, comparison, proof):
                self.assertFalse(artifact.exists(), str(artifact))


if __name__ == "__main__":
    unittest.main()
