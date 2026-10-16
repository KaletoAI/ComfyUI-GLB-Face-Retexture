"""Guard real CPU vision runs: refuse unusable photos, preserve geometry on success."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from tests.test_face_core import WORKER_IMPORT_ERROR

if WORKER_IMPORT_ERROR is None:
    import numpy as np
    from PIL import Image
    from tests.test_face_core import core, fixture

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
PORTRAIT = Path(__file__).resolve().parent / "fixtures" / "portrait.jpg"
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


    def test_matching_portrait_recolors_only_face_and_preserves_geometry(self):
        """Happy path: a real face is registered, painted, exported and verified."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.glb"
            textured_grid(source, PORTRAIT.read_bytes())
            original = source.read_bytes()
            with Image.open(PORTRAIT) as image:
                pixels = np.asarray(image.convert("RGB")).astype(float)
            # Same face, cooler color balance: the face region must visibly change.
            Image.fromarray((pixels * [.8, 1, 1.15]).clip(0, 255).astype(np.uint8)).save(root / "reference.png")
            outputs = root / "corrected.glb", root / "comparison.png", root / "proof.json"
            report = core.run_retexture(source, root / "reference.png", *outputs, MODEL_DIR, render_resolution=256)
            # Explicit defaults must retain GLB, comparison and proof bytes.
            expected = [path.read_bytes() for path in outputs]
            core.run_retexture(source, root / "reference.png", *outputs, MODEL_DIR, render_resolution=256,
                               light_balance=0, tone_match=0, brightness=1)
            self.assertEqual(expected, [path.read_bytes() for path in outputs])
            # Exercise the real semantic ring and all active controls as well.
            active = core.run_retexture(source, root / "reference.png", root / "active.glb",
                root / "active.png", root / "active.json", MODEL_DIR, render_resolution=256,
                light_balance=.5, tone_match=.5, brightness=.9)
            self.assertIn("tone", active["photo_adjust"])
            self.assertEqual(active["reference_rgb_sha256"], report["reference_rgb_sha256"])
            self.assertEqual(source.read_bytes(), original)
            proof = json.loads(outputs[2].read_text())
            self.assertEqual(proof, json.loads(json.dumps(report)))
            self.assertEqual(proof["schema_version"], 2)
            self.assertTrue(proof["geometry_unchanged"])
            self.assertTrue(proof["original_embedded_images_exact"])
            self.assertEqual(proof["protected_body_changed_render_pixels"], 0)
            self.assertGreater(proof["changed_head_render_pixels"], 1000)
            self.assertEqual(proof["output_triangle_count"], proof["original_triangle_count"])
            self.assertLess(0, proof["selected_triangles"])
            self.assertLess(proof["selected_triangles"], proof["original_triangle_count"] // 2)
            self.assertNotIn(33, proof["registration"]["diagnostic_landmark_indices"])
            self.assertLess(proof["registration"]["stable_rms_photo_eye_fraction"], .1)
            doc, binary = core.parse_glb(outputs[0].read_bytes())
            self.assertEqual(binary[:len(core.read_source(source)["binary"])], core.read_source(source)["binary"])
            self.assertEqual(len(doc["meshes"][0]["primitives"]), 2)
            with Image.open(outputs[1]) as comparison:
                self.assertEqual(comparison.size, (256 * 3, 256 + 32))


def textured_grid(path, texture, cells=24):
    """A flat +Z-facing grid whose UVs map the whole portrait texture."""
    n = cells + 1
    xs = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(xs, xs[::-1])
    positions = np.c_[xx.ravel(), yy.ravel(), np.zeros(n * n)].astype("<f4")
    uv = np.c_[(xx.ravel() + 1) / 2, (1 - yy.ravel()) / 2].astype("<f4")
    corner = (np.arange(cells)[:, None] * n + np.arange(cells)[None, :]).ravel()
    triangles = np.r_[np.c_[corner, corner + n, corner + 1], np.c_[corner + 1, corner + n, corner + n + 1]].astype("<u4")
    binary = bytearray()
    views = []
    for chunk in (triangles.tobytes(), positions.tobytes(), uv.tobytes(), texture):
        binary.extend(b"\0" * (-len(binary) % 4))
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(chunk)})
        binary.extend(chunk)
    binary.extend(b"\0" * (-len(binary) % 4))
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 1, "TEXCOORD_0": 2}, "indices": 0, "material": 0}]}],
           "buffers": [{"byteLength": len(binary)}], "bufferViews": views,
           "accessors": [{"bufferView": 0, "componentType": 5125, "count": triangles.size, "type": "SCALAR"},
                         {"bufferView": 1, "componentType": 5126, "count": n * n, "type": "VEC3",
                          "min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist()},
                         {"bufferView": 2, "componentType": 5126, "count": n * n, "type": "VEC2"}],
           "images": [{"bufferView": 3, "mimeType": "image/jpeg"}], "textures": [{"source": 0}],
           "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]}
    payload = json.dumps(doc).encode()
    payload += b" " * (-len(payload) % 4)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 28 + len(payload) + len(binary))
                     + struct.pack("<II", len(payload), 0x4E4F534A) + payload
                     + struct.pack("<II", len(binary), 0x004E4942) + binary)


if __name__ == "__main__":
    unittest.main()
