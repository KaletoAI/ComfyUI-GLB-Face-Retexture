"""Seam blend, projection mask and seam measurement on synthetic inputs (no ML models)."""
import io
from pathlib import Path
import tempfile
import unittest

from tests.test_face_core import WORKER_IMPORT_ERROR

if WORKER_IMPORT_ERROR is None:
    import numpy as np
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    from tests.test_face_core import core, fixture, srgb_to_linear

IDENTITY = [[1, 0], [0, 1], [0, 0]]


def scene(size=160, inner=(16, 144), ramp=8):
    """Uniform mesh skin around a square projection with a linear feather ramp."""
    surface = np.zeros((size, size, 3), dtype=np.uint8)
    surface[:] = (200, 150, 120)
    square = np.zeros((size, size), dtype=bool)
    square[inner[0]:inner[1], inner[0]:inner[1]] = True
    mask = np.clip(distance_transform_edt(square)/ramp, 0, 1).astype(np.float32)
    photo = np.zeros((size, size, 3), dtype=np.float64)
    photo[:] = (150, 110, 100)
    # High-frequency detail the blend must keep.
    photo[::2, ::2] += 12
    landmarks = np.full((478, 3), .5)
    landmarks[33, :2], landmarks[263, :2] = (.4, .5), (.6, .5)
    # Lips just inside the top edge: the band reaches them there.
    landmarks[list(core.FEATURES['lips']), :2] = (.5, 22/size)
    return surface, mask, photo, landmarks


@unittest.skipIf(WORKER_IMPORT_ERROR is not None, "Standalone worker dependencies unavailable")
class SeamFieldTests(unittest.TestCase):
    def field(self, skin=None):
        surface, mask, photo, landmarks = scene()
        owners = np.zeros(mask.shape, dtype=np.int32)
        skin = np.ones(mask.shape, dtype=np.float32) if skin is None else skin
        return core.seam_field(surface, owners, mask, skin, photo, np.ones_like(mask), landmarks,
                               np.array(IDENTITY, dtype=float), .18)

    def test_edge_low_frequencies_match_outside_skin(self):
        field, report = self.field()
        photo_mean = .75*srgb_to_linear([150, 110, 100]) + .25*srgb_to_linear([162, 122, 112])
        expected = np.log(srgb_to_linear([200, 150, 120])/photo_mean)
        # Outer feather on the left edge: the photo's mean is pulled to the mesh skin.
        np.testing.assert_allclose(field[80, 18], expected, atol=.03)
        self.assertGreater(report['edge_pixels'], 0)

    def test_interior_and_features_are_untouched(self):
        field, _ = self.field()
        # 40 px inside the edge, beyond feather + band.
        np.testing.assert_array_equal(field[56:104, 56:104], 0)
        # Lips sit 6 px inside the top edge, where the band is otherwise full.
        np.testing.assert_array_equal(field[22, 80], 0)
        self.assertGreater(np.abs(field[22, 40]).min(), .1)

    def test_no_correction_without_outside_skin(self):
        skin = np.ones((160, 160), dtype=np.float32)
        # A hairline: the mask edge comes from the semantics, so neither the
        # feather nor anything left of it is skin.
        skin[:, :24] = 0
        field, _ = self.field(skin)
        self.assertLess(np.abs(field[80, 18]).max(), 1e-3)
        self.assertGreater(np.abs(field[80, 141]).min(), .1)

    def test_deterministic(self):
        first, report = self.field()
        second, again = self.field()
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(report, again)


@unittest.skipIf(WORKER_IMPORT_ERROR is not None, "Standalone worker dependencies unavailable")
class PaintSeamTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        path = Path(self.directory.name) / 'source.glb'
        fixture(path)
        self.source = core.read_source(path)
        self.source['factor'] = np.ones(4)
        self.before = core.render_albedo(self.source, core.make_view(self.source['positions'], 64, 0))

    def tearDown(self):
        self.directory.cleanup()

    def paint(self, **options):
        photo = np.full((64, 64, 3), [100, 150, 200], dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        return core.paint_atlas(self.source, np.array([True, False]), self.before, mask, photo, mask,
                                np.array(IDENTITY), .75, **options)

    def test_mask_is_the_blend_amount_and_off_is_unchanged(self):
        atlas, report = self.paint()
        same, again, details = self.paint(with_mask=True)
        self.assertEqual(atlas, same)
        self.assertEqual(report, again)
        mask = details['mask']
        np.testing.assert_allclose(np.unique(mask), [0, .75])
        self.assertEqual(int((mask > 0).sum()), report['painted_texture_pixels'])

    def test_seam_gain_scales_the_photo_in_linear_light(self):
        seam = np.full((64, 64, 3), np.log(.5), dtype=np.float32)
        atlas, _, details = self.paint(seam=seam, with_mask=True)
        painted = np.asarray(Image.open(io.BytesIO(atlas)))[16, 16, :3]
        # The photo is halved in linear light, then blended over the old texel in sRGB.
        halved = core._linear_to_srgb(srgb_to_linear([100, 150, 200])*.5)
        np.testing.assert_array_equal(painted, np.rint(np.array([20, 40, 60])*.25 + halved*.75))
        self.assertEqual(details['mask'][16, 16], np.float32(.75))


@unittest.skipIf(WORKER_IMPORT_ERROR is not None, "Standalone worker dependencies unavailable")
class SeamMeasureTests(unittest.TestCase):
    def test_delta_e_across_a_known_step(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.glb'
            fixture(path)
            source = core.read_source(path)
        body = np.zeros((32, 32, 4), dtype=np.uint8)
        body[:] = (200, 150, 120, 255)
        face = body.copy()
        face[:, :16, :3] = (150, 110, 100)
        mask = np.zeros((32, 32), dtype=np.float32)
        mask[:, :16] = 1
        report = core.seam_delta_e(source, np.array([True, False]), body, face, mask)
        expected = np.linalg.norm(core._srgb_to_lab(np.array([150, 110, 100]))
                                  - core._srgb_to_lab(np.array([200, 150, 120])))
        self.assertGreater(report['samples'], 5)
        self.assertAlmostEqual(report['p50'], expected, places=6)
        self.assertAlmostEqual(report['p95'], expected, places=6)
        everything = np.ones((32, 32), dtype=bool)
        excluded = core.seam_delta_e(source, np.array([True, False]), body, face, mask, exclude=everything)
        self.assertEqual(excluded['outside_features'], {'samples': 0})
        # The body atlas is what non-selected triangles show: without selection there is no step.
        unselected = core.seam_delta_e(source, np.array([False, False]), body, face, mask)
        self.assertAlmostEqual(unselected['p95'], 0)

    def test_lab_reference_values(self):
        np.testing.assert_allclose(core._srgb_to_lab(np.array([255, 255, 255])), [100, 0, 0], atol=.01)
        np.testing.assert_allclose(core._srgb_to_lab(np.array([255, 0, 0])), [53.24, 80.09, 67.20], atol=.05)


if __name__ == '__main__':
    unittest.main()
