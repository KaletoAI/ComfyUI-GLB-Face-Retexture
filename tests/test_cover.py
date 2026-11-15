"""Fold cover, released feature veto, coverage mask and the GLB record (no ML models)."""
import io
import math
from pathlib import Path
import re
import tempfile
import unittest

from tests.test_face_core import WORKER_IMPORT_ERROR

if WORKER_IMPORT_ERROR is None:
    import numpy as np
    from PIL import Image
    from tests.test_face_core import core, fixture, texel_for

IDENTITY = [[1, 0], [0, 1], [0, 0]]
ORIGINAL = [20, 40, 60]


def two_layers(path, gap):
    """A front triangle and, `gap` behind it, a smaller back-wound triangle it hides; separate UVs."""
    back = 1-gap
    positions = [[-.8, -.8, 1], [.8, -.8, 1], [0, .8, 1],
                 [.4, -.4, back], [-.4, -.4, back], [0, .4, back]]
    uv = [[.05, .05], [.45, .05], [.25, .45], [.95, .55], [.55, .55], [.75, .95]]
    fixture(path, positions=positions, uv=uv)
    return core.read_source(path)


# Texel inside the hidden triangle's UV footprint (row from v, column from u).
BACK_TEXEL = (22, 24)


def oval_landmarks():
    """Face-oval landmarks on a circle covering most of the render."""
    landmarks = np.full((478, 3), .5)
    for i, index in enumerate(core.FACE_OVAL):
        angle = 2*math.pi*i/len(core.FACE_OVAL)
        landmarks[index, :2] = (.5+.49*math.cos(angle), .5+.49*math.sin(angle))
    return landmarks


@unittest.skipIf(WORKER_IMPORT_ERROR is not None, "Standalone worker dependencies unavailable")
class FoldCoverTests(unittest.TestCase):
    def paint(self, gap, cover):
        with tempfile.TemporaryDirectory() as directory:
            source = two_layers(Path(directory)/'source.glb', gap)
        rendered = core.render_albedo(source, core.make_view(source['positions'], 64, 0))
        ones = np.ones((64, 64), dtype=np.float32)
        photo = np.full((64, 64, 3), [100, 150, 200], dtype=np.uint8)
        if cover is not None:
            cover = {'band': np.full((64, 64), cover[0]), 'window': cover[1]}
        atlas, report, painted = core.paint_atlas(source, np.array([True, True]), rendered, ones, photo, ones,
                                                  np.array(IDENTITY), 1, with_mask=True, cover=cover)
        return np.asarray(Image.open(io.BytesIO(atlas))), report, painted, source, rendered

    def test_hidden_texels_near_the_surface_take_the_photo(self):
        atlas, report, painted, _, _ = self.paint(.05, (True, .1))
        np.testing.assert_array_equal(atlas[BACK_TEXEL][:3], [texel_for(100, .5), 150, 200])
        self.assertGreater(painted['mask'][BACK_TEXEL], 0)
        self.assertGreater(report['hidden_texture_pixels_painted'], 0)

    def test_without_cover_hidden_texels_keep_the_original(self):
        atlas, report, painted, _, _ = self.paint(.05, None)
        np.testing.assert_array_equal(atlas[BACK_TEXEL][:3], ORIGINAL)
        self.assertEqual(painted['mask'][BACK_TEXEL], 0)
        self.assertNotIn('hidden_texture_pixels_painted', report)

    def test_cover_stops_at_the_depth_window_and_the_band(self):
        for gap, cover in ((.2, (True, .1)), (.05, (False, .1))):
            atlas, report, _, _, _ = self.paint(gap, cover)
            np.testing.assert_array_equal(atlas[BACK_TEXEL][:3], ORIGINAL)
            self.assertEqual(report['hidden_texture_pixels_painted'], 0)

    def test_cover_leaves_the_visible_painting_unchanged(self):
        plain, _, plain_mask, _, _ = self.paint(.05, None)
        covered, _, covered_mask, _, _ = self.paint(.05, (True, .1))
        front = np.zeros(plain_mask['mask'].shape, dtype=bool)
        front[:16, :16] = True
        np.testing.assert_array_equal(plain[front], covered[front])
        np.testing.assert_array_equal(plain_mask['mask'][front], covered_mask['mask'][front])

    def test_cover_triangles_take_any_facing_within_the_window(self):
        for gap, band, expected in ((.05, True, [True, True]), (.2, True, [True, False]), (.05, False, [False, False])):
            with tempfile.TemporaryDirectory() as directory:
                source = two_layers(Path(directory)/'source.glb', gap)
            rendered = core.render_albedo(source, core.make_view(source['positions'], 64, 0))
            chosen = core.cover_triangles(source, rendered, np.ones((64, 64)), np.full((64, 64), band), .1)
            self.assertEqual(chosen.tolist(), expected)

    def test_coverage_mask_marks_written_missed_and_outside(self):
        _, _, covered, source, rendered = self.paint(.05, (True, .1))
        mask = core.coverage_mask(source, rendered, oval_landmarks(), covered['mask'], .1)
        self.assertEqual(mask[BACK_TEXEL], core.COVER_WRITTEN)
        self.assertEqual(mask[31, 0], core.COVER_OUTSIDE)
        _, _, plain, _, _ = self.paint(.05, None)
        missed = core.coverage_mask(source, rendered, oval_landmarks(), plain['mask'], .1)
        self.assertEqual(missed[BACK_TEXEL], core.COVER_MISSED)
        self.assertEqual(set(np.unique(missed)), {core.COVER_WRITTEN, core.COVER_MISSED, core.COVER_OUTSIDE})
        self.assertEqual(core.coverage_mask(source, rendered, oval_landmarks(), plain['mask'], .01)[BACK_TEXEL],
                         core.COVER_OUTSIDE)


def feature_landmarks(size):
    """Eye+brow landmark rings at (60, 80) and (140, 80) px of a `size` square render; the rest far below."""
    landmarks = np.full((478, 3), .5)
    landmarks[:, 1] = .95
    for key, centre in (('left_eye_brow', (60, 80)), ('right_eye_brow', (140, 80))):
        ids = core.FEATURES[key]
        for i, index in enumerate(ids):
            angle = 2*math.pi*i/len(ids)
            landmarks[index, :2] = ((centre[0]+10*math.cos(angle))/size, (centre[1]+6*math.sin(angle))/size)
    return landmarks


@unittest.skipIf(WORKER_IMPORT_ERROR is not None, "Standalone worker dependencies unavailable")
class FeatureVetoTests(unittest.TestCase):
    def test_painted_brow_blob_is_released_but_a_fringe_is_not(self):
        semantic = np.ones((200, 200), dtype=np.float32)
        semantic[70:84, 48:72] = 0      # dark brow block painted on the mesh
        semantic[0:84, 130:150] = 0     # fringe from the hairline down over the other brow
        semantic[150:160, 90:110] = 0   # a dark spot far from the eyes
        released, blobs = core.release_feature_veto(semantic, feature_landmarks(200))
        self.assertEqual(blobs, 1)
        self.assertTrue((released[70:84, 48:72] == 1).all())
        self.assertFalse(released[0:84, 130:150].any())
        self.assertFalse(released[150:160, 90:110].any())

    def test_nothing_to_release(self):
        released, blobs = core.release_feature_veto(np.ones((200, 200), dtype=np.float32), feature_landmarks(200))
        self.assertEqual(blobs, 0)
        self.assertFalse(released.any())


@unittest.skipIf(WORKER_IMPORT_ERROR is not None, "Standalone worker dependencies unavailable")
class RecordTests(unittest.TestCase):
    def test_version_matches_the_package(self):
        version = re.search(r'^version = "([^"]+)"', (Path(__file__).parents[1]/'pyproject.toml').read_text(), re.M)
        self.assertEqual(core.VERSION, version.group(1))

    def test_glb_carries_the_coverage_mask_and_version(self):
        coverage = io.BytesIO()
        Image.fromarray(np.array([[0, 128, 255]], dtype=np.uint8)).save(coverage, format='PNG')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.glb'
            _, binary, image = fixture(path, change=lambda doc: doc.update(extras={'kept': 1}))
            source = core.read_source(path)
            output = core.partition_glb(source, np.array([True, False]), image, coverage_png=coverage.getvalue(),
                                        meta={'version': core.VERSION, 'coverage_mask': {'written_texels': 1}})
        doc, emitted = core.parse_glb(output)
        self.assertEqual(emitted[:len(binary)], binary)
        record = doc['extras']['glb_face_retexture']
        self.assertEqual(doc['extras']['kept'], 1)
        self.assertEqual(record['version'], core.VERSION)
        self.assertEqual(record['coverage_mask']['written_texels'], 1)
        entry = doc['images'][record['coverage_mask']['image']]
        view = doc['bufferViews'][entry['bufferView']]
        self.assertEqual(emitted[view['byteOffset']:view['byteOffset']+view['byteLength']], coverage.getvalue())
        used = {texture['source'] for texture in doc['textures']}
        self.assertNotIn(record['coverage_mask']['image'], used)

    def test_non_object_extras_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.glb'
            _, _, image = fixture(path, change=lambda doc: doc.update(extras=[1]))
            source = core.read_source(path)
            with self.assertRaisesRegex(core.FaceTextureError, 'extras'):
                core.partition_glb(source, np.array([True, False]), image, meta={'version': core.VERSION})


if __name__ == '__main__':
    unittest.main()
