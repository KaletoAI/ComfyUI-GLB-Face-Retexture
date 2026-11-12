"""Face region labelling and its append-only GLB export on synthetic inputs (no ML models)."""
from pathlib import Path
import tempfile
import unittest

from tests.test_face_core import WORKER_IMPORT_ERROR

if WORKER_IMPORT_ERROR is None:
    import numpy as np
    from tests.test_face_core import core, fixture


def landmarks():
    """Eyes, brows, lips and nose as small clusters on a 100 px render."""
    points = np.full((478, 3), .02)
    for code, _, ids, _ in core.REGIONS:
        centre = {1: (.3, .4), 2: (.7, .4), 3: (.3, .28), 4: (.7, .28), 5: (.5, .8), 6: (.5, .58)}[code]
        offsets = np.linspace(-.04, .04, len(ids))
        points[list(ids), 0] = centre[0] + offsets
        points[list(ids), 1] = centre[1] + offsets[::-1]*.3
    points[33, :2], points[263, :2] = (.25, .4), (.75, .4)
    return points


@unittest.skipIf(WORKER_IMPORT_ERROR is not None, "Standalone worker dependencies unavailable")
class RegionLabelTests(unittest.TestCase):
    def test_region_image_codes_and_span(self):
        codes, span = core.region_image(landmarks(), (100, 100))
        self.assertAlmostEqual(span, 50)
        self.assertEqual(codes[40, 30], 1)
        self.assertEqual(codes[40, 70], 2)
        self.assertEqual(codes[28, 30], 3)
        self.assertEqual(codes[80, 50], 5)
        self.assertEqual(codes[58, 50], 6)
        self.assertEqual(codes[5, 95], 0)

    def test_hidden_vertices_within_the_depth_window_only(self):
        codes, _ = core.region_image(landmarks(), (100, 100))
        # A flat visible surface at depth 0 over the whole render.
        rendered = {'owners': np.zeros((100, 100), dtype=np.int32), 'depth': np.zeros((100, 100))}
        xy = np.array([[30.5, 40.5], [30.5, 40.5], [30.5, 40.5], [95.5, 5.5], [-3, 40]])
        depth = np.array([0., -.2, -.9, 0., 0.])  # surface, socket, back of head, no region, off-frame
        labels = core.label_vertices({'xy': xy, 'depth': depth}, rendered, codes, .5)
        np.testing.assert_array_equal(labels, [1, 1, 0, 0, 0])

    def test_regions_glb_appends_attributes_and_keeps_the_original(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.glb'
            fixture(path)
            source = core.read_source(path)
        labels = np.array([1, 0, 6, 0, 0, 5], dtype=np.uint8)
        depth = np.array([0, 0, .1, 0, 0, .25])
        output = core.regions_glb(source, {'_FACE_REGION': labels, '_FACE_DEPTH': depth}, {'version': 1})
        doc, binary = core.parse_glb(output)
        self.assertEqual(binary[:len(source['binary'])], source['binary'])
        attributes = doc['meshes'][0]['primitives'][0]['attributes']
        np.testing.assert_array_equal(core.accessor_array(doc, binary, attributes['_FACE_REGION']), labels)
        np.testing.assert_allclose(core.accessor_array(doc, binary, attributes['_FACE_DEPTH']), depth, rtol=1e-6)
        original = source['primitive']['attributes']
        self.assertEqual({k: v for k, v in attributes.items() if not k.startswith('_FACE')}, original)
        self.assertEqual(doc['extras']['glb_face_regions'], {'version': 1})
        for key in ('nodes', 'scenes', 'images', 'materials', 'textures'):
            self.assertEqual(doc[key], source['gltf'][key])
        again = core.regions_glb(source, {'_FACE_REGION': labels, '_FACE_DEPTH': depth}, {'version': 1})
        self.assertEqual(output, again)


if __name__ == '__main__':
    unittest.main()
