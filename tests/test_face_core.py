"""Guard append-only geometry and private-atlas isolation without ML models."""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

WORKER_IMPORT_ERROR = None
try:
    import numpy as np
    from PIL import Image
    import scipy
except ImportError as exc:
    WORKER_IMPORT_ERROR = str(exc)

if WORKER_IMPORT_ERROR is None:
    SPEC = importlib.util.spec_from_file_location('face_core', Path(__file__).parents[1] / 'face_core.py')
    core = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(core)


def srgb_to_linear(value):
    c = np.asarray(value, dtype=float) / 255
    return np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)


def linear_to_srgb(value):
    v = np.asarray(value, dtype=float)
    return 255 * np.where(v <= .0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - .055)


def texel_for(photo_value, factor):
    """The texel a glTF viewer turns into `photo_value` after the linear factor."""
    return int(np.rint(linear_to_srgb(srgb_to_linear(photo_value) / factor)))


def fixture(path, change=None):
    """Two separately indexed triangles intentionally share every UV texel."""
    positions = np.array([[-.8, -.8, 1], [.8, -.8, 1], [0, .8, 1],
                          [-.8, -.8, 0], [.8, -.8, 0], [0, .8, 0]], dtype='<f4')
    uv = np.array([[.1, .1], [.9, .1], [.5, .9]] * 2, dtype='<f4')
    normals = np.array([[0, 0, 1]] * 6, dtype='<f4')
    indices = np.arange(6, dtype='<u4')
    png = io.BytesIO()
    Image.new('RGBA', (32, 32), (20, 40, 60, 255)).save(png, format='PNG')
    chunks = [indices.tobytes(), positions.tobytes(), uv.tobytes(), normals.tobytes(), png.getvalue()]
    binary = bytearray()
    views = []
    for chunk in chunks:
        binary.extend(b'\0' * (-len(binary) % 4))
        views.append({'buffer': 0, 'byteOffset': len(binary), 'byteLength': len(chunk)})
        binary.extend(chunk)
    doc = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0]}],
           'nodes': [{'mesh': 0}], 'meshes': [{'primitives': [{'attributes': {'POSITION': 1, 'TEXCOORD_0': 2, 'NORMAL': 3}, 'indices': 0, 'material': 0}]}],
           'buffers': [{'byteLength': len(binary)}], 'bufferViews': views,
           'accessors': [{'bufferView': 0, 'componentType': 5125, 'count': 6, 'type': 'SCALAR'},
                         {'bufferView': 1, 'componentType': 5126, 'count': 6, 'type': 'VEC3'},
                         {'bufferView': 2, 'componentType': 5126, 'count': 6, 'type': 'VEC2'},
                         {'bufferView': 3, 'componentType': 5126, 'count': 6, 'type': 'VEC3'}],
           'images': [{'bufferView': 4, 'mimeType': 'image/png'}], 'textures': [{'source': 0}],
           'materials': [{'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}, 'baseColorFactor': [.5, 1, 1, 1], 'metallicFactor': .25, 'roughnessFactor': .8}, 'doubleSided': True}]}
    if change:
        change(doc)
    payload = json.dumps(doc).encode()
    payload += b' ' * (-len(payload) % 4)
    binary.extend(b'\0' * (-len(binary) % 4))
    raw = struct.pack('<4sII', b'glTF', 2, 28 + len(payload) + len(binary)) + struct.pack('<II', len(payload), 0x4E4F534A) + payload + struct.pack('<II', len(binary), 0x004E4942) + binary
    path.write_bytes(raw)
    return doc, bytes(binary), png.getvalue()


@unittest.skipIf(WORKER_IMPORT_ERROR is not None, "Standalone ComfyUI worker dependencies unavailable: " + str(WORKER_IMPORT_ERROR))
class PreservationTests(unittest.TestCase):
    def test_partition_keeps_every_attribute_triangle_and_original_image(self):
        """Rebuilding vertex rows or losing one partition must not silently pass."""
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / 'source.glb'
            original, binary, image = fixture(src)
            source = core.read_source(src)
            output = core.partition_glb(source, np.array([True, False]), image)
            doc, emitted = core.parse_glb(output)
            self.assertEqual(emitted[:len(binary)], binary)
            self.assertEqual(doc['accessors'][:4], original['accessors'])
            self.assertEqual(doc['bufferViews'][:5], original['bufferViews'])
            self.assertEqual(doc['images'][:1], original['images'])
            self.assertEqual(doc['materials'][:1], original['materials'])
            self.assertEqual(doc['nodes'], original['nodes'])
            primitives = doc['meshes'][0]['primitives']
            self.assertEqual(len(primitives), 2)
            for primitive in primitives:
                self.assertEqual(primitive['attributes'], {'POSITION': 1, 'TEXCOORD_0': 2, 'NORMAL': 3})
            np.testing.assert_array_equal(core.accessor_array(doc, emitted, primitives[0]['indices']), [3, 4, 5])
            np.testing.assert_array_equal(core.accessor_array(doc, emitted, primitives[1]['indices']), [0, 1, 2])
            self.assertEqual(primitives[0]['material'], 0)
            self.assertEqual(primitives[1]['material'], 1)
            self.assertEqual(doc['materials'][1]['pbrMetallicRoughness']['baseColorFactor'], [.5, 1, 1, 1])
            self.assertEqual(doc['materials'][1]['pbrMetallicRoughness']['roughnessFactor'], .8)

    def test_shared_uv_body_uses_original_texture_after_paint(self):
        """Painting a shared atlas must not recolor the protected body primitive."""
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / 'source.glb'
            dst = Path(directory) / 'result.glb'
            _, _, original_image = fixture(src)
            source = core.read_source(src)
            view = core.make_view(source['positions'], 64, 0)
            before = core.render_albedo(source, view)
            photo = np.full((64, 64, 3), [100, 150, 200], dtype=np.uint8)
            mask = np.ones((64, 64), dtype=np.float32)
            atlas, metrics = core.paint_atlas(source, np.array([True, False]), before, mask, photo, mask, np.array([[1, 0], [0, 1], [0, 0]]), 1)
            self.assertGreater(metrics['edited_texture_pixels'], 0)
            painted = np.asarray(Image.open(io.BytesIO(atlas)))
            painted_rgb = [texel_for(100, .5), 150, 200]
            np.testing.assert_array_equal(painted[16, 16, :3], painted_rgb)
            np.testing.assert_array_equal(painted[0, 16, :3], painted_rgb)
            np.testing.assert_array_equal(painted[31, 0], [20, 40, 60, 255])
            rear_view = core.make_view(source['positions'], 64, 180)
            body_before = core.render_albedo(source, rear_view)['rgb']
            body_after = core.render_albedo(source, rear_view, painted, np.array([True, False]))['rgb']
            np.testing.assert_array_equal(body_after, body_before)
            dst.write_bytes(core.partition_glb(source, np.array([True, False]), atlas))
            doc, binary = core.parse_glb(dst.read_bytes())
            body = doc['meshes'][0]['primitives'][0]
            material = doc['materials'][body['material']]
            texture = doc['textures'][material['pbrMetallicRoughness']['baseColorTexture']['index']]
            image = doc['images'][texture['source']]
            bv = doc['bufferViews'][image['bufferView']]
            self.assertEqual(binary[bv['byteOffset']:bv['byteOffset'] + bv['byteLength']], original_image)
            self.assertGreater(metrics['gutter_pixels'], 0)

    def test_projection_samples_gltf_texel_centers(self):
        """A half-texel UV shift must not misregister the portrait gradient."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.glb'
            fixture(path)
            source = core.read_source(path)
            view = {'xy': source['uv'] * 32, 'depth': source['positions'][:, 2],
                    'camera': source['positions'], 'resolution': 32}
            rendered = core.render_albedo(source, view)
            yy, xx = np.mgrid[:32, :32]
            photo = np.stack((xx, yy, np.full_like(xx, 80)), axis=2).astype(np.uint8)
            mask = np.ones((32, 32), dtype=np.float32)
            atlas, _ = core.paint_atlas(source, np.array([True, False]), rendered,
                                       mask, photo, mask, np.array([[1, 0], [0, 1], [0, 0]]), 1)
            painted = np.asarray(Image.open(io.BytesIO(atlas)))
            np.testing.assert_array_equal(painted[16, 16, :3], [texel_for(16, .5), 16, 80])

    def test_depth_gate_refuses_occluded_triangle_even_with_shared_uv(self):
        """Removing the depth test would paint the hidden back/body triangle."""
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / 'source.glb'
            fixture(src)
            source = core.read_source(src)
            rendered = core.render_albedo(source, core.make_view(source['positions'], 64, 0))
            photo = np.full((64, 64, 3), 200, dtype=np.uint8)
            mask = np.ones((64, 64), dtype=np.float32)
            with self.assertRaisesRegex(core.FaceTextureError, 'visible.*texels'):
                core.paint_atlas(source, np.array([False, True]), rendered, mask, photo, mask, np.array([[1, 0], [0, 1], [0, 0]]), 1)

    def test_base_color_factor_is_inverted_in_linear_light(self):
        """A glTF viewer must show the photo color, not an sRGB-space approximation."""
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / 'source.glb'
            fixture(src)
            source = core.read_source(src)
            view = core.make_view(source['positions'], 64, 0)
            before = core.render_albedo(source, view)
            mask = np.ones((64, 64), dtype=np.float32)
            affine = np.array([[1, 0], [0, 1], [0, 0]])
            photo = np.full((64, 64, 3), [100, 150, 200], dtype=np.uint8)
            atlas, metrics = core.paint_atlas(source, np.array([True, False]), before, mask, photo, mask, affine, 1)
            painted = np.asarray(Image.open(io.BytesIO(atlas)))
            shown = linear_to_srgb(srgb_to_linear(painted[16, 16, :3]) * source['factor'][:3])
            np.testing.assert_allclose(shown, [100, 150, 200], atol=1.5)
            after = core.render_albedo(source, view, painted, np.array([True, False]))['rgb']
            np.testing.assert_allclose(after[32, 32], [100, 150, 200], atol=1.5)
            self.assertEqual(metrics['factor_clipped_texture_pixels'], 0)
            bright = np.full((64, 64, 3), [230, 150, 200], dtype=np.uint8)
            _, clipped = core.paint_atlas(source, np.array([True, False]), before, mask, bright, mask, affine, 1)
            self.assertEqual(clipped['factor_clipped_texture_pixels'], clipped['painted_texture_pixels'])

    def test_gutter_never_overwrites_unpainted_face_texels(self):
        """Masked-out or occluded texels inside face triangles keep their original color."""
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / 'source.glb'
            fixture(src)
            source = core.read_source(src)
            view = core.make_view(source['positions'], 64, 0)
            before = core.render_albedo(source, view)
            half = np.ones((64, 64), dtype=np.float32)
            half[:, 32:] = 0
            photo = np.full((64, 64, 3), [100, 150, 200], dtype=np.uint8)
            atlas, metrics = core.paint_atlas(source, np.array([True, False]), before, half, photo,
                                              np.ones((64, 64), dtype=np.float32), np.array([[1, 0], [0, 1], [0, 0]]), 1)
            painted = np.asarray(Image.open(io.BytesIO(atlas)))
            after = core.render_albedo(source, view, painted, np.array([True, False]))['rgb']
            # Mask edge is column 32; bilinear mask and texture filtering reach < 4 pixels.
            far = before['owners'][:, 36:] == 0
            np.testing.assert_array_equal(after[:, 36:][far], before['rgb'][:, 36:][far])
            self.assertGreater(metrics['gutter_pixels'], 0)

    def test_vectorized_render_keeps_nearest_surface_and_first_on_ties(self):
        """The z buffer keeps the front triangle; equal depth keeps the earlier triangle."""
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / 'source.glb'
            fixture(src)
            source = core.read_source(src)
            front = core.render_albedo(source, core.make_view(source['positions'], 64, 0))
            self.assertEqual(set(np.unique(front['owners'])) - {-1}, {0})
            rear = core.render_albedo(source, core.make_view(source['positions'], 64, 180))
            self.assertEqual(set(np.unique(rear['owners'])) - {-1}, {1})
            flat = dict(source, positions=source['positions'] * [1, 1, 0])
            tie = core.render_albedo(flat, core.make_view(flat['positions'], 64, 0))
            self.assertEqual(set(np.unique(tie['owners'])) - {-1}, {0})

    def test_wrap_modes_sample_texture_edges(self):
        """REPEAT, CLAMP and MIRRORED_REPEAT choose different neighbors at the atlas edge."""
        atlas = np.zeros((1, 4, 4), dtype=np.uint8)
        atlas[0, :, 0] = [10, 20, 30, 40]
        uv = np.array([[0.0, .5]])
        results = {}
        for mode in (10497, 33071, 33648):
            source = {'sampler': {'wrapS': mode, 'wrapT': mode}, 'factor': np.ones(4)}
            results[mode] = core._texture_rgb(source, uv, atlas)[0, 0]
        self.assertAlmostEqual(results[10497], 25)
        self.assertAlmostEqual(results[33071], 10)
        self.assertAlmostEqual(results[33648], 10)
        with self.assertRaises(core.FaceTextureError):
            core._texture_rgb({'sampler': {'wrapS': 1234}, 'factor': np.ones(4)}, uv, atlas)

    def test_registration_residual_excludes_exact_fit_anchors(self):
        """Exactly fitted anchors must not dilute the independent landmark residual."""
        rng = np.random.default_rng(7)
        source = np.c_[rng.uniform(.3, .7, (468, 2)), np.zeros(468)]
        source[33, :2], source[263, :2], source[152, :2] = (.35, .4), (.65, .4), (.5, .85)
        independent = [index for index in core.STABLE if index not in (33, 263, 152)]
        shape = (100, 100, 3)
        moved = source.copy()
        moved[independent, 0] += .1  # 10 pixels = 33% of the 30-pixel eye span
        with self.assertRaisesRegex(core.FaceTextureError, 'residual'):
            core.register_faces(source, moved, shape, shape)
        nudged = source.copy()
        nudged[independent, 0] += .05
        _, report = core.register_faces(source, nudged, shape, shape)
        self.assertEqual(report['diagnostic_landmark_indices'], independent)
        self.assertAlmostEqual(report['stable_rms_pixels'], 5)

    def test_outputs_are_published_only_together(self):
        """A failing output write must not leave a partial corrected result behind."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blocked = root / 'proof.json'
            blocked.mkdir()
            with self.assertRaises(OSError):
                core._write_outputs([(root / 'out.glb', b'glb'), (root / 'cmp.png', b'png'), (blocked, b'{}')])
            self.assertEqual(sorted(p.name for p in root.iterdir()), ['proof.json'])
            core._write_outputs([(root / 'out.glb', b'glb'), (root / 'cmp.png', b'png')])
            self.assertEqual((root / 'out.glb').read_bytes(), b'glb')
            self.assertEqual(sorted(p.name for p in root.iterdir()), ['cmp.png', 'out.glb', 'proof.json'])

    def test_unsupported_assets_are_explicitly_rejected(self):
        """Unsupported deformation, compression and external assets must not pass."""
        mutations = [lambda d: d.update(skins=[{}]), lambda d: d.update(animations=[{}]),
                     lambda d: d['images'][0].update(uri='external.png'),
                     lambda d: d['meshes'][0]['primitives'][0].update(extensions={'KHR_draco_mesh_compression': {}}),
                     lambda d: d['bufferViews'][0].update(extensions={'EXT_meshopt_compression': {}}),
                     # Malformed references must fail as FaceTextureError, not KeyError/IndexError.
                     lambda d: d['meshes'][0]['primitives'][0].pop('attributes'),
                     lambda d: d['textures'][0].update(sampler=3),
                     lambda d: d['textures'][0].update(source=-1),
                     lambda d: d['nodes'][0].update(children=[9]),
                     lambda d: d['meshes'][0]['primitives'][0].update(indices=-1),
                     lambda d: d['images'][0].update(bufferView=0)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.glb'
            for change in mutations:
                with self.subTest(change=change):
                    fixture(path, change)
                    with self.assertRaises(core.FaceTextureError):
                        core.read_source(path)


class _Point:
    def __init__(self, x, y):
        self.x, self.y, self.z = x, y, 0.


class _FakeMp:
    class ImageFormat:
        SRGB = 'srgb'

    class Image:
        def __init__(self, image_format, data):
            self.data = data


def _bearded_face(chin_y):
    """468 image-normalised landmarks of a small frontal face; only the chin varies."""
    lm = np.zeros((478, 2))
    angles = np.linspace(0, 2*np.pi, 478, endpoint=False)
    lm[:] = np.c_[.5 + .06*np.cos(angles), .2 + .08*np.sin(angles)]
    for index, xy in {33: (.47, .095), 133: (.49, .097), 362: (.51, .097), 263: (.53, .095),
                      168: (.5, .093), 1: (.5, .116), 61: (.485, .136), 291: (.515, .136),
                      152: (.5, chin_y)}.items():
        lm[index] = xy
    return lm


class _ChinJumpDetector:
    """Full image: chin at the beard tip. Refinement crop: chin 0.43 eye spans higher."""
    def __init__(self, width, height):
        self.size, self.calls = (width, height), 0
        self.initial = _bearded_face(.177)
        self.refined = _bearded_face(.177 - .43*.06)

    def detect(self, image):
        self.calls += 1
        width, height = self.size
        lm = self.initial
        if self.calls > 1:
            x0, y0, x1, y1 = core._face_box(np.c_[self.initial, np.zeros(478)], width, height)
            lm = (self.refined*[width, height] - [x0, y0]) / [x1-x0, y1-y0]
        return type('Result', (), {'face_landmarks': [[_Point(*xy) for xy in lm]]})()


@unittest.skipIf(WORKER_IMPORT_ERROR is not None, "Standalone ComfyUI worker dependencies unavailable: " + str(WORKER_IMPORT_ERROR))
class DetectFaceRefinementTests(unittest.TestCase):
    """Job fd1f499d: a beard hides the chin of a small full-body face, so the
    bbox refinement moves only landmark 152 by 0.43 outer-eye spans."""

    def setUp(self):
        self.rgb = np.full((768, 768, 3), 245, np.uint8)

    def test_strict_detection_still_refuses_implausible_refinement(self):
        with self.assertRaisesRegex(core.FaceTextureError, 'changed the face implausibly'):
            core.detect_face(_ChinJumpDetector(768, 768), _FakeMp, self.rgb, 'Refined source head render')

    def test_locating_detection_keeps_initial_face_when_refinement_disagrees(self):
        detector = _ChinJumpDetector(768, 768)
        lm, box, observation = core.detect_face(detector, _FakeMp, self.rgb, 'Source mesh render',
                                                locate_only=True)
        np.testing.assert_allclose(lm[:, :2], detector.initial)
        self.assertEqual(box, core._face_box(lm, 768, 768))
        self.assertTrue(observation['refinement_rejected'])
        self.assertFalse(observation['refinement_used'])
        self.assertGreater(observation['refinement_max_pixels'], .4*observation['eye_span_pixels'])


_REAL, _REAL2, _GHOST = (200, 40, 40), (40, 160, 40), (40, 40, 200)


def _marker_face(rgb, colour):
    """Patch-normalised landmarks of a frontal face filling the marker's bounding box."""
    ys, xs = np.nonzero((np.abs(rgb.astype(int) - colour) < 12).all(axis=2))
    if len(xs) == 0:
        return None
    height, width = rgb.shape[:2]
    cx, cy = (xs.min()+xs.max()+1)/2/width, (ys.min()+ys.max()+1)/2/height
    rx, ry = (xs.max()-xs.min()+1)/2/width, (ys.max()-ys.min()+1)/2/height
    angles = np.linspace(0, 2*np.pi, 478, endpoint=False)
    lm = np.c_[cx + rx*np.cos(angles), cy + ry*np.sin(angles)]
    for index, (dx, dy) in {33: (-.5, -.3), 133: (-.2, -.28), 362: (.2, -.28), 263: (.5, -.3),
                            168: (0, -.33), 1: (0, 0), 61: (-.25, .4), 291: (.25, .4), 152: (0, .9)}.items():
        lm[index] = (cx + dx*rx, cy + dy*ry)
    return lm


class _PareidoliaDetector:
    """A real face is found in every view that shows it. The ghost (body texture)
    is reported only as a small part of a larger view, never in its own face crop."""

    def detect(self, image):
        faces = [lm for lm in (_marker_face(image.data, colour) for colour in (_REAL, _REAL2)) if lm is not None]
        ghost = _marker_face(image.data, _GHOST)
        if ghost is not None and np.ptp(ghost[:, 0]) < .1:
            faces.append(ghost)
        return type('Result', (), {'face_landmarks': [[_Point(*xy) for xy in lm] for lm in faces]})()


@unittest.skipIf(WORKER_IMPORT_ERROR is not None, "Standalone ComfyUI worker dependencies unavailable: " + str(WORKER_IMPORT_ERROR))
class DetectFaceConfirmationTests(unittest.TestCase):
    """Job a030ad58: a search crop of a full-body render reported the small head
    and a second "face" on the chest, which vanishes in its own face crop."""

    def render(self, *markers):
        rgb = np.full((768, 768, 3), 245, np.uint8)
        for colour, (x, y) in markers:
            rgb[y:y+40, x:x+40] = colour
        return rgb

    def detect(self, rgb, label='Source mesh render', **options):
        return core.detect_face(_PareidoliaDetector(), _FakeMp, rgb, label, **options)

    def test_unconfirmed_body_detection_is_dropped(self):
        rgb = self.render((_REAL, (365, 70)), (_GHOST, (310, 185)))
        lm, box, observation = self.detect(rgb, locate_only=True, mesh_render=True)
        np.testing.assert_allclose(lm[1, :2]*768, (385, 90), atol=1)
        self.assertEqual(box, core._face_box(lm, 768, 768))
        kinds = [attempt['kind'] for attempt in observation['attempts']]
        self.assertIn('full_image_confirmation', kinds)

    def test_confirmation_must_find_the_face_at_the_same_place(self):
        # The ghost's own crop shows the edge of the real head just above it.
        rgb = self.render((_REAL, (365, 70)), (_GHOST, (365, 112)))
        lm, _, _ = self.detect(rgb, 'Refined source head render', mesh_render=True)
        np.testing.assert_allclose(lm[1, :2]*768, (385, 90), atol=1)

    def test_two_confirmed_faces_in_a_render_are_refused(self):
        rgb = self.render((_REAL, (200, 70)), (_REAL2, (500, 70)))
        with self.assertRaisesRegex(core.FaceTextureError, 'multiple faces are unsupported'):
            self.detect(rgb, locate_only=True, mesh_render=True)

    def test_adjacent_faces_seen_in_each_others_crop_are_refused(self):
        rgb = self.render((_REAL, (365, 70)), (_REAL2, (410, 70)))
        with self.assertRaisesRegex(core.FaceTextureError, 'multiple faces are unsupported'):
            self.detect(rgb, 'Refined source head render', mesh_render=True)

    def test_photo_refuses_every_second_face(self):
        rgb = self.render((_REAL, (365, 70)), (_GHOST, (310, 185)))
        with self.assertRaisesRegex(core.FaceTextureError, 'multiple faces are unsupported'):
            self.detect(rgb, 'Reference photo')


def _figure(path, markers, cells=96):
    """A flat +Z figure in [-1, 1]²: small head, narrow neck, wide torso.

    Planar UVs map the square texture over the whole square, so a marker
    painted at model coordinates appears on the figure exactly there.
    """
    n = cells + 1
    xs = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(xs, xs[::-1])
    positions = np.c_[xx.ravel(), yy.ravel(), np.zeros(n * n)].astype('<f4')
    uv = np.c_[(xx.ravel() + 1) / 2, (1 - yy.ravel()) / 2].astype('<f4')
    corner = (np.arange(cells)[:, None] * n + np.arange(cells)[None, :]).ravel()
    cx, cy = positions[corner + n + 1, 0] - 1/cells, positions[corner + n + 1, 1] + 1/cells
    keep = (((np.abs(cx) < .15) & (cy > .6)) | ((np.abs(cx) < .075) & (cy > .44))
            | ((np.abs(cx) < .5) & (cy < .46)))
    corner = corner[keep]
    triangles = np.r_[np.c_[corner, corner + n, corner + 1], np.c_[corner + 1, corner + n, corner + n + 1]].astype('<u4')
    size = 1024
    texture = np.full((size, size, 3), 225, np.uint8)
    for colour, (x0, y0, x1, y1) in markers:
        texture[round((1-y1)/2*size):round((1-y0)/2*size), round((x0+1)/2*size):round((x1+1)/2*size)] = colour
    png = io.BytesIO()
    Image.fromarray(texture).save(png, format='PNG')
    binary = bytearray()
    views = []
    for chunk in (triangles.tobytes(), positions.tobytes(), uv.tobytes(), png.getvalue()):
        binary.extend(b'\0' * (-len(binary) % 4))
        views.append({'buffer': 0, 'byteOffset': len(binary), 'byteLength': len(chunk)})
        binary.extend(chunk)
    binary.extend(b'\0' * (-len(binary) % 4))
    doc = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0]}], 'nodes': [{'mesh': 0}],
           'meshes': [{'primitives': [{'attributes': {'POSITION': 1, 'TEXCOORD_0': 2}, 'indices': 0, 'material': 0}]}],
           'buffers': [{'byteLength': len(binary)}], 'bufferViews': views,
           'accessors': [{'bufferView': 0, 'componentType': 5125, 'count': triangles.size, 'type': 'SCALAR'},
                         {'bufferView': 1, 'componentType': 5126, 'count': n * n, 'type': 'VEC3'},
                         {'bufferView': 2, 'componentType': 5126, 'count': n * n, 'type': 'VEC2'}],
           'images': [{'bufferView': 3, 'mimeType': 'image/png'}], 'textures': [{'source': 0}],
           'materials': [{'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}}}]}
    payload = json.dumps(doc).encode()
    payload += b' ' * (-len(payload) % 4)
    path.write_bytes(struct.pack('<4sII', b'glTF', 2, 28 + len(payload) + len(binary))
                     + struct.pack('<II', len(payload), 0x4E4F534A) + payload
                     + struct.pack('<II', len(binary), 0x004E4942) + binary)


class _SmallHeadDetector:
    """Like MediaPipe on a full-body render: a marker narrower than `minimum`
    of the view is missed, so only the large body "face" is reported there."""

    def __init__(self, minimum=.12):
        self.minimum = minimum

    def detect(self, image):
        faces = [lm for lm in (_marker_face(image.data, colour) for colour in (_REAL, _GHOST))
                 if lm is not None and np.ptp(lm[:, 0]) >= self.minimum]
        return type('Result', (), {'face_landmarks': [[_Point(*xy) for xy in lm] for lm in faces]})()


_HEAD_FACE = (_REAL, (-.08, .7, .08, .9))
_CHEST_GHOST = (_GHOST, (-.3, -.3, .3, .1))


@unittest.skipIf(WORKER_IMPORT_ERROR is not None, "Standalone ComfyUI worker dependencies unavailable: " + str(WORKER_IMPORT_ERROR))
class HeadPriorTests(unittest.TestCase):
    """Job d2d91062: on a full-body render the small head was missed and the
    chest (nipples as eyes) was the only "face", so the chest got the photo."""

    def observe(self, markers, detector):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'figure.glb'
            _figure(path, markers)
            source = core.read_source(path)
        return core.observe_source(detector, _FakeMp, source, 768, 0)

    def test_shoulder_row_is_found_below_the_head(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'figure.glb'
            _figure(path, [])
            source = core.read_source(path)
        view = core.make_view(source['positions'], 768, 0)
        head = core.head_window(core.render_albedo(source, view)['owners'])
        # Model y .46 (top of the torso) in a 2.12-unit-high view.
        self.assertAlmostEqual(head['shoulder_row'], (1.06 - .46) / 2.12 * 768, delta=6)

    def test_head_only_mesh_has_no_shoulder_row(self):
        owners = np.full((64, 64), -1)
        owners[8:56, 16:48] = 0
        self.assertIsNone(core.head_window(owners))

    def test_body_face_below_the_shoulders_is_replaced_by_the_head(self):
        observed = self.observe([_HEAD_FACE, _CHEST_GHOST], _SmallHeadDetector())
        full_observation, view, source_lm = observed[2], observed[3], observed[5]
        self.assertEqual(full_observation['head_window']['reason'], 'face_below_shoulders')
        # The native head render is centred on the head, not on the chest.
        left, bottom, right, top = view['bounds']
        self.assertGreater(bottom, .46)
        nose = (left + source_lm[1, 0]*(right-left), top - source_lm[1, 1]*(top-bottom))
        np.testing.assert_allclose(nose, (0, .8), atol=.02)

    def test_head_window_also_runs_when_the_full_render_finds_nothing(self):
        # Half-size search crops still show the head below 20 % of their width.
        observed = self.observe([_HEAD_FACE], _SmallHeadDetector(minimum=.2))
        self.assertEqual(observed[2]['head_window']['reason'], 'no_face_in_full_render')

    def test_face_above_the_shoulders_keeps_the_full_render_path(self):
        observed = self.observe([_HEAD_FACE], _SmallHeadDetector(minimum=0))
        self.assertNotIn('head_window', observed[2])
        self.assertEqual(observed[2]['attempts'][0]['kind'], 'full_image')

    def test_no_face_on_the_head_is_refused(self):
        with self.assertRaisesRegex(core.FaceTextureError, 'Source mesh render'):
            self.observe([_CHEST_GHOST], _SmallHeadDetector())



class _TwiceDetector(_PareidoliaDetector):
    """MediaPipe sometimes reports one face twice, noses a pixel apart."""

    def detect(self, image):
        result = super().detect(image)
        twins = []
        for face in result.face_landmarks:
            twins += [face, [_Point(p.x + .001, p.y + .001) for p in face]]
        return type('Result', (), {'face_landmarks': twins})()


def _shift(lm, dx, dy):
    lm = np.array(lm, dtype=float)
    lm[:, 0] += dx
    lm[:, 1] += dy
    return lm


@unittest.skipIf(WORKER_IMPORT_ERROR is not None, "Standalone ComfyUI worker dependencies unavailable: " + str(WORKER_IMPORT_ERROR))
class ReviewFindingTests(unittest.TestCase):
    def test_one_face_reported_twice_is_one_face(self):
        rgb = np.full((768, 768, 3), 245, np.uint8)
        rgb[540:580, 365:405] = _REAL
        for label, options in (('Reference photo', {}), ('Source mesh render', {'mesh_render': True, 'locate_only': True})):
            with self.subTest(label=label):
                lm, _, _ = core.detect_face(_TwiceDetector(), _FakeMp, rgb, label, **options)
                np.testing.assert_allclose(lm[1, :2]*768, (385, 560), atol=1)

    def test_locating_detection_accepts_a_small_face(self):
        rgb = np.full((768, 768, 3), 245, np.uint8)
        rgb[100:114, 380:394] = _REAL  # outer-eye span 7 px
        with self.assertRaisesRegex(core.FaceTextureError, 'below 12 pixels'):
            core.detect_face(_PareidoliaDetector(), _FakeMp, rgb, 'Refined source head render', mesh_render=True)
        lm, _, observation = core.detect_face(_PareidoliaDetector(), _FakeMp, rgb, 'Source mesh render',
                                              locate_only=True, mesh_render=True)
        self.assertLess(observation['eye_span_pixels'], 12)

    def test_other_quarter_turns_are_tried_when_the_requested_yaw_fails(self):
        calls = []
        def observe(detector, mp, source, resolution, yaw):
            calls.append(yaw)
            if yaw != 90:
                raise core.FaceTextureError('Source mesh render: no usable face detected at %g' % yaw)
            return 'full_view', 'full_render', {'attempts': []}, 'view', 'before', 'lm', 'box', 'observation'
        source = {'positions': np.array([[0, 0, 0], [1, 2, .5]])}
        with patch.object(core, '_observe_source', observe):
            observed = core.observe_source(None, None, source, 768, 0)
        self.assertEqual(calls, [0, -180, 90])
        self.assertEqual(observed[2]['yaw_fallback']['used_degrees'], 90)
        self.assertEqual([f['yaw_degrees'] for f in observed[2]['yaw_fallback']['failures']], [0, -180])
        calls.clear()
        lying = {'positions': np.array([[0, 0, 0], [1, .5, 2]])}
        with patch.object(core, '_observe_source', lambda *a: observe(*a[:4], 45)), \
                self.assertRaisesRegex(core.FaceTextureError,
                                       r'at 45 \(also tried yaw -180, 90, -90 degrees\); the mesh is longest along Z'):
            core.observe_source(None, None, lying, 768, 0)

    def test_head_re_render_must_show_the_locating_face(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'figure.glb'
            _figure(path, [])
            source = core.read_source(path)
        head = _bearded_face(.177)
        head = np.c_[_shift(head, 0, 0), np.zeros(478)]
        def run(offset):
            def detect(detector, mp, rgb, label, **options):
                if label == 'Source mesh render':
                    return head, core._face_box(head, 768, 768), {'attempts': []}
                # The same model points seen through the re-render camera, moved by `offset`.
                full_view = core.make_view(source['positions'], 768, 0)
                view = core.make_view(source['positions'], 768, 0,
                                      core._crop_bounds(full_view, core._face_box(head, 768, 768)))
                left, bottom, right, top = view['bounds']
                xy = core._model_xy(full_view, head[:, :2])
                seen = np.c_[(xy[:, 0]-left)/(right-left), (top-xy[:, 1])/(top-bottom), np.zeros(478)]
                return _shift(seen, offset, 0), (0, 0, 768, 768), {'attempts': []}
            with patch.object(core, 'detect_face', detect):
                return core._observe_source(None, None, source, 768, 0)
        run(0)
        with self.assertRaisesRegex(core.FaceTextureError, 'shows a different face'):
            run(.3)

    def test_labelled_glb_is_refused_before_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def label(doc):
                doc['meshes'][0]['primitives'][0]['attributes']['_FACE_REGION'] = 1
            fixture(root / 'labelled.glb', label)
            (root / 'face_landmarker.task').write_bytes(b'model')
            with self.assertRaisesRegex(core.FaceTextureError, 'already carries face region labels'):
                core.run_locate(root / 'labelled.glb', root / 'o.glb', root / 'p.png', root / 'p.json', root)

    def test_bad_reference_photo_is_an_input_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'photo.png'
            path.write_bytes(b'not an image')
            with self.assertRaisesRegex(core.FaceTextureError, 'cannot be decoded'):
                core.read_photo(path)
            Image.new('RGB', (8, 8)).save(path)
            with patch.object(core, 'MAX_TEXTURE_PIXELS', 63), self.assertRaisesRegex(core.FaceTextureError, 'exceeds'):
                core.read_photo(path)
            self.assertEqual(core.read_photo(path).shape, (8, 8, 3))

    @unittest.skipIf(os.name == 'nt', 'POSIX file modes')
    def test_outputs_get_umask_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'out.bin'
            core._write_outputs([(path, b'x')])
            umask = os.umask(0)
            os.umask(umask)
            self.assertEqual(path.stat().st_mode & 0o777, 0o666 & ~umask)

    def test_oversized_triangle_is_rasterised_in_bands(self):
        triangles = np.array([[[0, 0], [700, 10], [30, 650]], [[5, 5], [9, 5], [5, 9]]], dtype=float)
        whole = [np.concatenate(c) for c in zip(*core._raster_batch(triangles, 720, 720, .5))]
        parts = list(core._raster_batch(triangles, 720, 720, .5, budget=4096))
        self.assertGreater(len(parts), 50)
        self.assertLessEqual(max(len(part[0]) for part in parts), 4096)
        for a, b in zip(whole, (np.concatenate(c) for c in zip(*parts))):
            np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
