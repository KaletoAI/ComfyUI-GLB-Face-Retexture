"""Guard append-only geometry and private-atlas isolation without ML models."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

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


if __name__ == '__main__':
    unittest.main()
