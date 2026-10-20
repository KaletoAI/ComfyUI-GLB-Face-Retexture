"""Guard append-only geometry and private-atlas isolation without ML models."""
import importlib.util
import io
import json
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


def encoded(image, format='WEBP', **kwargs):
    output = io.BytesIO()
    image.save(output, format=format, **kwargs)
    return output.getvalue()


def pixels():
    atlas = np.full((32, 32, 4), [20, 40, 60, 255], dtype=np.uint8)
    atlas[3, 7, 3] = 200
    return atlas


def webp_blobs():
    return [encoded(Image.fromarray(pixels()), lossless=True, exact=True),
            encoded(Image.new('RGB', (32, 32), (0, 128, 255)), lossless=True, exact=True)]


def fixture(path, change=None, blobs=None):
    """Two separately indexed triangles intentionally share every UV texel."""
    positions = np.array([[-.8, -.8, 1], [.8, -.8, 1], [0, .8, 1],
                          [-.8, -.8, 0], [.8, -.8, 0], [0, .8, 0]], dtype='<f4')
    uv = np.array([[.1, .1], [.9, .1], [.5, .9]] * 2, dtype='<f4')
    normals = np.array([[0, 0, 1]] * 6, dtype='<f4')
    indices = np.arange(6, dtype='<u4')
    blobs = webp_blobs() if blobs is None else blobs
    chunks = [indices.tobytes(), positions.tobytes(), uv.tobytes(), normals.tobytes(), *blobs]
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
    doc['images'] = [{'bufferView': 4+i, 'mimeType': 'image/webp'} for i in range(len(blobs))]
    doc['textures'] = [{'extensions': {'EXT_texture_webp': {'source': i}}} for i in range(2)]
    doc['extensionsUsed'] = ['EXT_texture_webp']
    doc['extensionsRequired'] = ['EXT_texture_webp']
    doc['materials'][0]['pbrMetallicRoughness'].update(baseColorFactor=[1, 1, 1, 1], metallicRoughnessTexture={'index': 1}, metallicFactor=1, roughnessFactor=1)
    doc['materials'][0].update(alphaMode='OPAQUE', doubleSided=False)
    if change:
        change(doc)
    payload = json.dumps(doc).encode()
    payload += b' ' * (-len(payload) % 4)
    binary.extend(b'\0' * (-len(binary) % 4))
    raw = struct.pack('<4sII', b'glTF', 2, 28 + len(payload) + len(binary)) + struct.pack('<II', len(payload), 0x4E4F534A) + payload + struct.pack('<II', len(binary), 0x004E4942) + binary
    path.write_bytes(raw)
    return doc, bytes(binary), blobs


@unittest.skipIf(WORKER_IMPORT_ERROR is not None, "Standalone ComfyUI worker dependencies unavailable: " + str(WORKER_IMPORT_ERROR))
class WebPTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'source.glb'

    def test_read_pixels_and_raw_blobs(self):
        blobs = webp_blobs()
        fixture(self.path, blobs=blobs)
        source = core.read_source(self.path)
        np.testing.assert_array_equal(source['atlas'], pixels())
        self.assertEqual(source['images'], blobs)

    def test_partition_png_and_preserved_webp(self):
        original, binary, _ = fixture(self.path)
        source = core.read_source(self.path)
        painted = np.full((32, 32, 4), [100, 150, 200, 255], dtype=np.uint8)
        png = encoded(Image.fromarray(painted), 'PNG')
        for recolor in (False, True):
            with self.subTest(recolor=recolor):
                doc, emitted = core.parse_glb(core.partition_glb(source, np.array([True, False]), png, png if recolor else None))
                self.assertEqual(doc['images'][:2], original['images'])
                self.assertEqual(emitted[:len(binary)], binary)
                self.assertEqual(doc['extensionsRequired'], ['EXT_texture_webp'])
                self.assertEqual(doc['extensionsUsed'], ['EXT_texture_webp'])
                self.assertEqual(doc['textures'][:2], original['textures'])
                body, face = doc['meshes'][0]['primitives']
                if not recolor:
                    self.assertEqual(body['material'], 0)
                    self.assertEqual(doc['materials'][0]['pbrMetallicRoughness']['baseColorTexture']['index'], 0)
                for primitive in (body, face):
                    pbr = doc['materials'][primitive['material']]['pbrMetallicRoughness']
                    self.assertEqual(pbr['metallicRoughnessTexture']['index'], 1)
                    texture = doc['textures'][pbr['baseColorTexture']['index']]
                    if primitive is face or recolor:
                        self.assertNotIn('extensions', texture)
                        self.assertEqual(doc['images'][texture['source']]['mimeType'], 'image/png')
                    atlas, _ = core._material_atlas(doc, emitted, primitive['material'])
                    np.testing.assert_array_equal(atlas, painted if primitive is face or recolor else pixels())

    def test_webp_preferred_over_png_fallback(self):
        png = encoded(Image.new('RGBA', (32, 32), (200, 10, 30, 255)), 'PNG')
        def change(d):
            d['images'][2]['mimeType'] = 'image/png'
            d['textures'][0]['source'] = 2
        fixture(self.path, change, webp_blobs()+[png])
        np.testing.assert_array_equal(core.read_source(self.path)['atlas'], pixels())

    def test_jpeg_core_source(self):
        jpeg = encoded(Image.new('RGB', (32, 32), (70, 90, 110)), 'JPEG')
        def change(d):
            d['images'][0]['mimeType'] = 'image/jpeg'
            d['textures'][0] = {'source': 0}
        fixture(self.path, change, [jpeg, webp_blobs()[1]])
        with Image.open(io.BytesIO(jpeg)) as image:
            expected = np.array(image.convert('RGBA'))
        np.testing.assert_array_equal(core.read_source(self.path)['atlas'], expected)

    def test_invalid_references_and_extensions(self):
        changes = [lambda d: d.pop('extensionsUsed'),
                   lambda d: d.update(extensionsUsed=[]),
                   lambda d: d.update(extensionsRequired=['KHR_texture_basisu']),
                   lambda d: d.update(extensionsRequired=['VENDOR_unknown']),
                   lambda d: d['textures'][1].update(extensions={'KHR_texture_basisu': {'source': 1}}),
                   lambda d: (d['textures'][0].pop('extensions'), d['textures'][0].update(source=0)),
                   lambda d: d['textures'][1]['extensions'].update(EXT_texture_webp=1),
                   lambda d: d.update(extensions={'EXT_texture_webp': {}}),
                   lambda d: d['textures'][0].update(source=1)]
        for index in (-1, True, 9):
            changes.append(lambda d, index=index: d['textures'][1]['extensions']['EXT_texture_webp'].update(source=index))
        for i, change in enumerate(changes):
            with self.subTest(case=i):
                fixture(self.path, change)
                with self.assertRaises(core.FaceTextureError):
                    core.read_source(self.path)

    def test_unknown_required_extension_is_named(self):
        for name in ('KHR_texture_basisu', 'VENDOR_unknown'):
            with self.subTest(name=name):
                fixture(self.path, lambda d: d.update(extensionsRequired=[name]))
                with self.assertRaisesRegex(core.FaceTextureError, name):
                    core.read_source(self.path)

    def test_invalid_image_payloads(self):
        png = encoded(Image.new('RGBA', (32, 32)), 'PNG')
        animated = encoded(Image.new('RGBA', (32, 32), 'red'), save_all=True,
            append_images=[Image.new('RGBA', (32, 32), 'blue')], duration=100, loop=0, lossless=True, exact=True)
        cases = [(png, None), (webp_blobs()[0][:-12], None), (animated, None),
                 (png, lambda d: d['images'][0].update(mimeType='image/png'))]
        for i, (blob, change) in enumerate(cases):
            with self.subTest(case=i):
                fixture(self.path, change, [blob, webp_blobs()[1]])
                with self.assertRaises(core.FaceTextureError):
                    core.read_source(self.path)
                if change is None:
                    fixture(self.path, blobs=[webp_blobs()[0], blob])
                    with self.assertRaises(core.FaceTextureError):
                        core.read_source(self.path)

    def test_header_limits(self):
        fixture(self.path)
        for constant, value in (('MAX_TEXTURE_SIDE', 31), ('MAX_TEXTURE_PIXELS', 1023)):
            with self.subTest(constant=constant), patch.object(core, constant, value):
                with self.assertRaises(core.FaceTextureError):
                    core.read_source(self.path)


if __name__ == '__main__':
    unittest.main()
