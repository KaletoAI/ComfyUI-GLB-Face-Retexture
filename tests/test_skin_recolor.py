"""Colour-only body correction (luminance kept) and append-only atlas export."""
import io
from pathlib import Path
import tempfile
import unittest
from tests.test_face_core import WORKER_IMPORT_ERROR
if WORKER_IMPORT_ERROR is None:
    import numpy as np
    from PIL import Image
    from tests.test_face_core import core, fixture


@unittest.skipIf(WORKER_IMPORT_ERROR is not None, 'Standalone worker dependencies unavailable')
class SkinRecolorTests(unittest.TestCase):
    def test_skin_gain_preserves_cloth_hair_alpha_and_factor(self):
        for factor in (np.ones(3), np.full(3, .6), np.array([.5, .8, 1.])):
            atlas = np.empty((32, 32, 4), dtype=np.uint8)
            atlas[:] = [190, 120, 100, 255]
            atlas[:, 16:24] = [125, 115, 80, 255]
            atlas[:, 24:] = [48, 27, 20, 255]
            atlas[0, :16, 3] = 254
            source = dict(atlas=atlas, factor=np.r_[factor, 1])
            skin = core._linear_to_srgb(core._srgb_to_linear(atlas[:1, :16, :3])*factor)
            full = dict(rgb=skin, owners=np.zeros((1,16), dtype=int))
            target = np.array([.38, .24, .17])*factor
            photo = np.broadcast_to(core._linear_to_srgb(target), skin.shape)
            mask = np.ones((1,16))
            for strength in (0, .5, 1):
                recolored, report = core.recolor_skin(source, full, mask, mask*0,
                    skin, mask, photo, mask, strength)
                # Colour only: the chroma (R/G, B/G) moves geometrically toward the
                # photo, the texel keeps its own luminance.
                original = core._srgb_to_linear([190,120,100])*factor
                expected = original**(1-strength)*target**strength
                observed = np.median(core._srgb_to_linear(recolored[1:,:16,:3])*factor, axis=(0,1))
                lum = np.array([.2126,.7152,.0722])
                np.testing.assert_allclose(observed[[0,2]]/observed[1], expected[[0,2]]/expected[1], rtol=.03)
                np.testing.assert_allclose(observed @ lum, original @ lum, rtol=.03)
                np.testing.assert_array_equal(recolored[:,16:], atlas[:,16:])
                np.testing.assert_array_equal(recolored[0], atlas[0])
                if strength:
                    self.assertEqual(report['texels_weighted'], 31*16)
                    self.assertEqual(report['texels_changed'], 31*16)
                else:
                    np.testing.assert_array_equal(recolored, atlas)
                    self.assertEqual(report, {})

    def test_wide_skin_shading_does_not_select_olive_cloth_or_dark_hair(self):
        skin_linear = core._srgb_to_linear([190,120,100])
        samples = core._linear_to_srgb(skin_linear*np.linspace(.15,1.5,100)[:,None])
        full = dict(rgb=samples[None],owners=np.zeros((1,100),dtype=int))
        atlas = np.array([[[125,115,80,255],[48,27,20,255],[190,120,100,255]]],dtype=np.uint8)
        source = dict(atlas=atlas,factor=np.ones(4))
        mask = np.ones((1,100))
        face = np.broadcast_to([190,120,100], (1,100,3))
        photo = core._linear_to_srgb(core._srgb_to_linear(face)*[1,1.5,1.5])
        recolored,_ = core.recolor_skin(source,full,mask,mask*0,face,mask,photo,mask,1)
        np.testing.assert_array_equal(recolored[0,:2],atlas[0,:2])
        self.assertGreater(int(recolored[0,2,1]),int(atlas[0,2,1]))

    def test_recolor_requires_confident_owned_skin(self):
        atlas = np.full((4,4,4), 255, dtype=np.uint8)
        source = dict(atlas=atlas, factor=np.ones(4))
        mask = np.ones((4,4))
        for owners, skin, hair in ((-mask, mask, mask*0), (mask, mask*.6, mask*0), (mask, mask, mask*.1)):
            with self.assertRaisesRegex(core.FaceTextureError, 'confident.*skin'):
                core.recolor_skin(source, dict(rgb=atlas[:,:,:3],owners=owners), skin, hair,
                    atlas[:,:,:3],mask,atlas[:,:,:3],mask,1)

    def test_export_appends_body_atlas_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory)/'source.glb'
            _, _, original_image = fixture(src)
            source = core.read_source(src)
            selected = np.array([True,False])
            body = source['atlas'].copy()
            body[:,:,:3] = [80,90,100]
            buffer = io.BytesIO()
            Image.fromarray(body).save(buffer,format='PNG')
            output = core.partition_glb(source,selected,buffer.getvalue(),body_atlas_png=buffer.getvalue())
            doc,binary = core.parse_glb(output)
            self.assertEqual(binary[:len(source['binary'])],source['binary'])
            view = doc['bufferViews'][doc['images'][0]['bufferView']]
            self.assertEqual(binary[view['byteOffset']:view['byteOffset']+view['byteLength']], original_image)
            self.assertEqual(doc['images'][0],source['gltf']['images'][0])
            primitives = doc['meshes'][0]['primitives']
            body_material = primitives[0]['material']
            texture = doc['materials'][body_material]['pbrMetallicRoughness']['baseColorTexture']['index']
            self.assertGreater(doc['textures'][texture]['source'],0)
            atlas,_ = core._material_atlas(doc,binary,body_material)
            np.testing.assert_array_equal(atlas,body)
            for primitive,indices in zip(primitives,(source['triangles'][~selected],source['triangles'][selected])):
                self.assertEqual(primitive['attributes'],source['primitive']['attributes'])
                np.testing.assert_array_equal(core.accessor_array(doc,binary,primitive['indices']).reshape(-1,3),indices)
            view = core.make_view(source['positions'],64,180)
            expected = core.render_albedo(dict(source,atlas=body),view)['rgb']
            actual = core.render_albedo(source,view,atlases=[atlas,atlas],assignment=selected.astype(int))['rgb']
            self.assertEqual(int(np.any(expected!=actual,axis=2).sum()),0)
