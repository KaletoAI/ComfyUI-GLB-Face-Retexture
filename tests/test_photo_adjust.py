"""Linear photo controls, including rolled midlines and asymmetric detail."""
import unittest
from tests.test_face_core import WORKER_IMPORT_ERROR
if WORKER_IMPORT_ERROR is None:
    import numpy as np
    from tests.test_face_core import core


def synthetic():
    yy, xx = np.mgrid[:256, :256]
    mid = 128 + .08*(yy-128)
    dx = xx-mid
    mask = ((dx/90)**2 + ((yy-128)/110)**2 < 1).astype(np.float32)
    lm = np.zeros((468, 3))
    for index, y in zip((10,168,6,1,0,13,14,152), (25,75,95,120,145,150,155,230)):
        lm[index, :2] = [(128+.08*(y-128))/256, y/256]
    for index, angle in zip(core.FACE_OVAL, np.linspace(-np.pi/2, 3*np.pi/2, len(core.FACE_OVAL), endpoint=False)):
        y = 128+110*np.sin(angle)
        lm[index, :2] = [(128+.08*(y-128)+90*np.cos(angle))/256, y/256]
    lm[33, :2], lm[263, :2] = [.3,.35], [.7,.35]
    linear = np.repeat((.26*(1+.45*dx/90))[...,None], 3, axis=2).astype(np.float32)
    mole = (xx-88)**2+(yy-130)**2 < 9
    linear[mole] *= .4
    photo = core._linear_to_srgb(linear).astype(np.float32)
    return photo, mask, lm, dx, mole


@unittest.skipIf(WORKER_IMPORT_ERROR is not None, "Standalone worker dependencies unavailable")
class PhotoAdjustTests(unittest.TestCase):
    def test_balance_preserves_detail_and_roll(self):
        """For L/R=r, square-root reciprocal gains give r**(1-strength)."""
        photo, mask, lm, dx, mole = synthetic()
        results = []
        for strength in (0, .5, 1):
            adjusted, report = core.adjust_photo(photo, mask, lm, light_balance=strength)
            linear = core._srgb_to_linear(adjusted)
            patches = (mask > .5) & (np.abs(dx) > 30) & (np.abs(dx) < 60) & (np.abs(np.indices(mask.shape)[0]-128) < 30)
            ratio = linear[patches & (dx<0)].mean()/linear[patches & (dx>0)].mean()
            results.append(ratio)
            np.testing.assert_array_equal(adjusted[mask==0], photo[mask==0])
            original = core._srgb_to_linear(photo)
            local = (np.indices(mask.shape)[1]-88)**2+(np.indices(mask.shape)[0]-130)**2
            surround = (local>16)&(local<49)
            contrast = linear[mole].mean()/linear[surround].mean()
            initial = original[mole].mean()/original[surround].mean()
            self.assertLess(abs(contrast/initial-1), .1)
            if strength == 1:
                print('synthetic mole contrast before/after:', float(initial), float(contrast))
            if strength:
                self.assertAlmostEqual(report['midline']['a'], .08, places=5)
        self.assertLess(abs(results[2]-1), .05)
        self.assertLess(abs(results[1]-np.sqrt(results[0])), .03)
        print('synthetic cheek L/R (off, half, full):', results)

    def test_balance_removes_edge_falloff(self):
        """A face darker toward its edge (the cheeks turning away from the camera)
        is evened out. Shading 1 - .7 r**2: rim r in (.75, .9) has mean r**2 ~ .68, so
        ~ .52; the centre r < .3 ~ .97; rim/core ~ .54 before, within 8 % of 1 at full
        strength."""
        photo, mask, lm, dx, mole = synthetic()
        yy, xx = np.indices(mask.shape)
        r = np.sqrt((dx/90)**2 + ((yy-128)/110)**2)
        linear = np.repeat((.3*(1-.7*r**2))[..., None], 3, axis=2).astype(np.float32)
        linear[mole] *= .4
        photo = core._linear_to_srgb(linear).astype(np.float32)
        rim, centre = (mask > .5) & (r > .75) & (r < .9) & ~mole, (mask > .5) & (r < .3)
        def rim_ratio(img):
            lum = core._srgb_to_linear(img).mean(axis=2)
            return lum[rim].mean()/lum[centre].mean()
        adjusted, report = core.adjust_photo(photo, mask, lm, light_balance=1)
        self.assertLess(rim_ratio(photo), .6)
        self.assertLess(abs(rim_ratio(adjusted)-1), .08)
        self.assertGreater(report['shading_p10_p90_after'], report['shading_p10_p90_before'])
        print('synthetic rim/core before/after:', rim_ratio(photo), rim_ratio(adjusted))

    def test_tone_matches_source_face_skin(self):
        """Target = median of the mesh's own face skin (source_mask > .5); the
        darker neck/hairline around it (here: everything outside the mask) must
        not pull the tone. tone t gives photo**(1-t) * target**t per channel."""
        photo, mask, lm, _, _ = synthetic()
        photo[:] = core._linear_to_srgb(np.array([.2,.25,.3]))
        target = np.full_like(photo, core._linear_to_srgb(np.array([.05,.04,.03])))
        target[mask > .5] = core._linear_to_srgb(np.array([.3,.3,.24]))
        for strength in (.5, 1):
            adjusted, report = core.adjust_photo(photo, mask, lm, tone_match=strength,
                source_rgb=target, source_mask=mask)
            median = np.median(core._srgb_to_linear(adjusted)[mask>.5], axis=0)
            expected = np.array([.2,.25,.3])**(1-strength)*np.array([.3,.3,.24])**strength
            np.testing.assert_allclose(median, expected, rtol=.03)
            self.assertEqual(report['tone']['target_region'], 'source_face')
            print('synthetic tone', strength, median.tolist())
        with self.assertRaises(core.FaceTextureError):
            core.adjust_photo(photo, mask, lm, tone_match=1, source_rgb=target, source_mask=np.zeros_like(mask))

    def test_worker_cli_passes_controls(self):
        import sys
        from unittest.mock import patch
        import worker
        import types
        fake_core = types.SimpleNamespace(run_retexture=lambda **kw: captured.update(kw))
        captured = {}
        arguments = ['worker.py']
        for flag in ('source-glb','reference-image','output-glb','comparison-png','proof-json','model-dir'):
            arguments += ['--'+flag, '/tmp/unused']
        arguments += ['--light-balance', '.4', '--tone-match', '.6', '--brightness', '.8', '--skin-recolor', '.7']
        with patch.object(sys, 'argv', arguments), patch.object(worker, 'worker_python'), \
                patch.object(worker, 'validate_worker_identity'), patch.dict(sys.modules, {'face_core': fake_core}):
            self.assertEqual(worker.main(), 0)
        self.assertEqual([captured[key] for key in ('light_balance','tone_match','brightness','skin_recolor')], [.4,.6,.8,.7])

    def test_brightness_and_validation(self):
        photo, mask, lm, _, _ = synthetic()
        adjusted, _ = core.adjust_photo(photo, mask, lm, brightness=.5)
        np.testing.assert_allclose(core._srgb_to_linear(adjusted), core._srgb_to_linear(photo)*.5, rtol=1e-5)
        for name, values in [('light_balance', [-.1,1.1,float('nan')]), ('tone_match', [-.1,1.1,float('inf')]), ('brightness',[.49,1.51,float('nan')]), ('skin_recolor',[-.1,1.1,float('nan'),float('inf')])]:
            for value in values:
                with self.subTest(name=name,value=value), self.assertRaisesRegex(core.FaceTextureError, name):
                    core.run_retexture(None,None,None,None,None,None, **{name:value})
