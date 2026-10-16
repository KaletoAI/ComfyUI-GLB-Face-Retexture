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

    def test_tone_ring_half_and_fallback(self):
        photo, mask, lm, _, _ = synthetic()
        photo[:] = core._linear_to_srgb(np.array([.2,.25,.3]))
        target = np.full_like(photo, core._linear_to_srgb(np.array([.3,.3,.24])))
        target[core._oval_region(lm, mask.shape)] = core._linear_to_srgb(np.array([.1,.12,.15]))
        maps = {'skin': np.ones_like(mask), 'hair': np.zeros_like(mask)}
        # A non-skin corner and hair strip must never contaminate ring statistics.
        target[:45] = 255
        maps['hair'][:45] = 1
        for strength in (.5, 1):
            adjusted, report = core.adjust_photo(photo, mask, lm, tone_match=strength,
                source_rgb=target, source_mask=mask, source_lm=lm, source_box=(0,0,256,256), source_maps=maps)
            median = np.median(core._srgb_to_linear(adjusted)[mask>.5], axis=0)
            expected = np.array([.2,.25,.3])**(1-strength)*np.array([.3,.3,.24])**strength
            np.testing.assert_allclose(median, expected, rtol=.03)
            self.assertEqual(report['tone']['target_region'], 'ring')
            print('synthetic tone', strength, median.tolist())
        maps['skin'][:] = 0
        _, report = core.adjust_photo(photo, mask, lm, tone_match=1, source_rgb=target,
            source_mask=mask, source_lm=lm, source_box=(0,0,256,256), source_maps=maps)
        self.assertEqual(report['tone']['target_region'], 'source_face')

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
        arguments += ['--light-balance', '.4', '--tone-match', '.6', '--brightness', '.8']
        with patch.object(sys, 'argv', arguments), patch.object(worker, 'worker_python'), \
                patch.object(worker, 'validate_worker_identity'), patch.dict(sys.modules, {'face_core': fake_core}):
            self.assertEqual(worker.main(), 0)
        self.assertEqual([captured[key] for key in ('light_balance','tone_match','brightness')], [.4,.6,.8])

    def test_brightness_and_validation(self):
        photo, mask, lm, _, _ = synthetic()
        adjusted, _ = core.adjust_photo(photo, mask, lm, brightness=.5)
        np.testing.assert_allclose(core._srgb_to_linear(adjusted), core._srgb_to_linear(photo)*.5, rtol=1e-5)
        for name, values in [('light_balance', [-.1,1.1,float('nan')]), ('tone_match', [-.1,1.1,float('inf')]), ('brightness',[.49,1.51,float('nan')])]:
            for value in values:
                with self.subTest(name=name,value=value), self.assertRaisesRegex(core.FaceTextureError, name):
                    core.run_retexture(None,None,None,None,None,None, **{name:value})
