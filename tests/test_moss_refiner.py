import importlib.util
import types
import unittest
from unittest.mock import patch
from pathlib import Path


PATH = Path(__file__).parents[1] / 'lib' / 'moss_refiner.py'
SPEC = importlib.util.spec_from_file_location('moss_refiner_test', PATH)
MOSS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOSS)


class Header:
    def __init__(self, keys): self._keys = keys
    def keys(self): return self._keys


class MossRefinerTests(unittest.TestCase):
    moss_modules = ['moss-te.safetensors', 'flux2-vae.safetensors']

    @staticmethod
    def component_header(path):
        if 'moss-te' in path:
            return Header(['model.language_model.embed_tokens.weight',
                           'model.language_model.layers.0.linear_attn.A_log'])
        if 'flux2-vae' in path:
            return Header(['bn.running_mean', 'encoder.quant_conv.weight',
                           'decoder.post_quant_conv.weight'])
        return Header(['foreign.weight'])

    def test_identifies_bf16_and_quantized_moss_structures(self):
        required = ['shared_block_modulation.weight', 'layers.0.proj.ffn_up_gate.weight']
        self.assertTrue(MOSS.identify('bf16', lambda _: Header(required)))
        self.assertTrue(MOSS.identify('int8', lambda _: Header(required + ['layers.0.proj.ffn_up_gate.comfy_quant'])))

    def test_rejects_krea_or_unreadable_checkpoint(self):
        self.assertFalse(MOSS.identify('krea', lambda _: Header(['transformer_blocks.0.attn.to_q.weight'])))
        def broken(_): raise OSError('bad header')
        self.assertFalse(MOSS.identify('bad', broken))

    def test_stale_scheduler_falls_back_to_current_native_default(self):
        profile = {'family': 'moss_image', 'scheduler': 'MOSS'}
        self.assertEqual(MOSS.normalize_profile(profile, {'Simple': object()})['scheduler'], 'Simple')
        self.assertTrue(profile['scheduler_fallback'])

    def test_valid_scheduler_and_non_moss_are_unchanged(self):
        profile = {'family': 'moss_image', 'scheduler': 'SGM Uniform'}
        self.assertIs(MOSS.normalize_profile(profile, {'SGM Uniform': object()}), profile)
        other = {'family': 'krea2', 'scheduler': 'Simple'}
        self.assertIs(MOSS.normalize_profile(other, {}), other)

    def test_installed_moss_component_structures_are_accepted(self):
        profile = {'family': 'moss_image', 'modules': self.moss_modules}
        MOSS.validate_modules(profile, self.component_header)

    def test_foreign_or_incomplete_components_are_rejected(self):
        for modules in (['moss-te.safetensors'],
                        ['moss-te.safetensors', 'krea-vae.safetensors'],
                        ['qwen-vl.safetensors', 'flux2-vae.safetensors']):
            with self.subTest(modules=modules), self.assertRaisesRegex(ValueError, 'requires exactly|incompatible'):
                MOSS.validate_modules({'family': 'moss_image', 'modules': modules}, self.component_header)

    def test_native_core_backend_is_accepted(self):
        seen = []
        def importer(name):
            seen.append(name)
            return types.SimpleNamespace(MOSSImage=type('MOSSImage', (), {}))
        profile = {'family': 'moss_image'}
        with patch.object(MOSS, 'validate_modules'):
            MOSS.validate(profile, importer)
        self.assertEqual(seen, ['backend.diffusion_engine.moss_image'])

    def test_missing_or_incomplete_core_backend_fails_clearly(self):
        def missing(_): raise ImportError('missing')
        with self.assertRaisesRegex(ValueError, 'update Forge'):
            MOSS.validate({'family': 'moss_image'}, missing)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            MOSS.validate({'family': 'moss_image'}, lambda _: types.SimpleNamespace())

    def test_non_moss_is_untouched(self):
        def importer(_): self.fail('non-MOSS must not probe MOSS')
        MOSS.validate({'family': 'qwen_image21'}, importer)
        job = types.SimpleNamespace(override_settings={}, extra_generation_params={})
        MOSS.configure_job(job, {'family': 'qwen_image21'})
        self.assertEqual(job.override_settings, {})
        self.assertEqual(job.extra_generation_params, {})

    def test_moss_forces_conventional_img2img_locally(self):
        job = types.SimpleNamespace(
            override_settings={'img2img_fix_steps': True, 'moss_do_reference': 'Moss-Image-Edit'},
            extra_generation_params={})
        MOSS.configure_job(job, {'family': 'moss_image'})
        self.assertEqual(job.override_settings['moss_do_reference'], 'Disable')
        self.assertIn('conventional img2img', job.extra_generation_params['MOSS refiner input mode'])


if __name__ == '__main__':
    unittest.main()
