import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location('ideogram_refiner', ROOT / 'lib' / 'ideogram_refiner.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class FakeDetect:
    def __init__(self, headers): self.headers = headers
    def read_safetensors_header(self, path): return self.headers[Path(path).name]
    def _looks_like_qwen3vl_te(self, h): return h.get('owner_te', False)
    def _looks_like_flux2_vae(self, h): return h.get('owner_vae', False)
    def te_vae_search_dirs(self, roots): return roots


def te_header():
    return {'owner_te': True, 'model.embed_tokens.weight': {},
            'model.layers.35.input_layernorm.weight': {}}


def vae_header(native=True):
    h = {'owner_vae': True, 'encoder.quant_conv.weight': {},
         'decoder.post_quant_conv.weight': {}}
    if native:
        h.update({'encoder.down.0.block.0.conv1.weight': {},
                  'decoder.up.0.block.0.conv1.weight': {}})
    return h


class IdeogramModuleTests(unittest.TestCase):
    def test_preferred_valid_pair_wins_and_is_ordered_vae_then_te(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); te = root / 'te.safetensors'; vae = root / 'vae.safetensors'
            te.touch(); vae.touch()
            detect = FakeDetect({te.name:te_header(), vae.name:vae_header()})
            result = MOD.resolve_modules('checkpoint', preferred=[te, vae], detect=detect,
                                         roots=[root], autolink=())
            self.assertEqual(result, [str(vae.resolve()), str(te.resolve())])

    def test_rejects_diffusers_vae_and_finds_deterministic_native_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root / 'Ideogram4-wrong.safetensors'
            vae = root / 'FLUX.2-VAE--For-Ideogram4.safetensors'
            te = root / 'Qwen3-VL-8B--For-Ideogram4.safetensors'
            for path in (bad, vae, te): path.touch()
            detect = FakeDetect({bad.name:vae_header(False), vae.name:vae_header(),
                                 te.name:te_header()})
            result = MOD.resolve_modules('checkpoint', preferred=[bad], detect=detect,
                                         roots=[root], autolink=())
            self.assertEqual(result, [str(vae.resolve()), str(te.resolve())])

    def test_incomplete_encoder_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); te = root / 'te.safetensors'; vae = root / 'vae.safetensors'
            te.touch(); vae.touch()
            detect = FakeDetect({te.name:{'owner_te':True}, vae.name:vae_header()})
            with self.assertRaisesRegex(ValueError, 'complete Qwen3-VL-8B'):
                MOD.resolve_modules(detect=detect, roots=[root], autolink=())

    def test_guidance_auto_prefers_matching_full_and_preserves_explicit_lora(self):
        detect = types.SimpleNamespace(
            model_search_dirs=lambda roots:['checkpoints'],
            find_matching_uncond=lambda checkpoint, dirs:'matching-uncond.safetensors')
        lora = types.SimpleNamespace(
            split_lora_tags=lambda prompt, roots: ('scene', [('mine', 1, 'uncond_replace')])
            if '<lora:mine:1>' in prompt else (prompt, []))
        importer = lambda name: detect if name.endswith('.detect') else lora
        with patch.object(MOD, 'import_module', side_effect=importer), \
             patch.object(MOD, '_model_roots', return_value=['models']):
            self.assertEqual(MOD.guidance_prompt('cond', 'scene', 4, 'Auto'),
                             ('scene', 'full transformer (matching-uncond.safetensors)'))
            self.assertEqual(MOD.guidance_prompt('cond', 'scene <lora:mine:1>', 4, 'Auto')[0],
                             'scene <lora:mine:1>')

    def test_guidance_auto_falls_back_to_verified_private_ostris_tag(self):
        adapter = Path('Ostris-Uncond.safetensors')
        detect = types.SimpleNamespace(model_search_dirs=lambda roots:['checkpoints'],
                                       find_matching_uncond=lambda checkpoint, dirs:None)
        def split(prompt, roots):
            return ('', [('Ostris-Uncond', 1.0, 'uncond_replace')]) if '<lora:' in prompt else (prompt, [])
        lora = types.SimpleNamespace(split_lora_tags=split,
                                     find_uncond_replacement_lora=lambda roots:adapter)
        importer = lambda name: detect if name.endswith('.detect') else lora
        with patch.object(MOD, 'import_module', side_effect=importer), \
             patch.object(MOD, '_model_roots', return_value=['models']):
            result, label = MOD.guidance_prompt('cond', 'scene', 4, 'Auto')
        self.assertEqual(result, 'scene <lora:Ostris-Uncond:1>')
        self.assertIn('Ostris uncond LoRA', label)


if __name__ == '__main__':
    unittest.main()
