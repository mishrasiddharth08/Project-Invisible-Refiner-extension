import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('final_lora_test',
    Path(__file__).parents[1] / 'lib' / 'refiner_lora.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class FinalLoraTests(unittest.TestCase):
    def test_strips_all_adapter_kinds_but_preserves_other_prompt_tags(self):
        text = 'person <LoRA:base:1> <lokr:other:.8> <lyco:third:1> <lycoris:fourth:1> <embedding:keep>'
        self.assertEqual(helper.strip_tags(text), 'person     <embedding:keep>')
        self.assertEqual(helper.strip_tags([text, 'plain']), [helper.strip_tags(text), 'plain'])

    def test_same_model_native_patches_and_prompt_cache_are_cleared(self):
        base = types.SimpleNamespace(clear_prompt_cache=Mock())
        job = types.SimpleNamespace(clear_prompt_cache=Mock(), disable_extra_networks=True)
        shared = types.SimpleNamespace(opts=types.SimpleNamespace(data={'sd_lora':'base-auto'}))
        model = types.SimpleNamespace(forge_objects_original=object())
        sd_models = types.SimpleNamespace(model_data=types.SimpleNamespace(sd_model=model))
        networks = types.SimpleNamespace(load_networks=Mock())
        with patch.dict(sys.modules, {'networks':networks}):
            with self.assertRaisesRegex(RuntimeError, 'generation failed'):
                with helper.final_pass(base, job, shared, sd_models):
                    networks.load_networks.assert_called_once_with([], [], [])
                    self.assertFalse(job.disable_extra_networks)
                    self.assertEqual(shared.opts.data['sd_lora'], 'None')
                    raise RuntimeError('generation failed')
        self.assertEqual(shared.opts.data['sd_lora'], 'base-auto')
        base.clear_prompt_cache.assert_called_once()
        job.clear_prompt_cache.assert_called_once()

    def test_unloaded_target_does_not_invoke_native_lora_loader(self):
        shared = types.SimpleNamespace(opts=types.SimpleNamespace(data={}))
        networks = types.SimpleNamespace(load_networks=Mock())
        with patch.dict(sys.modules, {'networks':networks}):
            with helper.final_pass(object(), types.SimpleNamespace(), shared,
                                   types.SimpleNamespace(model_data=types.SimpleNamespace(sd_model=object()))):
                self.assertEqual(shared.opts.data['sd_lora'], 'None')
        networks.load_networks.assert_not_called()
        self.assertNotIn('sd_lora', shared.opts.data)

    def test_same_checkpoint_external_cache_released_once_per_batch(self):
        from test_engine import Tests
        fixture = Tests()
        fixture.setUp()
        engine = fixture.engine
        info = object()
        engine._preflight = lambda *args: info
        engine._target_profile = lambda *args: {}
        fixture.modules.sd_models.get_closet_checkpoint_match.return_value = info
        runtime = types.SimpleNamespace(release=Mock())
        ideogram = types.SimpleNamespace(_PIPELINE_CACHE={'old-lora':object()})
        lifecycle = types.SimpleNamespace(release_changed=Mock())
        with patch.dict(sys.modules, {'pi_qwen21.lib.runtime':runtime,
                                     'pi_ideogram_forge.run':ideogram,
                                     'pi_ideogram_forge.lifecycle':lifecycle}):
            with engine._checkpoint_session(fixture.p, 'base'):
                for _ in range(3):
                    with engine._checkpoint_session(fixture.p, 'base'):
                        pass
        runtime.release.assert_called_once()
        lifecycle.release_changed.assert_called_once()
        self.assertIsNot(lifecycle.release_changed.call_args.args[1], None)
        self.assertNotEqual(lifecycle.release_changed.call_args.args[1], 'base')

    def test_preflight_and_explicit_final_text_are_clean_without_mutating_base(self):
        from test_engine import Tests
        fixture = Tests()
        fixture.setUp()
        fixture.p.all_prompts = ['person <lora:base:1>']
        fixture.p.all_negative_prompts = ['<lokr:negative:1>']
        self.assertEqual(fixture.engine._target_text(fixture.p), (['person '], ['']))
        self.assertEqual(fixture.p.all_prompts, ['person <lora:base:1>'])
        fixture.p._pi_refiner_text = ('final <lyco:foreign:1>', '<lora:negative:1>')
        self.assertEqual(fixture.engine._target_text(fixture.p, 0), ('final ', ''))
        self.assertEqual(fixture.p._pi_refiner_text, ('final <lyco:foreign:1>', '<lora:negative:1>'))


if __name__ == '__main__':
    unittest.main()
