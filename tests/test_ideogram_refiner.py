import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).parents[1]


class IdeogramRefinerTests(unittest.TestCase):
    def setUp(self):
        self.saved = {name: sys.modules.get(name) for name in
                      ('modules', 'pi_ideogram_forge.selection', 'pi_ideogram_forge.routing',
                       'pi_refiner_ideogram_refiner')}
        self.info = types.SimpleNamespace(filename='ideogram4_int8_convrot.safetensors')
        self.opts = types.SimpleNamespace(data={
            'sd_model_checkpoint':'base', 'ideogram4_i2i_sampler':'Euler',
            'ideogram4_i2i_scheduler':'Simple', 'ideogram4_i2i_step':20,
            'ideogram4_i2i_cfg':7.0})
        modules = types.ModuleType('modules')
        modules.scripts = types.SimpleNamespace(Script=object, AlwaysVisible=True, scripts_data=[])
        modules.script_callbacks = types.SimpleNamespace(on_before_ui=Mock(), on_script_unloaded=Mock(), on_app_started=Mock())
        modules.processing = types.SimpleNamespace()
        modules.shared = types.SimpleNamespace(opts=self.opts, state=types.SimpleNamespace())
        modules.sd_models = types.SimpleNamespace(get_closet_checkpoint_match=Mock(return_value=self.info))
        self.modules = modules
        sys.modules['modules'] = modules
        sys.modules['pi_ideogram_forge.selection'] = types.SimpleNamespace(_ci_is_ideogram4=lambda info: info is self.info)
        self.routing = types.SimpleNamespace(_enforce_loadable_policy=Mock())
        sys.modules['pi_ideogram_forge.routing'] = self.routing
        self.resolver = Mock(return_value=['ideogram-vae', 'ideogram-te'])
        sys.modules['pi_refiner_ideogram_refiner'] = types.SimpleNamespace(
            resolve_modules=self.resolver, guidance_prompt=lambda _file, prompt, _cfg, _mode: (prompt, 'test guidance'))
        spec = importlib.util.spec_from_file_location('ideogram_refiner_engine_test', ROOT / 'scripts' / 'engine.py')
        self.engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.engine)

    def tearDown(self):
        for name, module in self.saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def request(self, **changes):
        values = dict(prompt='target', negative_prompt='', all_prompts=['target'],
                      all_negative_prompts=[''], image_mask=None, subseed_strength=0,
                      _pi_refiner=('ideogram', .75, 0), _pi_refiner_text=None,
                      _pi_refiner_steps=0, steps=30)
        values.update(changes)
        return types.SimpleNamespace(**values)

    def test_profile_marks_owner_pipeline_external_and_uses_native_i2i_defaults(self):
        profile = self.engine._target_profile('ideogram', self.info)
        self.assertEqual(profile, {'family':'ideogram4', 'external':True,
            'preset':'ideogram4', 'checkpoint':self.info.filename,
            'modules':['ideogram-vae', 'ideogram-te'], 'sampler':'Euler',
            'scheduler':'Simple', 'step':20, 'cfg':7.0})
        p = self.request(_pi_target_profile=profile)
        self.assertEqual(self.engine._fast_steps(p, .25), 5)
        self.assertEqual(self.engine._target_cfg(p, 0), 7.0)

    def test_preflight_calls_owner_policy_and_accepts_full_frame_empty_negative(self):
        p = self.request()
        self.assertIs(self.engine._preflight('ideogram', p), self.info)
        self.routing._enforce_loadable_policy.assert_called_once_with(self.info.filename)

    def test_preflight_rejects_unsupported_intent_before_gpu_switch(self):
        cases = tuple((({name:object()}, 'does not support masks') for name in
                       ('image_mask', 'mask', 'mask_for_overlay', 'latent_mask'))) + (
                 ({'_pi_refiner_text':('target', 'explicit negative')}, 'empty target negative'),
                 ({'subseed_strength':.2}, 'variation seed strength'))
        for changes, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.engine._preflight('ideogram', self.request(**changes))

    def test_inherited_base_negative_is_left_for_target_only_suppression(self):
        p = self.request(negative_prompt='base negative', all_negative_prompts=['base negative'])
        self.assertIs(self.engine._preflight('ideogram', p), self.info)

    def test_missing_owner_route_fails_closed(self):
        sys.modules.pop('pi_ideogram_forge.routing')
        with self.assertRaisesRegex(ValueError, 'routing is unavailable'):
            self.engine._preflight('ideogram', self.request())

    def test_api_override_binds_owner_loading_record_only_inside_request(self):
        base = types.SimpleNamespace(filename='base.safetensors', title='base')
        self.info.title = 'Ideogram4 - int8'
        original_loading = {'checkpoint_info':base, 'additional_modules':['base-vae'],
                            'unet_storage_dtype':'base-dtype'}
        self.modules.sd_models.model_data = types.SimpleNamespace(
            forge_loading_parameters=original_loading)
        self.modules.sd_models.get_closet_checkpoint_match.side_effect = lambda name: (
            self.info if name in ('ideogram override', self.info.title) else base)
        p = types.SimpleNamespace(override_settings={'sd_model_checkpoint':'ideogram override'})

        with patch.object(self.engine, '_release_for_switch'), \
             self.assertRaisesRegex(RuntimeError, 'request failed'):
            with self.engine._ideogram_base_override(p):
                bound = self.modules.sd_models.model_data.forge_loading_parameters
                self.assertEqual(self.opts.data['sd_model_checkpoint'], self.info.title)
                self.assertIs(bound['checkpoint_info'], self.info)
                self.assertEqual(bound['additional_modules'], ['ideogram-vae', 'ideogram-te'])
                self.assertIsNone(bound['unet_storage_dtype'])
                raise RuntimeError('request failed')

        self.assertEqual(self.opts.data['sd_model_checkpoint'], 'base')
        self.assertIs(self.modules.sd_models.model_data.forge_loading_parameters,
                      original_loading)

    def test_missing_owner_attestation_retains_base_image(self):
        original = types.SimpleNamespace(width=768, height=512, info={})
        refined = types.SimpleNamespace(info={})
        job = types.SimpleNamespace(close=Mock(), cfg_scale=7, steps=5)
        self.modules.processing.StableDiffusionProcessingImg2Img = Mock(return_value=job)
        self.modules.processing.process_images = Mock(return_value=types.SimpleNamespace(
            images=[refined], index_of_first_image=0, extra_generation_params={}))
        self.modules.shared.state = types.SimpleNamespace(
            interrupted=False, skipped=False, stopping_generation=False)
        p = self.request(_pi_active_checkpoint='ideogram',
            _pi_target_profile=self.engine._target_profile('ideogram', self.info),
            seed=1, subseed=0, all_seeds=[1], all_subseeds=[0], cfg_scale=7,
            distilled_cfg_scale=3.5, sampler_name='Euler', scheduler='Simple',
            extra_generation_params={})
        pp = types.SimpleNamespace(image=original, index=0)

        self.engine.Script().postprocess_image_after_composite(p, pp)

        self.assertIs(pp.image, original)
        self.assertIn('did not confirm refinement', p.extra_generation_params['Refiner error'])
        job.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
