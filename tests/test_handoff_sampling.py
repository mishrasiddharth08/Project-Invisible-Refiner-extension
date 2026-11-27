"""CPU regressions for the actual engine handoff wrapper, with Forge mocked."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class HandoffSamplingTests(unittest.TestCase):
    def setUp(self):
        self.base = types.SimpleNamespace(filename='base.safetensors')
        self.target = types.SimpleNamespace(filename='target.safetensors')
        self.loading = {'checkpoint_info': self.base, 'additional_modules': ['clip', 'vae'], 'unet_storage_dtype': 'Automatic'}
        self.opts = types.SimpleNamespace(data={'sd_checkpoint_hash': 'base-hash'})
        self.host = types.SimpleNamespace(sampling_cleanup=Mock(), sampling_prepare=Mock())
        self.extra = types.SimpleNamespace(activate=Mock())
        self.sd_models = types.SimpleNamespace(model_data=types.SimpleNamespace(forge_loading_parameters=self.loading),
            get_closet_checkpoint_match=Mock(return_value=self.target))
        self.modules = types.ModuleType('modules')
        self.modules.scripts = types.SimpleNamespace(Script=object, AlwaysVisible=True)
        self.modules.script_callbacks = types.SimpleNamespace(on_script_unloaded=Mock(), on_before_ui=Mock(), on_app_started=Mock())
        self.modules.processing = types.SimpleNamespace()
        self.modules.sd_models = self.sd_models
        self.modules.shared = types.SimpleNamespace(opts=self.opts)
        self.modules.sd_samplers_common, self.modules.extra_networks = self.host, self.extra
        self.module_patch = patch.dict(sys.modules, {'modules': self.modules})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        spec = importlib.util.spec_from_file_location('pi_handoff_sampling_test', Path(__file__).parents[1] / 'scripts' / 'engine.py')
        self.engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.engine)
        self.compatibility = Mock(return_value=True)
        self.engine._refiner_helper = Mock(return_value=types.SimpleNamespace(handoff_compatible=self.compatibility))
        self.network_data = {'lora': ['base-adapter']}
        self.base_model = self.make_model(self.base)
        self.target_model = self.make_model(self.target)
        self.p = types.SimpleNamespace(_pi_refiner_mode=self.engine._LAST_STEPS, _pi_dispatching=True,
            _pi_refining=False, _pi_refiner=('target', .75, 3), _pi_handoff_iteration=None,
            iteration=0, extra_generation_params={}, sampler_name='Euler', sd_model=self.base_model,
            cfg_scale=7, extra_network_data=self.network_data, disable_extra_networks=False,
            clear_prompt_cache=Mock(), setup_conds=Mock())
        def reload():
            info = self.sd_models.model_data.forge_loading_parameters['checkpoint_info']
            self.p.sd_model = self.base_model if info is self.base else self.target_model
            self.opts.data['sd_checkpoint_hash'] = 'loaded-' + info.filename
        self.sd_models.forge_model_reload = Mock(side_effect=reload)
        self.denoiser = types.SimpleNamespace(p=self.p, total_steps=20, step=15,
            model_wrap=object(), update_inner_model=Mock())
        self.original = Mock(return_value='original-result')
        self.x, self.sigma = object(), object()

    @staticmethod
    def make_model(info):
        objects = types.SimpleNamespace(unet=types.SimpleNamespace(controlnet_linked_list=None))
        applied = types.SimpleNamespace(shallow_copy=Mock(return_value=objects))
        return types.SimpleNamespace(sd_checkpoint_info=info, forge_objects=objects,
                                     forge_objects_after_applying_lora=applied)

    def apply(self):
        return self.engine._apply_handoff(self.original, self.denoiser, self.x, self.sigma)

    def test_pixel_handoff_uses_solver_steps_and_target_budget(self):
        class EndBase(BaseException):
            pass
        self.host.InterruptedException = EndBase
        self.p.batch_size, self.p.iteration = 2, 1
        self.compatibility.return_value = False
        self.engine._preflight = Mock(return_value=self.target)
        self.engine._target_profile = Mock(return_value={'family': 'flux2_klein9', 'modules': ['vae', 'encoder']})
        self.modules.shared.state = types.SimpleNamespace(current_latent=self.x)
        self.denoiser.steps = 20
        sampler = types.SimpleNamespace(p=self.p, model_wrap_cfg=self.denoiser)
        callback = self.engine._sampler_callback(Mock())
        with patch.dict(sys.modules, {'modules.sd_samplers_common': self.host}):
            for name in ('Euler a', 'Heun', 'DPM++ 2M', 'DDIM'):
                self.p.sampler_name = name
                self.p._pi_image_steps, self.p._pi_bridge_examined = {}, None
                self.p._pi_handoff_iteration = None
                with self.subTest(name=name), self.assertRaises(EndBase):
                    if name == 'Euler a':
                        self.denoiser.step = 15
                        self.apply()
                    else:
                        callback(sampler, {'i': 13, 'denoised': self.x})
                        self.assertFalse(self.p._pi_image_steps)
                        callback(sampler, {'i': 14, 'denoised': self.x})
                self.assertEqual(self.p._pi_image_steps, {2: 5, 3: 5})
                self.assertIs(self.modules.shared.state.current_latent, self.x)
                self.assertEqual(self.p.extra_generation_params['Refiner handoff step'], 16)

    def test_threshold_then_success_preserves_recipe_and_target_conditioning(self):
        self.denoiser.step = 14
        self.assertFalse(self.apply())
        self.sd_models.forge_model_reload.assert_not_called()
        self.assertIsNone(self.p._pi_handoff_iteration)
        observed = []
        self.p.setup_conds.side_effect = lambda: observed.append((self.p.sd_model, self.p.cfg_scale, self.p.extra_network_data))
        self.denoiser.step = 15
        self.assertTrue(self.apply())
        self.assertEqual(observed, [(self.target_model, 3, {})])
        self.assertIs(self.p.sd_model, self.target_model)
        self.assertIs(self.sd_models.model_data.forge_loading_parameters, self.loading)
        self.assertIs(self.p.extra_network_data, self.network_data)
        self.assertEqual((self.p.cfg_scale, self.p.refiner_cfg), (7, 3))
        self.assertEqual(self.opts.data['sd_checkpoint_hash'], 'base-hash')
        self.assertEqual(self.p.extra_generation_params['Refiner mode'], self.engine._LAST_STEPS)
        self.assertEqual(self.p.extra_generation_params['Refiner handoff step'], 16)
        self.host.sampling_cleanup.assert_called_once_with(self.base_model.forge_objects.unet)
        self.host.sampling_prepare.assert_called_once_with(self.target_model.forge_objects.unet, x=self.x)
        self.original.assert_not_called()

    def test_donor_guided_edit_finishes_on_the_base_for_both_handoffs(self):
        self.p.extra_generation_params.update({'Qwen Head Swap': 'identity', 'Headshots used': 2})
        self.p._pi_bridge_examined = None
        self.p._pi_image_steps = {}
        self.engine._image_step_boundary(self.p, self.x, 15, 20)
        self.assertFalse(self.p._pi_image_steps)
        self.assertFalse(self.apply())
        self.assertIn('donor-guided', self.p.extra_generation_params['Refiner fallback'])
        self.assertEqual(self.p.extra_generation_params['Refiner mode'], self.engine._FINAL_ONLY)
        self.sd_models.forge_model_reload.assert_not_called()

    def test_same_checkpoint_falls_back_without_reloading(self):
        self.sd_models.get_closet_checkpoint_match.return_value = self.base
        self.assertFalse(self.apply())
        self.sd_models.forge_model_reload.assert_not_called()
        self.assertIn('Refiner fallback', self.p.extra_generation_params)

    def test_incompatible_pair_falls_back_without_touching_sampler(self):
        self.compatibility.return_value = False
        self.assertFalse(self.apply())
        self.assertEqual(self.p.extra_generation_params['Refiner mode'], self.engine._FINAL_ONLY)
        self.assertIn('verified latent handoff', self.p.extra_generation_params['Refiner fallback'])
        self.host.sampling_cleanup.assert_not_called()
        self.sd_models.forge_model_reload.assert_not_called()

    def test_alternate_sampler_uses_final_image_fallback(self):
        self.p.sampler_name = 'DPM++ 2M'
        self.assertFalse(self.apply())
        self.compatibility.assert_not_called()
        self.sd_models.forge_model_reload.assert_not_called()
        self.assertIn('native Euler', self.p.extra_generation_params['Refiner fallback'])

    def test_repeated_evaluations_switch_only_once_per_iteration(self):
        self.assertTrue(self.apply())
        self.denoiser.step = 16
        self.assertFalse(self.apply())
        self.sd_models.forge_model_reload.assert_called_once()
        # Forge loads the base recipe before starting the next batch.
        self.p.sd_model = self.base_model
        self.p.iteration = 1
        self.denoiser.step = 14
        self.assertFalse(self.apply())
        self.denoiser.step = 15
        self.assertTrue(self.apply())
        self.assertEqual(self.sd_models.forge_model_reload.call_count, 2)
        self.assertEqual(self.p._pi_handoff_iteration, 1)

    def test_failed_target_load_restores_base_adapters_before_next_evaluation(self):
        reload_normal = self.sd_models.forge_model_reload.side_effect
        def fail_target_load():
            reload_normal()
            if self.p.sd_model is self.target_model:
                raise RuntimeError('target load failed')
        self.sd_models.forge_model_reload.side_effect = fail_target_load
        observed = []
        self.p.setup_conds.side_effect = lambda: observed.append((self.p.sd_model, self.p.cfg_scale, self.p.extra_network_data))
        self.assertFalse(self.apply())
        self.assertEqual(self.sd_models.forge_model_reload.call_count, 2)
        self.assertIs(self.p.sd_model, self.base_model)
        self.extra.activate.assert_called_once_with(self.p, self.network_data)
        self.base_model.forge_objects_after_applying_lora.shallow_copy.assert_called_once()
        self.assertEqual(observed, [(self.base_model, 7, self.network_data)])
        self.assertIs(self.sd_models.model_data.forge_loading_parameters, self.loading)
        self.assertIs(self.p.extra_network_data, self.network_data)
        self.assertEqual(self.opts.data['sd_checkpoint_hash'], 'base-hash')
        self.host.sampling_prepare.assert_called_once_with(self.base_model.forge_objects.unet, x=self.x)
        self.denoiser.update_inner_model.assert_called_once()
        self.assertIn('target load failed', self.p.extra_generation_params['Refiner fallback'])
        self.assertFalse(self.apply())
        self.assertEqual(self.sd_models.forge_model_reload.call_count, 2)

    def test_other_modes_delegate_to_host(self):
        self.p._pi_refiner_mode = self.engine._FINAL_ONLY
        self.assertEqual(self.apply(), 'original-result')
        self.original.assert_called_once_with(self.denoiser, self.x, self.sigma)


if __name__ == '__main__':
    unittest.main()
