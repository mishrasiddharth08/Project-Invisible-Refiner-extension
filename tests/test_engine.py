import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import logging
logging.disable(logging.CRITICAL)

class Tests(unittest.TestCase):
    def setUp(self):
        self.opts = types.SimpleNamespace(data={'sd_model_checkpoint': 'base'})
        self.state = types.SimpleNamespace(interrupted=False, skipped=False, job_no=4)
        modules = types.ModuleType('modules')
        modules.scripts = types.SimpleNamespace(Script=object, AlwaysVisible=True)
        modules.script_callbacks = types.SimpleNamespace(on_script_unloaded=Mock(), on_before_ui=Mock(), on_app_started=Mock())
        modules.shared = types.SimpleNamespace(opts=self.opts, state=self.state)
        modules.sd_models = types.SimpleNamespace(get_closet_checkpoint_match=Mock(return_value=object()), forge_model_reload=Mock())
        self.job = types.SimpleNamespace(close=Mock(), cfg_scale=7, steps=5)
        self.output = object()
        modules.processing = types.SimpleNamespace(StableDiffusionProcessingImg2Img=Mock(return_value=self.job), process_images=Mock(return_value=types.SimpleNamespace(images=[self.output], index_of_first_image=0)))
        native = types.ModuleType('modules.processing_scripts.refiner')
        native.ScriptRefiner = type('ScriptRefiner', (), {'setup': Mock(), 'ui': Mock()})
        self.entry = types.SimpleNamespace(checkpoint_change=Mock(side_effect=lambda *a, **k: self.opts.data.update(sd_model_checkpoint='target')), refresh_model_loading_parameters=Mock())
        sys.modules.update({'modules': modules, 'modules.processing_scripts': types.ModuleType('modules.processing_scripts'), 'modules.processing_scripts.refiner': native, 'modules_forge': types.SimpleNamespace(main_entry=self.entry)})
        spec = importlib.util.spec_from_file_location('engine_test', Path(__file__).parents[1] / 'scripts' / 'engine.py')
        self.engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.engine)
        self.native = native.ScriptRefiner
        self.original_setup = self.native.setup
        modules.scripts.scripts_data = [types.SimpleNamespace(script_class=self.native)]
        self.engine._install()
        self.engine._prompt = lambda p: p
        self.modules = modules
        self.p = types.SimpleNamespace(_pi_refiner=('target', .75, 0), prompt='base', negative_prompt='', seed=0, subseed=0, subseed_strength=0, steps=20, cfg_scale=7, distilled_cfg_scale=3.5, sampler_name='Euler', scheduler='Simple', all_prompts=['first','second'], all_seeds=[0,42], extra_generation_params={})
        self.original = types.SimpleNamespace(width=768, height=512, info={})
        self.pp = types.SimpleNamespace(image=self.original, index=1)

    def run_pass(self):
        self.engine.Script().postprocess_image_after_composite(self.p, self.pp)

    def test_metadata_free_sample_and_batch_identity(self):
        self.run_pass()
        self.assertIs(self.pp.image, self.output)
        args = self.modules.processing.StableDiffusionProcessingImg2Img.call_args.kwargs
        self.assertEqual((args['seed'],args['prompt'],args['width'],args['scheduler']), (42,'second',768,'Simple'))
        self.assertEqual(self.opts.data['sd_model_checkpoint'], 'base')
        self.job.close.assert_called_once()
        self.assertFalse(self.job.disable_extra_networks)

    def test_failure_restores_and_retains_image(self):
        self.modules.processing.process_images.side_effect = RuntimeError('unsupported image input')
        self.run_pass()
        self.assertIs(self.pp.image, self.original)
        self.assertIn('Refiner error', self.p.extra_generation_params)
        self.assertEqual(self.opts.data['sd_model_checkpoint'], 'base')
        self.assertFalse(self.p._pi_refining)

    def test_cancel_does_not_publish_partial(self):
        def cancel(job):
            self.state.interrupted = True
            return types.SimpleNamespace(images=[self.output])
        self.modules.processing.process_images.side_effect = cancel
        self.run_pass()
        self.assertIs(self.pp.image, self.original)
        self.assertTrue(self.state.interrupted)

    def test_disabled_and_zero_strength(self):
        for spec in [None, ('target',1,0)]:
            self.p._pi_refiner = spec
            self.run_pass()
        self.modules.processing.process_images.assert_not_called()

    def test_native_setup_disables_latent_switch(self):
        self.engine._setup(self.native(),self.p,True,'target',.8,7,'')
        self.assertIsNone(self.p.refiner_checkpoint)
        self.assertEqual(self.p._pi_refiner, ('target',.8,7))

    def test_unload_restores_native(self):
        self.engine._unload()
        self.assertIs(self.native.setup,self.original_setup)

    def test_takeover_without_native_callbacks_refines_and_rewrites_saved_sample(self):
        self.output = types.SimpleNamespace(info={})
        self.modules.processing.process_images.return_value.images = [self.output]
        self.p._pi_saved_files = {id(self.original): [('sample.png', None)]}
        result = types.SimpleNamespace(images=[self.original], index_of_first_image=0,
            all_seeds=[123], all_prompts=['actual takeover prompt'],
            infotexts=['base parameters'], info='base parameters', extra_generation_params={})
        self.engine._rewrite_saved = Mock()
        returned = self.engine._finish(self.p, result)
        self.assertIs(returned.images[0], self.output)
        self.assertEqual(self.modules.processing.StableDiffusionProcessingImg2Img.call_args.kwargs['seed'], 123)
        self.engine._rewrite_saved.assert_called_once()
        self.assertIn('Refiner: target', result.infotexts[0])
        self.modules.sd_models.forge_model_reload.assert_not_called()
        self.assertEqual(self.p.all_seeds, [0,42])

    def test_stock_callback_is_not_refined_twice(self):
        self.p._pi_standard_callback = True
        result = types.SimpleNamespace(images=[self.original])
        self.engine._finish(self.p, result)
        self.modules.processing.process_images.assert_not_called()

    def test_capture_save_uses_actual_returned_path(self):
        self.p._pi_saved_files = {}
        def save(image, path, basename, p=None):
            return ('actual.png', None)
        self.engine._record_save(save, self.original, 'output', '', p=self.p)
        self.assertEqual(self.p._pi_saved_files[id(self.original)], [('actual.png', None)])

    def test_dispatch_arms_bypassed_native_setup_and_cleans_up(self):
        native = self.native()
        native.args_from, native.args_to = 1, 6
        runner = types.SimpleNamespace(alwayson_scripts=[native])
        original = Mock(return_value=None)
        self.p._pi_refiner = None
        args = (runner, self.p, 0, True, 'target', .5, 7, 'ignored rule')
        self.engine._dispatch(original, self.p, args, {}, runner)
        self.assertEqual(self.p._pi_refiner, ('target', .5, 7))
        self.assertFalse(self.p._pi_dispatching)
        self.assertIsNone(self.p._pi_saved_files)

    def test_dispatch_exception_cleans_up(self):
        original = Mock(side_effect=RuntimeError('base failure'))
        with self.assertRaisesRegex(RuntimeError, 'base failure'):
            self.engine._dispatch(original,self.p,(self.p,),{})
        self.assertFalse(self.p._pi_dispatching)
        self.assertIsNone(self.p._pi_saved_files)

    def test_options_runner_preserves_settings_without_postprocessing(self):
        marker = object()
        runner = self.engine._OptionsOnlyRunner(types.SimpleNamespace(alwayson_scripts=[marker]))
        self.assertEqual(runner.alwayson_scripts, [marker])
        self.assertIsNone(runner.postprocess(self.p, None))

    def test_lora_rules_are_not_applied(self):
        # The removed textbox must not leave stale replacement rules active.
        self.assertEqual(self.engine._prompt('cat <lora:high_noise:1>'), 'cat <lora:high_noise:1>')

    def test_failed_rewrite_retains_original_gallery(self):
        self.modules.processing.process_images.return_value.images = [types.SimpleNamespace(info={})]
        result = types.SimpleNamespace(images=[self.original], index_of_first_image=0, infotexts=['base'], comments='')
        self.engine._rewrite_saved = Mock(side_effect=OSError('disk full'))
        self.assertIs(self.engine._finish(self.p, result).images[0], self.original)
        self.assertIn('disk full', result.comments)

    def test_api_fields_arm_refiner(self):
        self.p._pi_refiner = None
        self.p.refiner_checkpoint = 'target'
        self.p.refiner_switch_at = 0
        self.p.refiner_cfg = 3
        self.engine._arm(self.p)
        self.assertEqual(self.p._pi_refiner, ('target', 0, 3))

    def test_atomic_saved_file_replacement(self):
        import tempfile
        from PIL import Image, PngImagePlugin
        def save(image, info, filename, **kwargs):
            meta = PngImagePlugin.PngInfo()
            meta.add_text('parameters', info)
            image.save(filename, pnginfo=meta)
        self.modules.images = types.SimpleNamespace(save_image_with_geninfo=save)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'sample.png'
            Image.new('RGB',(8,8),'red').save(target)
            image = Image.new('RGB',(8,8),'blue')
            self.engine._rewrite_saved(image, 'refined metadata', [(str(target), None)])
            with Image.open(target) as loaded:
                self.assertEqual(loaded.getpixel((0,0)),(0,0,255))
                self.assertEqual(loaded.info['parameters'], 'refined metadata')
            self.assertEqual([f.name for f in Path(folder).iterdir()], ['sample.png'])

    def distinct_checkpoints(self):
        self.infos = {}
        self.modules.sd_models.get_closet_checkpoint_match.side_effect = lambda key: self.infos.setdefault(key, types.SimpleNamespace(filename=str(key)))

    def batch_result(self, count=3):
        return types.SimpleNamespace(images=[types.SimpleNamespace(width=512,height=512,info={}) for _ in range(count)],
            index_of_first_image=0, all_seeds=list(range(count)), all_prompts=['sample']*count,
            infotexts=['base']*count, info='base', extra_generation_params={})

    def test_batch_switches_once_and_restores_family_options(self):
        self.distinct_checkpoints()
        self.opts.data.update(forge_preset='base-family', forge_unet_storage_dtype='forced-fp8')
        original = self.opts.data.copy()
        def change(*args, **kwargs):
            self.opts.data.update(sd_model_checkpoint='target',forge_preset='target-family',
                                  forge_additional_modules_target=['target-vae'])
        self.entry.checkpoint_change.side_effect = change
        self.modules.processing.process_images.return_value.images = [types.SimpleNamespace(info={})]
        self.engine._rewrite_saved = Mock()
        self.engine._finish(self.p,self.batch_result())
        self.entry.checkpoint_change.assert_called_once()
        self.assertEqual(self.modules.processing.process_images.call_count,3)
        self.assertEqual(self.opts.data,original)
        self.modules.sd_models.forge_model_reload.assert_not_called()
        self.entry.refresh_model_loading_parameters.assert_not_called()

    def test_same_checkpoint_does_not_switch_or_reload(self):
        self.p._pi_refiner = ('base', .75, 0)
        self.modules.processing.process_images.return_value.images = [types.SimpleNamespace(info={})]
        self.engine._rewrite_saved = Mock()
        self.engine._finish(self.p,self.batch_result())
        self.entry.checkpoint_change.assert_not_called()
        self.modules.sd_models.forge_model_reload.assert_not_called()

    def test_format_pairings_are_forwarded_without_conversion(self):
        self.distinct_checkpoints()
        formats = ('bf16','fp8_scaled','int8_convrot','nvfp4','convrot_w4a4','gguf','mxfp8','future-format')
        for source in formats:
            for target in formats:
                with self.subTest(source=source,target=target):
                    self.opts.data = {'sd_model_checkpoint':source,'forge_unet_storage_dtype':'forced-fp8'}
                    self.entry.checkpoint_change.side_effect = lambda name, **kw: self.opts.data.update(sd_model_checkpoint=name)
                    with self.engine._checkpoint_session(self.p,target):
                        self.assertEqual(self.opts.data['sd_model_checkpoint'],target)
                        self.assertEqual(self.opts.data['forge_unet_storage_dtype'], 'forced-fp8' if source==target else 'Automatic')
                    self.assertEqual(self.opts.data,{'sd_model_checkpoint':source,'forge_unet_storage_dtype':'forced-fp8'})

    def test_partial_selection_failure_restores_all_changed_keys(self):
        self.distinct_checkpoints()
        def change(*args, **kwargs):
            self.opts.data.update(sd_model_checkpoint='target',forge_checkpoint_target='temporary')
            raise RuntimeError('missing target encoder')
        self.entry.checkpoint_change.side_effect = change
        result = self.batch_result()
        images = list(result.images)
        self.engine._finish(self.p,result)
        self.assertEqual(self.opts.data,{'sd_model_checkpoint':'base'})
        self.assertEqual(result.images,images)
        self.assertIn('missing target encoder',result.comments)

    def test_loading_configuration_restored_without_refresh(self):
        self.distinct_checkpoints()
        loading = {'checkpoint_info':object(),'additional_modules':['base-vae'],'unet_storage_dtype':object()}
        self.modules.sd_models.model_data=types.SimpleNamespace(forge_loading_parameters=loading)
        self.entry.dynamic_args=types.SimpleNamespace(online_lora=False)
        with self.engine._checkpoint_session(self.p,'target'):
            self.modules.sd_models.model_data.forge_loading_parameters={'checkpoint_info':'target'}
            self.entry.dynamic_args.online_lora=True
        self.assertEqual(self.modules.sd_models.model_data.forge_loading_parameters,loading)
        self.assertFalse(self.entry.dynamic_args.online_lora)

    def test_target_preset_sampling_preferences(self):
        self.p._pi_changed_preset=True
        self.opts.data.update(forge_preset='target',target_i2i_sampler='DPM++ 2M',target_i2i_step=8)
        self.assertEqual(self.engine._target_setting(self.p,'sampler','Euler'),'DPM++ 2M')
        self.assertEqual(self.engine._target_setting(self.p,'step',20),8)
        self.assertEqual(self.engine._target_setting(self.p,'cfg',7),7)

    def test_native_callback_defers_until_base_batch_finishes(self):
        self.p._pi_dispatching=True
        self.run_pass()
        self.modules.processing.process_images.assert_not_called()
        self.assertFalse(getattr(self.p,'_pi_standard_callback',False))

    def test_known_text_only_endpoint_fails_before_switch(self):
        self.modules.sd_models.get_closet_checkpoint_match.return_value=types.SimpleNamespace(refiner_supports_img2img=False)
        result=self.batch_result()
        self.engine._finish(self.p,result)
        self.assertIn('cannot accept images',result.comments)
        self.entry.checkpoint_change.assert_not_called()

    def test_wrapped_save_function_keeps_processing_argument(self):
        self.p._pi_saved_files={}
        def save(*args,**kwargs): return ('image.png',None)
        self.engine._record_save(save,self.original,p=self.p)
        self.assertEqual(self.p._pi_saved_files[id(self.original)],[('image.png',None)])

    def test_zero_strength_does_not_switch(self):
        self.p._pi_refiner=('target',1,0)
        self.engine._finish(self.p,self.batch_result())
        self.entry.checkpoint_change.assert_not_called()

    def test_external_engine_cleanup_on_switch_only(self):
        self.distinct_checkpoints()
        names=('pi_qwen21.lib.runtime','pi_boogu_image.lib.pipeline','pi_sensenova.run','pi_h3.runtime')
        runtimes={name:types.SimpleNamespace(release=Mock()) for name in names}
        with patch.dict(sys.modules,runtimes):
            self.engine._release_for_switch('base')
            for runtime in runtimes.values(): runtime.release.assert_not_called()
            self.engine._release_for_switch('target')
            for runtime in runtimes.values(): runtime.release.assert_called_once()

    def test_cleanup_failure_still_restores_options(self):
        self.distinct_checkpoints()
        self.engine._release_for_switch=Mock(side_effect=[None,RuntimeError('worker cleanup failed')])
        with self.assertRaisesRegex(RuntimeError,'worker cleanup failed'):
            with self.engine._checkpoint_session(self.p,'target'):
                pass
        self.assertEqual(self.opts.data,{'sd_model_checkpoint':'base'})
        self.assertIsNone(self.p._pi_active_checkpoint)

    def test_fast_step_budget_respects_one_step_and_caps_cost(self):
        for base, strength, expected in ((1,.25,1),(20,.25,5),(100,.9,8),(20,.125,3)):
            self.p.steps=base
            self.assertEqual(self.engine._fast_steps(self.p,strength),expected)

    def test_fast_cfg_and_native_steps_without_double_denoise_scaling(self):
        self.run_pass()
        kwargs=self.modules.processing.StableDiffusionProcessingImg2Img.call_args.kwargs
        self.assertEqual(kwargs['cfg_scale'],1)
        self.assertEqual(kwargs['steps'],5)
        self.assertEqual(kwargs['override_settings'],{'img2img_fix_steps':True,'live_previews_enable':False})
        self.p._pi_refiner=('target',.75,4)
        self.pp.image=self.original
        self.run_pass()
        self.assertEqual(self.modules.processing.StableDiffusionProcessingImg2Img.call_args.kwargs['cfg_scale'],4)

    def test_cross_model_conditioning_release_keeps_source_and_mask(self):
        self.distinct_checkpoints()
        self.p._pi_external_finish=True
        self.p.c=self.p.uc=self.p.sampler=self.p.init_latent=object()
        self.p.image_mask=object()
        mask=self.p.image_mask
        self.p.init_images=[self.original]
        self.p.clear_prompt_cache=Mock()
        with self.engine._checkpoint_session(self.p,'target'):
            self.assertIsNone(self.p.c)
            self.assertIsNone(self.p.init_latent)
            self.assertIs(self.p.image_mask,mask)
            self.assertEqual(self.p.init_images,[self.original])
        self.p.clear_prompt_cache.assert_called_once()

    def test_same_model_keeps_conditioning_cache(self):
        self.distinct_checkpoints()
        self.p._pi_external_finish=True
        self.p.clear_prompt_cache=Mock()
        with self.engine._checkpoint_session(self.p,'base'):
            pass
        self.p.clear_prompt_cache.assert_not_called()

if __name__ == '__main__':
    unittest.main()
