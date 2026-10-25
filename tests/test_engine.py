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
        self.entry.refresh_model_loading_parameters.assert_called_once_with(refresh=True)

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
        self.assertEqual(self.p.clear_prompt_cache.call_count,2)

    def test_same_model_keeps_conditioning_cache(self):
        self.distinct_checkpoints()
        self.p._pi_external_finish=True
        self.p.clear_prompt_cache=Mock()
        with self.engine._checkpoint_session(self.p,'base'):
            pass
        self.p.clear_prompt_cache.assert_not_called()

    def test_native_masks_are_not_refined_or_misindexed(self):
        result=self.batch_result(2)
        first,second=result.images
        mask=object()
        result.images=[first,mask,second]
        self.p._pi_deferred_slots=[types.SimpleNamespace(image=first,index=0),types.SimpleNamespace(image=second,index=1)]
        self.modules.processing.process_images.return_value.images=[types.SimpleNamespace(info={})]
        self.engine._rewrite_saved=Mock()
        self.engine._finish(self.p,result)
        self.assertEqual(self.modules.processing.process_images.call_count,2)
        self.assertIs(result.images[1],mask)
        self.assertEqual([call.kwargs['seed'] for call in self.modules.processing.StableDiffusionProcessingImg2Img.call_args_list],[0,1])
        self.assertNotIn('Refiner:',result.infotexts[1])
        self.assertIn('Refiner:',result.infotexts[2])
        self.assertIsNone(self.p._pi_deferred_slots)

    def test_saved_grid_updated_when_not_returned_in_gallery(self):
        result=self.batch_result(1)
        self.p._pi_saved_grids=[123]
        self.p._pi_saved_files={123:[('grid.png',None)]}
        self.modules.processing.process_images.return_value.images=[types.SimpleNamespace(info={})]
        grid=types.SimpleNamespace(info={})
        self.modules.images=types.SimpleNamespace(image_grid=Mock(return_value=grid))
        self.engine._rewrite_saved=Mock()
        self.engine._finish(self.p,result)
        self.assertEqual(len(result.images),1)
        self.engine._rewrite_saved.assert_any_call(grid,result.info,[('grid.png',None)])

    def test_no_grid_rebuild_when_refinement_fails(self):
        result=self.batch_result(1)
        result.images.insert(0,object())
        result.index_of_first_image=1
        self.modules.processing.process_images.side_effect=RuntimeError('unsupported model')
        self.modules.images=types.SimpleNamespace(image_grid=Mock())
        self.engine._finish(self.p,result)
        self.modules.images.image_grid.assert_not_called()

    def test_component_list_mutation_does_not_corrupt_loader_snapshot(self):
        self.distinct_checkpoints()
        self.modules.sd_models.model_data=types.SimpleNamespace(forge_loading_parameters={'additional_modules':['base-vae']})
        with self.engine._checkpoint_session(self.p,'target'):
            self.modules.sd_models.model_data.forge_loading_parameters['additional_modules'].append('target-encoder')
        self.assertEqual(self.modules.sd_models.model_data.forge_loading_parameters['additional_modules'],['base-vae'])

    def test_target_preset_dtype_reaches_loader_refresh(self):
        self.distinct_checkpoints()
        self.opts.data.update(forge_preset='base',forge_unet_storage_dtype_target='target-fp8')
        def change(*args,**kwargs):
            self.assertFalse(kwargs['refresh'])
            self.opts.data.update(sd_model_checkpoint='target',forge_preset='target')
        self.entry.checkpoint_change.side_effect=change
        dtypes=[]
        self.entry.refresh_model_loading_parameters.side_effect=lambda **kw:dtypes.append(self.opts.data['forge_unet_storage_dtype'])
        with self.engine._checkpoint_session(self.p,'target'): pass
        self.assertEqual(dtypes,['target-fp8'])
        self.assertEqual(self.opts.data,{'sd_model_checkpoint':'base','forge_preset':'base','forge_unet_storage_dtype_target':'target-fp8'})

    def test_unresolved_target_preset_does_not_inherit_source_dtype(self):
        self.distinct_checkpoints()
        self.opts.data.update(forge_preset='source',forge_unet_storage_dtype_source='forced-fp8')
        with self.engine._checkpoint_session(self.p,'target'):
            self.assertEqual(self.opts.data['forge_unet_storage_dtype'],'Automatic')

    def test_backend_options_mutated_during_pass_are_restored(self):
        self.distinct_checkpoints()
        self.opts.data['existing']='original'
        with self.engine._checkpoint_session(self.p,'target'):
            self.opts.data['runtime-only']='new'
            self.opts.data.pop('existing')
        self.assertEqual(self.opts.data,{'sd_model_checkpoint':'base','existing':'original'})

    def test_forward_nondefault_mask_and_image_guidance_settings(self):
        mask=object()
        self.p.image_mask=mask
        settings=dict(mask_round=False,inpaint_full_res=False,inpaint_full_res_padding=48,
                      initial_noise_multiplier=.7,latent_mask=object(),image_cfg_scale=2.5)
        for key,value in settings.items(): setattr(self.p,key,value)
        self.run_pass()
        self.assertIs(self.job.image_mask,mask)
        for key,value in settings.items():
            if key=='image_cfg_scale':
                self.assertEqual(self.modules.processing.StableDiffusionProcessingImg2Img.call_args.kwargs[key],value)
            else: self.assertEqual(getattr(self.job,key),value)

    def test_saved_image_and_sidecar_rollback_together(self):
        import tempfile,os
        from PIL import Image
        self.modules.images=types.SimpleNamespace(save_image_with_geninfo=lambda image,info,filename,**kw:image.save(filename))
        with tempfile.TemporaryDirectory() as folder:
            image_path=Path(folder)/'image.png';text_path=Path(folder)/'image.txt'
            Image.new('RGB',(4,4),'red').save(image_path);text_path.write_text('original')
            before={p.name:p.read_bytes() for p in Path(folder).iterdir()}
            replace=os.replace
            count=0
            def fail_second(source,target):
                nonlocal count
                count+=1
                if count==2: raise OSError('second commit failed')
                return replace(source,target)
            with patch('os.replace',side_effect=fail_second):
                with self.assertRaisesRegex(OSError,'second commit failed'):
                    self.engine._rewrite_saved(Image.new('RGB',(4,4),'blue'),'new',[(str(image_path),str(text_path))])
            self.assertEqual({p.name:p.read_bytes() for p in Path(folder).iterdir()},before)

    def test_sidecar_staging_failure_does_not_change_image(self):
        import tempfile
        from PIL import Image
        self.modules.images=types.SimpleNamespace(save_image_with_geninfo=lambda image,info,filename,**kw:image.save(filename))
        with tempfile.TemporaryDirectory() as folder:
            image_path=Path(folder)/'image.png';text_path=Path(folder)/'image.txt'
            Image.new('RGB',(4,4),'red').save(image_path);text_path.write_text('original')
            before={p.name:p.read_bytes() for p in Path(folder).iterdir()}
            with patch.object(Path,'write_text',side_effect=OSError('text write failed')):
                with self.assertRaisesRegex(OSError,'text write failed'):
                    self.engine._rewrite_saved(Image.new('RGB',(4,4),'blue'),'new',[(str(image_path),str(text_path))])
            self.assertEqual({p.name:p.read_bytes() for p in Path(folder).iterdir()},before)

    def test_returned_grid_preserves_grid_text_and_excludes_masks(self):
        result=self.batch_result(2)
        first,second=result.images;mask=object();old_grid=object()
        result.images=[old_grid,first,mask,second];result.index_of_first_image=1
        result.info='GRID XYZ axes';result.infotexts=['GRID XYZ axes','sample first','sample second']
        self.p._pi_deferred_slots=[types.SimpleNamespace(image=first,index=0),types.SimpleNamespace(image=second,index=1)]
        self.modules.processing.process_images.side_effect=[types.SimpleNamespace(images=[types.SimpleNamespace(info={})],index_of_first_image=0) for _ in range(2)]
        self.engine._rewrite_saved=Mock()
        self.modules.images=types.SimpleNamespace(image_grid=Mock(return_value=types.SimpleNamespace(info={})))
        self.engine._finish(self.p,result)
        self.assertTrue(result.info.startswith('GRID XYZ axes'))
        self.assertTrue(result.infotexts[0].startswith('GRID XYZ axes'))
        self.assertTrue(result.infotexts[3].startswith('sample second'))
        self.assertNotIn('Refiner:',result.infotexts[2])
        grid_sources=self.modules.images.image_grid.call_args.args[0]
        self.assertEqual(len(grid_sources),2)
        self.assertNotIn(mask,grid_sources)

    def test_stranded_wrapper_is_inactive_after_unload(self):
        self.modules.scripts.ScriptRunner=type('Runner',(),{'run':lambda self,p,*args:None})
        self.modules.images=types.SimpleNamespace(save_image=lambda image,*args,**kwargs:('sample.png',None))
        original=self.modules.processing.process_images
        self.engine._install_routes()
        stranded=self.modules.processing.process_images
        self.modules.processing.process_images=lambda p:stranded(p)
        self.engine._unload()
        self.engine._finish=Mock(side_effect=AssertionError('unloaded refiner ran'))
        self.modules.processing.process_images(self.p)
        original.assert_called_once_with(self.p)

    def test_cross_batch_unloads_native_weights_and_clears_latents(self):
        self.distinct_checkpoints()
        self.modules.sd_models.unload_model_weights=Mock()
        self.p._pi_external_finish=True
        self.p.latents_after_sampling=[object()];self.p.hr_c=object();self.p.hr_uc=object()
        with self.engine._checkpoint_session(self.p,'target'):
            self.assertEqual(self.p.latents_after_sampling,[])
            self.assertIsNone(self.p.hr_c)
            self.assertIsNone(self.p.hr_uc)
        self.assertEqual(self.modules.sd_models.unload_model_weights.call_count,2)

    def test_later_success_does_not_hide_earlier_failure(self):
        result=self.batch_result(2);first=result.images[0]
        self.engine._rewrite_saved=Mock()
        self.modules.processing.process_images.side_effect=[RuntimeError('first image failed'),types.SimpleNamespace(images=[types.SimpleNamespace(info={})],index_of_first_image=0)]
        self.engine._finish(self.p,result)
        self.assertIs(result.images[0],first)
        self.assertIn('Image 1: first image failed',result.comments)

    def test_transient_autolink_refresh_cannot_save_config(self):
        self.distinct_checkpoints()
        state=types.SimpleNamespace(checkpoint_change=False)
        save=Mock()
        def refresh(**kwargs):
            if not state.checkpoint_change: save()
        self.entry.refresh_model_loading_parameters.side_effect=refresh
        with patch.dict(sys.modules,{'model_autolink.integration':types.SimpleNamespace(_state=state)}):
            with self.engine._checkpoint_session(self.p,'target'): pass
        save.assert_not_called()
        self.assertFalse(state.checkpoint_change)

    def test_saved_only_grid_keeps_its_own_infotext(self):
        result=self.batch_result(1)
        self.p._pi_saved_grids=[(123,'GRID own main prompt and axis settings')]
        self.p._pi_saved_files={123:[('grid.png',None)]}
        self.modules.processing.process_images.return_value.images=[types.SimpleNamespace(info={})]
        grid=types.SimpleNamespace(info={})
        self.modules.images=types.SimpleNamespace(image_grid=Mock(return_value=grid))
        self.engine._rewrite_saved=Mock()
        self.engine._finish(self.p,result)
        grid_calls=[call for call in self.engine._rewrite_saved.call_args_list if call.args[0] is grid]
        self.assertTrue(grid_calls[0].args[1].startswith('GRID own main prompt and axis settings'))
        self.assertNotIn('Refiner error',grid_calls[0].args[1])

    def test_save_capture_preserves_grid_info(self):
        self.p._pi_saved_files={};self.p._pi_saved_grids=[]
        def save(image,**kwargs): return ('grid.png',None)
        self.engine._record_save(save,self.original,p=self.p,grid=True,info='GRID unique metadata')
        self.assertEqual(self.p._pi_saved_grids,[(id(self.original),'GRID unique metadata')])

    def test_stranded_native_setup_and_ui_are_inactive_after_unload(self):
        wrapped_setup=self.native.setup;wrapped_ui=self.native.ui
        original_ui=self.engine._original_uis[self.native]
        self.native.setup=lambda *args,**kwargs:wrapped_setup(*args,**kwargs)
        self.native.ui=lambda *args,**kwargs:wrapped_ui(*args,**kwargs)
        self.engine._unload()
        self.engine._setup=Mock(side_effect=AssertionError('inactive setup ran'))
        self.engine._ui=Mock(side_effect=AssertionError('inactive UI ran'))
        instance=self.native()
        self.p._pi_refiner=None
        instance.setup(self.p,True,'target',.75,0,'')
        instance.ui(False)
        self.original_setup.assert_called_once()
        original_ui.assert_called_once()
        self.assertIsNone(self.p._pi_refiner)

    def test_completed_base_pixel_cache_released_without_dropping_gallery(self):
        result=self.batch_result(2)
        self.p.pixels_after_sampling=list(result.images)
        self.modules.processing.process_images.return_value.images=[types.SimpleNamespace(info={})]
        self.engine._rewrite_saved=Mock()
        self.engine._finish(self.p,result)
        self.assertEqual(self.p.pixels_after_sampling,[])
        self.assertEqual(len(result.images),2)

    def test_ideogram_error_card_is_not_accepted_as_refined_image(self):
        self.modules.processing.process_images.return_value=types.SimpleNamespace(images=[types.SimpleNamespace(info={})],index_of_first_image=0,info='Ideogram 4 error: unsupported weights')
        self.run_pass()
        self.assertIs(self.pp.image,self.original)
        self.assertIn('unsupported weights',self.p.extra_generation_params['Refiner error'])

    def test_failed_base_error_card_does_not_start_refinement(self):
        result=self.batch_result(1);result.info='Ideogram 4 error: base failed'
        self.engine._finish(self.p,result)
        self.modules.processing.process_images.assert_not_called()

    def test_ideogram_actual_step_metadata_is_preserved(self):
        self.modules.processing.process_images.return_value.extra_generation_params={
            'Ideogram 4 refiner steps':2,'Ideogram 4 refiner recipe':'distilled-cfg1'}
        self.run_pass()
        self.assertEqual(self.p.extra_generation_params['Refiner steps'],2)
        self.assertEqual(self.p.extra_generation_params['Refiner Ideogram 4 refiner recipe'],'distilled-cfg1')

    def test_flux_reference_mode_does_not_claim_denoising_strength(self):
        self.modules.processing.process_images.return_value.extra_generation_params={
            'FLUX.3 reference mode':'direct conditioning; denoising strength unused'}
        self.run_pass()
        self.assertNotIn('Refiner denoising strength',self.p.extra_generation_params)
        self.assertIn('direct reference',self.p.extra_generation_params['Refiner input mode'])

    def test_ideogram_quant_policy_rejects_before_switch(self):
        selection=types.SimpleNamespace(_ci_is_ideogram4=lambda info:True)
        routing=types.SimpleNamespace(_enforce_loadable_policy=Mock(side_effect=ValueError('GGUF blocked')))
        self.modules.sd_models.get_closet_checkpoint_match.return_value=types.SimpleNamespace(filename='model.gguf')
        with patch.dict(sys.modules,{'pi_ideogram_forge.selection':selection,'pi_ideogram_forge.routing':routing}):
            result=self.batch_result(1)
            self.engine._finish(self.p,result)
        self.entry.checkpoint_change.assert_not_called()
        self.assertIn('GGUF blocked',result.comments)

    def test_flux_acknowledgements_are_not_bypassed(self):
        flux=types.SimpleNamespace(selected=lambda **kwargs:True,_setting=lambda *args:False)
        with patch.dict(sys.modules,{'flux3_forge.bridge':flux}):
            result=self.batch_result(1)
            self.engine._finish(self.p,result)
        self.entry.checkpoint_change.assert_not_called()
        self.assertIn('acknowledgement',result.comments)

    def fibo_modules(self, variant='edit-turbo'):
        contract = types.SimpleNamespace(is_fibo2_checkpoint=lambda name: True, validate_request=Mock())
        runtime = types.SimpleNamespace(_validate_metadata=lambda *args: ({'verified_official_variant':variant}, []))
        self.modules.sd_models.get_closet_checkpoint_match.return_value = types.SimpleNamespace(filename='packed.safetensors')
        self.p.width, self.p.height = 768, 512
        return {'forge_fibo2.contract':contract, 'forge_fibo2.runtime':runtime}

    def test_fibo_reference_uses_native_recipe_and_keeps_metadata(self):
        modules = self.fibo_modules()
        self.job.override_settings = {}
        self.modules.processing.process_images.return_value.extra_generation_params = {'FIBO precision':'int8_convrot'}
        with patch.dict(sys.modules, modules):
            self.run_pass()
        args = self.modules.processing.StableDiffusionProcessingImg2Img.call_args.kwargs
        self.assertEqual((args['steps'], args['cfg_scale'], args['sampler_name'], args['scheduler']), (4, 1, 'Euler', 'Simple'))
        self.assertIs(args['init_images'][0], self.original)
        self.assertEqual(self.job.override_settings['sd_vae_decode_method'], 'Full')
        self.assertEqual(modules['forge_fibo2.contract'].validate_request.call_count, 2)
        self.assertEqual(self.p.extra_generation_params['Refiner FIBO precision'], 'int8_convrot')
        self.assertNotIn('Refiner denoising strength', self.p.extra_generation_params)

    def test_fibo_dev_has_its_own_cfg_and_step_defaults(self):
        modules = self.fibo_modules('dev')
        with patch.dict(sys.modules, modules):
            profile = self.engine._target_profile('target', self.modules.sd_models.get_closet_checkpoint_match())
        self.p._pi_target_profile = profile
        self.assertEqual(profile['preset'], 'fibo2_dev')
        self.assertEqual(self.engine._target_cfg(self.p, 0), 5)
        self.assertEqual(self.engine._fast_steps(self.p, .25), 30)

    def test_fibo_cold_start_imports_enabled_runtime(self):
        modules = self.fibo_modules()
        runtime = modules.pop('forge_fibo2.runtime')
        modules['forge_fibo2.runtime'] = None
        with patch.dict(sys.modules, modules), patch('importlib.import_module', return_value=runtime) as load:
            profile = self.engine._target_profile('target', types.SimpleNamespace(filename='packed.safetensors'))
        load.assert_called_once_with('forge_fibo2.runtime')
        self.assertEqual(profile['steps'], 4)

    def test_refiner_runner_binds_target_fibo_without_rerunning_base_scripts(self):
        unrelated = types.SimpleNamespace(process=Mock())
        runner = self.engine._OptionsOnlyRunner(types.SimpleNamespace(alwayson_scripts=[unrelated]))
        model = types.SimpleNamespace(is_fibo2=True, bind=Mock())
        self.p.sd_model = model
        def generate(job):
            job.sd_model = model
            job.scripts.process(job)
            return types.SimpleNamespace(images=[self.output])
        self.p.scripts = types.SimpleNamespace(alwayson_scripts=[unrelated])
        self.modules.processing.process_images.side_effect = generate
        self.run_pass()
        model.bind.assert_called_once_with(self.job)
        unrelated.process.assert_not_called()
        self.assertIs(self.pp.image, self.output)

    def test_refiner_runner_preserves_fibo_masked_alpha_callback(self):
        self.p.sd_model = types.SimpleNamespace(is_fibo2=True)
        self.p._fibo2_edit_context = object()
        restore = Mock(return_value=self.output)
        with patch.dict(sys.modules, {'forge_fibo2.editing':types.SimpleNamespace(restore_masked_alpha=restore)}):
            self.engine._OptionsOnlyRunner(None).postprocess_image_after_composite(self.p, self.pp)
        restore.assert_called_once_with(self.original, self.p._fibo2_edit_context)
        self.assertIs(self.pp.image, self.output)

    def test_refiner_runner_leaves_other_model_lifecycle_alone(self):
        model = types.SimpleNamespace(bind=Mock())
        self.p.sd_model = model
        self.engine._OptionsOnlyRunner(None).process(self.p)
        model.bind.assert_not_called()

    def test_fibo_rejects_inherited_negative_before_model_switch(self):
        modules = self.fibo_modules()
        self.p.negative_prompt = 'base negative'
        with patch.dict(sys.modules, modules):
            self.run_pass()
        self.entry.checkpoint_change.assert_not_called()
        self.modules.processing.process_images.assert_not_called()
        self.assertIs(self.pp.image, self.original)
        self.assertIn('empty target negative', self.p.extra_generation_params['Refiner error'])

    def test_fibo_explicit_target_negative_preserves_base_negative(self):
        modules = self.fibo_modules()
        self.p.negative_prompt = 'base negative'
        self.p._pi_refiner_text = ('refine details', '')
        self.job.override_settings = {}
        with patch.dict(sys.modules, modules):
            self.run_pass()
        args = self.modules.processing.StableDiffusionProcessingImg2Img.call_args.kwargs
        self.assertEqual((args['prompt'], args['negative_prompt']), ('refine details', ''))
        self.assertEqual(self.p.negative_prompt, 'base negative')

    def test_qwen_and_klein_resolve_target_components_without_persistence(self):
        for family, preset in [('qwen_image21', 'qwen21'), ('flux2_klein4', 'klein'), ('flux2_klein9', 'klein')]:
            resolver = types.SimpleNamespace(
                checkpoint=Mock(return_value=(None, types.SimpleNamespace(label=family))),
                resolve=Mock(return_value=(['target-te', 'target-vae'], family, 'saved')))
            integration = types.SimpleNamespace(_resolver=resolver)
            with patch.dict(sys.modules, {'model_autolink.integration':integration}):
                profile = self.engine._target_profile('target', types.SimpleNamespace(filename='model.safetensors'))
            self.assertEqual(profile['preset'], preset)
            self.assertEqual(profile['modules'], ['target-te', 'target-vae'])
            resolver.resolve.assert_called_once_with('model.safetensors', persist=False)

    def test_missing_target_components_fail_before_unloading_base(self):
        resolver = types.SimpleNamespace(
            checkpoint=lambda *args: (None, types.SimpleNamespace(label='qwen_image21')),
            resolve=Mock(side_effect=RuntimeError('matching VAE missing')))
        self.modules.sd_models.get_closet_checkpoint_match.return_value = types.SimpleNamespace(filename='model.safetensors')
        with patch.dict(sys.modules, {'model_autolink.integration':types.SimpleNamespace(_resolver=resolver)}):
            self.run_pass()
        self.entry.checkpoint_change.assert_not_called()
        self.assertIn('matching VAE missing', self.p.extra_generation_params['Refiner error'])

    def test_native_saved_presets_work_without_autolink(self):
        info = types.SimpleNamespace(filename='native.safetensors')
        self.modules.sd_models.get_closet_checkpoint_match.return_value = info
        for preset in ('qwen21', 'klein'):
            self.opts.data = {'forge_checkpoint_' + preset:'saved alias',
                              'forge_additional_modules_' + preset:['native-te', 'native-vae']}
            with patch.dict(sys.modules, {'model_autolink.integration':None}):
                profile = self.engine._target_profile('target', info)
            self.assertEqual(profile['preset'], preset)
            self.assertEqual(profile['modules'], ['native-te', 'native-vae'])

    def test_missing_native_saved_components_fail_before_switch(self):
        info = types.SimpleNamespace(filename='native.safetensors')
        self.modules.sd_models.get_closet_checkpoint_match.return_value = info
        self.opts.data['forge_checkpoint_qwen21'] = 'saved alias'
        with patch.dict(sys.modules, {'model_autolink.integration':None}):
            self.run_pass()
        self.entry.checkpoint_change.assert_not_called()
        self.assertIs(self.pp.image, self.original)
        self.assertIn('matching text encoder/VAE', self.p.extra_generation_params['Refiner error'])

    def test_mask_blur_axes_reach_target_job(self):
        self.p.image_mask = object()
        self.p.mask_blur_x, self.p.mask_blur_y = 3, 9
        self.run_pass()
        self.assertEqual((self.job.mask_blur_x, self.job.mask_blur_y), (3, 9))

    def test_qwen_native_session_uses_target_modules_then_restores_base(self):
        self.distinct_checkpoints()
        lookup = self.modules.sd_models.get_closet_checkpoint_match.side_effect
        def info(name):
            result = lookup(name)
            result.filename = str(name) + '.safetensors'
            return result
        self.modules.sd_models.get_closet_checkpoint_match.side_effect = info
        resolver = types.SimpleNamespace(
            checkpoint=lambda *args: (None, types.SimpleNamespace(label='qwen_image21')),
            resolve=lambda *args, **kwargs: (['qwen-te', 'qwen-vae'], 'qwen_image21', 'saved'))
        original = dict(self.opts.data)
        with patch.dict(sys.modules, {'model_autolink.integration':types.SimpleNamespace(_resolver=resolver)}):
            with self.engine._checkpoint_session(self.p, 'target'):
                self.assertEqual(self.opts.data['forge_preset'], 'qwen21')
                self.assertEqual(self.opts.data['forge_additional_modules'], ['qwen-te', 'qwen-vae'])
                self.assertEqual(self.engine._target_setting(self.p, 'sampler', 'foreign-sampler'), 'Euler')
        self.assertEqual(self.opts.data, original)

    def test_fibo_worker_released_when_switching_away(self):
        self.distinct_checkpoints()
        release = Mock()
        with patch.dict(sys.modules, {'forge_fibo2.engine':types.SimpleNamespace(release_current=release)}):
            self.engine._release_for_switch('target')
        release.assert_called_once()

    def test_moss_profile_uses_native_scheduler_cfg_and_bounded_steps(self):
        info = types.SimpleNamespace(filename='moss-packed.safetensors')
        self.modules.sd_models.get_closet_checkpoint_match.return_value = info
        self.opts.data.update(forge_additional_modules_moss=['moss-te', 'flux2-vae'])
        with patch.dict(sys.modules, {'forge_moss.native.assets':types.SimpleNamespace(valid_checkpoint=lambda path: True), 'model_autolink.integration':None}):
            profile = self.engine._target_profile('target', info)
        self.p._pi_target_profile = profile
        self.assertEqual((profile['preset'], profile['scheduler'], profile['modules']), ('moss', 'MOSS', ['moss-te', 'flux2-vae']))
        self.assertEqual(self.engine._target_cfg(self.p, 0), 3)
        self.assertEqual(self.engine._fast_steps(self.p, .25), 8)

    def test_moss_missing_backend_fails_before_switch(self):
        self.engine._target_profile = lambda *args: {'family':'moss_image'}
        with patch.dict(sys.modules, {'forge_moss.native.patches':None}):
            self.run_pass()
        self.entry.checkpoint_change.assert_not_called()
        self.assertIn('MOSS backend is unavailable', self.p.extra_generation_params['Refiner error'])

    def test_moss_spectrum_forwards_only_owner_hooks(self):
        calls = []
        session = object()
        scope = {'Session':session, 'calls':calls}
        exec('class Spectrum:\n def process_before_every_sampling(self,p,**kwargs): calls.append("sampling")\n def postprocess_batch(self,p,*args,**kwargs): calls.append("batch")\n def postprocess(self,p,*args): calls.append("done")', scope)
        unrelated = types.SimpleNamespace(process_before_every_sampling=Mock())
        runner = self.engine._OptionsOnlyRunner(types.SimpleNamespace(alwayson_scripts=[scope['Spectrum'](), unrelated]))
        self.p.sd_model = types.SimpleNamespace(is_moss_image=True)
        with patch.dict(sys.modules, {'forge_moss.native.spectrum':types.SimpleNamespace(Session=session)}):
            runner.process_before_every_sampling(self.p)
            runner.postprocess_batch(self.p)
            runner.postprocess(self.p, None)
            self.p.sd_model.is_moss_image = False
            runner.process_before_every_sampling(self.p)
        self.assertEqual(calls, ['sampling', 'batch', 'done'])
        unrelated.process_before_every_sampling.assert_not_called()

    def test_moss_actual_spectrum_metadata_is_reported(self):
        self.modules.processing.process_images.return_value.extra_generation_params = {'MOSS Spectrum':'Off (no forecast qualified)', 'MOSS Spectrum calls':'8 real, 0 forecast, 0 rejected'}
        self.run_pass()
        self.assertEqual(self.p.extra_generation_params['Refiner MOSS Spectrum calls'], '8 real, 0 forecast, 0 rejected')

if __name__ == '__main__':
    unittest.main()
