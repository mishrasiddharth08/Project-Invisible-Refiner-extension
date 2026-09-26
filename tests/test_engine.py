import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

class Tests(unittest.TestCase):
    def setUp(self):
        self.opts = types.SimpleNamespace(data={'sd_model_checkpoint': 'base'})
        self.state = types.SimpleNamespace(interrupted=False, skipped=False, job_no=4)
        modules = types.ModuleType('modules')
        modules.scripts = types.SimpleNamespace(Script=object, AlwaysVisible=True)
        modules.script_callbacks = types.SimpleNamespace(on_script_unloaded=Mock(), on_before_ui=Mock())
        modules.shared = types.SimpleNamespace(opts=self.opts, state=self.state)
        modules.sd_models = types.SimpleNamespace(get_closet_checkpoint_match=Mock(return_value=object()), forge_model_reload=Mock())
        self.job = types.SimpleNamespace(close=Mock(), cfg_scale=7)
        self.output = object()
        modules.processing = types.SimpleNamespace(StableDiffusionProcessingImg2Img=Mock(return_value=self.job), process_images=Mock(return_value=types.SimpleNamespace(images=[self.output], index_of_first_image=0)))
        native = types.ModuleType('modules.processing_scripts.refiner')
        native.ScriptRefiner = type('ScriptRefiner', (), {'setup': Mock()})
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

if __name__ == '__main__':
    unittest.main()
