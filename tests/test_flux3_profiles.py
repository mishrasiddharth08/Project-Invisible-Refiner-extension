import types,sys,tempfile,unittest
from unittest.mock import patch,Mock
import test_engine as support

class Flux3Profiles(unittest.TestCase):
 setUp=support.Tests.setUp
 run_pass=support.Tests.run_pass
 distinct_checkpoints=support.Tests.distinct_checkpoints
 batch_result=support.Tests.batch_result
 def flux(self,**extra):
  values=dict(selected=lambda **kwargs:True,_setting=lambda *args:True);values.update(extra)
  context=patch.dict(sys.modules,{'flux3_forge.bridge':types.SimpleNamespace(**values)});context.start();self.addCleanup(context.stop)
  return values
 def test_distinct_fl3_profile_owns_components_sampler_and_full_reference_steps(self):
  self.flux();self.distinct_checkpoints();original=dict(self.opts.data);self.opts.data.update(forge_preset='qwen21',forge_additional_modules=['base-TE','base-VAE']);original=dict(self.opts.data)
  with self.engine._checkpoint_session(self.p,'target'):
   self.assertEqual(self.opts.data['forge_preset'],'flux3');self.assertEqual(self.opts.data['forge_additional_modules'],[])
   self.assertEqual(self.engine._target_setting(self.p,'sampler','iPNDM'),'Euler');self.assertEqual(self.engine._target_setting(self.p,'scheduler','SGM Uniform'),'Automatic')
   for strength in [.1,.25,.75]:self.assertEqual(self.engine._fast_steps(self.p,strength),16)
  self.assertEqual(self.opts.data,original);self.assertIsNone(self.p._pi_target_profile)
 def test_fl3_invalid_cfg_stops_before_switch(self):
  self.flux();self.p._pi_refiner=('target',.75,5)
  result=self.batch_result(1);self.engine._finish(self.p,result)
  self.entry.checkpoint_change.assert_not_called();self.assertIn('CFG1',result.comments)
 def test_fl3_nonempty_negative_requires_explicit_target_override(self):
  self.flux();self.p.negative_prompt='base negative';self.p.all_negative_prompts=['base negative']
  result=self.batch_result(1);self.engine._finish(self.p,result);self.assertIn('negative prompts',result.comments);self.entry.checkpoint_change.assert_not_called()
 def test_separate_target_prompt_does_not_mutate_base_or_inherit_negative(self):
  self.flux();self.p.prompt='base <lora:foreign:1>';self.p.negative_prompt='base negative';self.p.all_negative_prompts=['base negative']
  self.engine._setup(None,self.p,True,'target',.75,0,'',True,'A complete target caption','',16)
  self.run_pass();args=self.modules.processing.StableDiffusionProcessingImg2Img.call_args.kwargs
  self.assertEqual(args['prompt'],'A complete target caption');self.assertEqual(args['negative_prompt'],'');self.assertEqual(args['steps'],16)
  self.assertEqual(self.p.prompt,'base <lora:foreign:1>');self.assertEqual(self.p.negative_prompt,'base negative')
 def test_explicit_empty_positive_rejected_before_switch(self):
  self.flux();self.p._pi_refiner_text=('','');result=self.batch_result(1);self.engine._finish(self.p,result);self.entry.checkpoint_change.assert_not_called();self.assertIn('positive prompt is empty',result.comments)
 def test_mask_and_variation_fail_before_checkpoint_handoff(self):
  self.flux()
  for key,value in [('image_mask',object()),('subseed_strength',.3)]:
   with self.subTest(key=key):
    setattr(self.p,key,value)
    with self.assertRaises(ValueError):self.engine._preflight('target',self.p)
    delattr(self.p,key)
  self.entry.checkpoint_change.assert_not_called()
 def test_target_steps_explicit_and_invalid_values(self):
  self.flux();self.p._pi_refiner_steps=12
  with self.engine._checkpoint_session(self.p,'target'):self.assertEqual(self.engine._fast_steps(self.p,.25),12)
  for value in [-1,101,1.5,'16']:
   self.p._pi_refiner_steps=value
   with self.subTest(value=value),self.assertRaises(ValueError):self.engine._preflight('target',self.p)
 def test_foreign_adapter_header_rejected_without_removal(self):
  import torch
  from pathlib import Path
  from safetensors.torch import save_file
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'foreign.safetensors';save_file({'x':torch.ones(1)},str(path),metadata={'ss_base_model_version':'qwen_image_2'})
   self.flux(_resolve_adapters=lambda prompt:(prompt,[{'path':str(path)}]))
   self.p.prompt='<lora:foreign:1>';self.p.all_prompts=['<lora:foreign:1>']
   with self.assertRaisesRegex(ValueError,'qwen_image_2'):self.engine._preflight('target',self.p)
   self.assertEqual(self.p.prompt,'<lora:foreign:1>');self.entry.checkpoint_change.assert_not_called()
 def test_reference_metadata_uses_actual_steps_and_removes_denoising_claim(self):
  self.flux();self.modules.processing.process_images.return_value.extra_generation_params={'FLUX.3 reference mode':'direct conditioning; denoising strength unused','FLUX.3 steps':16}
  self.run_pass();self.assertEqual(self.p.extra_generation_params['Refiner steps'],16);self.assertNotIn('Refiner denoising strength',self.p.extra_generation_params)
 def test_failure_is_visible_in_result_metadata_and_ui_without_losing_base(self):
  self.flux();self.p.negative_prompt='AutoNeg base negative';result=self.batch_result(1);result.extra_generation_params={};source=result.images[0];warning=Mock()
  with patch.dict(sys.modules,{'gradio':types.SimpleNamespace(Warning=warning)}):self.engine._finish(self.p,result)
  self.assertIs(result.images[0],source);self.assertIn('negative prompts',result.extra_generation_params['Refiner error']);warning.assert_called_once();self.assertIn('base images retained',result.comments)
 def test_pass_two_does_not_rerun_autoneg_or_mutate_base_negative(self):
  self.flux();process=Mock(side_effect=AssertionError('AutoNeg reran'))
  self.p.scripts=types.SimpleNamespace(alwayson_scripts=[types.SimpleNamespace(process=process)]);self.p.negative_prompt='AutoNeg base negative';self.p._pi_refiner_text=('Explicit full target caption','')
  def generate(job):
   job.scripts.process(job,True);self.assertEqual(job.negative_prompt,'');return types.SimpleNamespace(images=[self.output],index_of_first_image=0)
  def make_job(**kwargs):
   self.job.negative_prompt=kwargs['negative_prompt'];return self.job
  self.modules.processing.StableDiffusionProcessingImg2Img.side_effect=make_job;self.modules.processing.process_images.side_effect=generate
  self.run_pass();process.assert_not_called();self.assertEqual(self.p.negative_prompt,'AutoNeg base negative');self.assertIs(self.pp.image,self.output)
