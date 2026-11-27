"""CPU-only compatibility tests using synthetic safetensors headers."""
import importlib.util
import json
import os
import struct
import tempfile
import types
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('pi_handoff_test', Path(__file__).parents[1] / 'lib' / 'refiner_handoff.py')
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.header = {key: {'shape': list(shape), 'dtype': 'F16', 'data_offsets': [0, 0]}
                       for key, shape in handoff._REQUIRED.items()}
        self.base = self.write('unrelated-name.safetensors')
        self.target = self.write('another-name.safetensors')
        cls = type('StableDiffusion', (), {'__module__': 'backend.diffusion_engine.sd15'})
        self.model = cls()
        self.model.is_sd1, self.model.is_inpaint, self.model.use_shift = True, False, False
        self.model.sd_checkpoint_info = self.base
        cfg = type('SD15', (), {'__module__': 'huggingface_guess.model_list'})
        self.model.model_config = cfg()

    def write(self, name, header=None):
        path = Path(self.folder.name) / name
        data = json.dumps(self.header if header is None else header).encode()
        path.write_bytes(struct.pack('<Q', len(data)) + data)
        return types.SimpleNamespace(filename=str(path))

    def compatible(self):
        return handoff.handoff_compatible(self.base, self.target, self.model)

    def test_same_standard_sd1_architecture_accepts_without_tensor_reads(self):
        self.assertTrue(self.compatible())

    def test_name_does_not_override_incompatible_context(self):
        self.header[next(key for key in self.header if 'attn2.to_k' in key)]['shape'][-1] = 1024
        self.target = self.write('sd15.safetensors')
        self.assertFalse(self.compatible())

    def test_all_unet_shapes_must_match(self):
        self.header[handoff._UNET + 'extra.weight'] = {'shape': [1], 'dtype': 'F16'}
        self.target = self.write('different.safetensors')
        self.assertFalse(self.compatible())

    def test_wrong_loaded_model_rejected(self):
        self.model.sd_checkpoint_info = self.target
        self.assertFalse(self.compatible())

    def test_sdxl_and_external_engines_rejected(self):
        for module, name in [('backend.diffusion_engine.sdxl', 'StableDiffusionXL'), ('backend.diffusion_engine.moss_image', 'MOSSImage')]:
            with self.subTest(name=name):
                self.assertFalse(handoff.handoff_compatible(self.base, self.target, type(name, (), {'__module__': module})()))

    def test_specialized_quantized_and_prediction_variants_rejected(self):
        variants = []
        for mutation in ('inpaint', 'quant', 'flow', 'lcm', 'vpred'):
            header = json.loads(json.dumps(self.header))
            if mutation == 'inpaint': header[handoff._UNET + 'input_blocks.0.0.weight']['shape'][1] = 9
            elif mutation == 'quant': header[handoff._UNET + 'out.2.weight']['dtype'] = 'F8_E4M3'
            elif mutation == 'flow': header['edm_mean'] = {'dtype': 'F32', 'shape': [1]}
            elif mutation == 'lcm': header['__metadata__'] = {'modelspec.title': 'LCM adapter merge'}
            else: header['__metadata__'] = {'prediction_type': 'v_prediction'}
            variants.append((mutation, header))
        for name, header in variants:
            with self.subTest(name=name):
                self.target = self.write(name + '.safetensors', header)
                self.assertFalse(self.compatible())

    def test_missing_or_malformed_evidence_falls_back(self):
        for data in (b'', struct.pack('<Q', 2**40), struct.pack('<Q', 2) + b'[]', struct.pack('<Q', 2) + b'xx'):
            with self.subTest(data=data):
                Path(self.target.filename).write_bytes(data)
                self.assertFalse(self.compatible())
        self.assertFalse(handoff.handoff_compatible(None, self.target, self.model))
        self.target.filename += '.missing'
        self.assertFalse(self.compatible())

    def test_missing_vae_or_clip_falls_back(self):
        for key in ('first_stage_model.post_quant_conv.weight', 'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight'):
            with self.subTest(key=key):
                header = dict(self.header)
                del header[key]
                self.target = self.write('missing-component.safetensors', header)
                self.assertFalse(self.compatible())

    def test_local_flux_srpo_krea_headers_accept_only_native_dev_engine(self):
        forge = Path(os.environ.get('FORGE_ROOT', Path(__file__).resolve().parents[3]))
        folder = forge / 'models' / 'Stable-diffusion' / 'FLUX SRPO'
        first = folder / 'FLUX.1-Dev-SRPO-BFL-BF16.safetensors'
        second = folder / 'FLUX.1-Krea-Dev-BF16.safetensors'
        if not first.is_file() or not second.is_file():
            self.skipTest('Optional installed native FLUX.1-dev checkpoint pair unavailable')
        self.base = types.SimpleNamespace(filename=str(first))
        self.target = types.SimpleNamespace(filename=str(second))
        engine = type('Flux', (), {'__module__': 'backend.diffusion_engine.flux'})
        self.model = engine()
        self.model.is_inpaint, self.model.use_shift = False, False
        self.model.use_distilled_cfg_scale = True
        self.model.sd_checkpoint_info = self.base
        config = type('Flux', (), {'__module__': 'huggingface_guess.model_list'})
        self.model.model_config = config()
        self.model.model_config.unet_config = {'guidance_embed': True}
        self.assertTrue(self.compatible())
        self.model.model_config.unet_config['guidance_embed'] = False
        self.assertFalse(self.compatible())
        self.model.model_config.unet_config['guidance_embed'] = True
        self.model.model_config = type('FluxSchnell', (), {'__module__': 'huggingface_guess.model_list'})()
        self.assertFalse(self.compatible())


if __name__ == '__main__':
    unittest.main()
