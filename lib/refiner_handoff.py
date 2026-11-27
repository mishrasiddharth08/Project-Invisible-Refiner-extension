"""Fail-closed native latent handoff eligibility; never load tensor payloads.

Only conventional full SD1.x and native FLUX.1-dev BF16/FP16/FP32 pairs
are currently supported (FLUX requires identical complete BFL headers).
SDXL and every unrecognized route retain decoded-image refinement. Matching
architecture establishes a latent contract, not output quality or speed.
"""
import json
import os
import struct
from pathlib import Path

_UNET = 'model.diffusion_model.'
_REQUIRED = {
    _UNET + 'input_blocks.0.0.weight': (320, 4, 3, 3),
    _UNET + 'out.2.weight': (4, 320, 3, 3),
    _UNET + 'input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight': (320, 768),
    'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight': (49408, 768),
    'first_stage_model.encoder.conv_in.weight': (128, 3, 3, 3),
    'first_stage_model.decoder.conv_out.weight': (3, 128, 3, 3),
    'first_stage_model.quant_conv.weight': (8, 8, 1, 1),
    'first_stage_model.post_quant_conv.weight': (4, 4, 1, 1),
}
_FLUX_REQUIRED = {
    'img_in.weight': (3072, 64),
    'txt_in.weight': (3072, 4096),
    'guidance_in.in_layer.weight': (3072, 256),
    'final_layer.linear.weight': (64, 3072),
    'double_blocks.18.img_attn.qkv.weight': (9216, 3072),
    'single_blocks.37.linear1.weight': (21504, 3072),
}


def _identity(info):
    filename = getattr(info, 'filename', None)
    if not filename:
        return None
    return os.path.normcase(os.path.realpath(filename))


def _header(info, family='sd15'):
    path = Path(info.filename)
    if path.suffix.lower() != '.safetensors':
        return None
    with path.open('rb') as handle:
        length_data = handle.read(8)
        if len(length_data) != 8:
            return None
        length = struct.unpack('<Q', length_data)[0]
        if not 2 <= length <= 32 * 1024 * 1024:
            return None
        payload = handle.read(length)
        if len(payload) != length:
            return None
        header = json.loads(payload)
    if not isinstance(header, dict):
        return None
    metadata = header.pop('__metadata__', {})
    if not isinstance(metadata, dict):
        return None
    # Shape-compatible velocity/flow/LCM variants are not a verified contract.
    for key, value in metadata.items():
        allowed_prediction = ('eps', 'epsilon') if family == 'sd15' else ('flux', 'flow')
        if 'prediction' in key.lower() and str(value).lower() not in allowed_prediction:
            return None
        markers = ('lcm', 'lightning', 'hyper-sd', 'v_prediction', 'v-prediction', 'edm', 'zero_terminal_snr')
        if family == 'sd15': markers += ('rectified',)
        if key == 'modelspec.thumbnail': continue
        if any(marker in (key + ' ' + str(value)).lower() for marker in markers):
            return None
    shapes = {}
    for key, tensor in header.items():
        prefixes = (_UNET, 'cond_stage_model.', 'first_stage_model.', 'model_ema.') if family == 'sd15' else (
            'double_blocks.', 'single_blocks.', 'final_layer.', 'guidance_in.', 'img_in.', 'time_in.', 'txt_in.', 'vector_in.')
        if not key.startswith(prefixes):
            return None
        if not isinstance(tensor, dict) or not isinstance(tensor.get('shape'), list):
            return None
        shape = tensor['shape']
        if not all(isinstance(dim, int) and not isinstance(dim, bool) and dim > 0 for dim in shape):
            return None
        if tensor.get('dtype') not in ('F16', 'BF16', 'F32'):
            if not (key.endswith('.position_ids') and tensor.get('dtype') == 'I64'):
                return None
        shapes[key] = tuple(shape)
    required = _REQUIRED if family == 'sd15' else _FLUX_REQUIRED
    if any(shapes.get(key) != shape for key, shape in required.items()):
        return None
    if family == 'flux':
        # Complete vanilla FLUX.1-dev BFL architecture; exclude truncated,
        # quantized, packed, Kontext and reduced-block derivatives.
        if len(shapes) != 780:
            return None
        return shapes
    return {key: shape for key, shape in shapes.items() if key.startswith(_UNET)}


def handoff_compatible(base_info, target_info, model):
    """Return True only for a verified native SD1.x/FLUX.1-dev contract.

    Unsupported formats/engines, missing evidence, and header errors return
    False so the caller can use its decoded-image fallback.
    """
    try:
        engine = type(model)
        engines = {('backend.diffusion_engine.sd15', 'StableDiffusion'): ('sd15', 'SD15'),
                   ('backend.diffusion_engine.flux', 'Flux'): ('flux', 'Flux')}
        route = engines.get((engine.__module__, engine.__name__))
        if route is None:
            return False
        family, config_name = route
        config = getattr(model, 'model_config', None)
        if (type(config).__module__, type(config).__name__) != ('huggingface_guess.model_list', config_name):
            return False
        if family == 'sd15' and not getattr(model, 'is_sd1', False):
            return False
        if getattr(model, 'is_inpaint', True) or getattr(model, 'use_shift', True):
            return False
        if family == 'flux' and (not getattr(model, 'use_distilled_cfg_scale', False)
                                 or getattr(config, 'unet_config', {}).get('guidance_embed') is not True):
            return False
        if _identity(base_info) is None or _identity(base_info) != _identity(getattr(model, 'sd_checkpoint_info', None)):
            return False
        base = _header(base_info, family)
        target = _header(target_info, family)
        return bool(base and base == target)
    except (OSError, ValueError, TypeError, AttributeError, struct.error, OverflowError):
        return False
