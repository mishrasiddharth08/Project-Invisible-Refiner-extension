"""Adapter between Project Invisible Refiner and Forge's native MOSS engine."""
import importlib


def is_moss(profile):
    return (profile or {}).get('family') == 'moss_image'


def identify(filename, opener=None):
    """Recognize native MOSS checkpoints from Forge's two structural keys."""
    if not filename:
        return False
    if opener is None:
        from safetensors import safe_open
        opener = lambda path: safe_open(path, framework='pt', device='cpu')
    try:
        keys = tuple(opener(filename).keys())
    except (OSError, RuntimeError, ValueError):
        return False
    return (any(key.endswith('shared_block_modulation.weight') for key in keys) and
            any(key.endswith('layers.0.proj.ffn_up_gate.weight') for key in keys))


def normalize_profile(profile, scheduler_names=None):
    """Replace a removed saved scheduler with Forge's current MOSS default."""
    if not is_moss(profile):
        return profile
    if scheduler_names is None:
        try:
            scheduler_names = importlib.import_module('modules.sd_schedulers').schedulers_map
        except ImportError:
            # Minimal isolated test hosts do not install Forge's scheduler module.
            scheduler_names = {'Simple': object()}
    if profile.get('scheduler') not in scheduler_names:
        profile['scheduler'] = 'Simple'
        profile['scheduler_fallback'] = True
    return profile


def _component_kind(filename, opener):
    keys = tuple(opener(filename).keys())
    has = lambda suffix: any(key.endswith(suffix) for key in keys)
    if has('model.language_model.embed_tokens.weight') and has('model.language_model.layers.0.linear_attn.A_log'):
        return 'qwen35_9b'
    if has('bn.running_mean') and has('encoder.quant_conv.weight') and has('decoder.post_quant_conv.weight'):
        return 'flux2_vae'
    return None


def validate_modules(profile, opener=None):
    """Reject saved components that are structurally foreign to native MOSS."""
    if not is_moss(profile):
        return
    modules = list(profile.get('modules') or [])
    if len(modules) != 2:
        raise ValueError('MOSS Refiner requires exactly its Qwen3.5 9B text encoder and FLUX.2 VAE.')
    if opener is None:
        from safetensors import safe_open
        opener = lambda path: safe_open(path, framework='pt', device='cpu')
    try:
        kinds = {_component_kind(path, opener) for path in modules}
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError('MOSS Refiner could not validate its selected text encoder/VAE headers.') from exc
    if kinds != {'qwen35_9b', 'flux2_vae'}:
        raise ValueError('MOSS Refiner selected components are incompatible; choose Qwen3.5 9B and FLUX.2 VAE.')


def validate(profile, importer=importlib.import_module):
    """Require Forge's core MOSS engine, not the retired optional extension."""
    if not is_moss(profile):
        return
    try:
        module = importer('backend.diffusion_engine.moss_image')
    except (ImportError, OSError) as exc:
        raise ValueError(
            'This Forge build has no native MOSS image-to-image backend; update Forge before using MOSS as Refiner.'
        ) from exc
    if not callable(getattr(module, 'MOSSImage', None)):
        raise ValueError('Forge native MOSS image-to-image backend is incomplete.')
    validate_modules(profile)


def configure_job(job, profile):
    """Keep a MOSS refiner pass on conventional denoising-strength img2img."""
    if not is_moss(profile):
        return
    job.override_settings['moss_do_reference'] = 'Disable'
    job.extra_generation_params['MOSS refiner input mode'] = 'conventional img2img; reference mode disabled'
    if profile.get('scheduler_fallback'):
        job.extra_generation_params['MOSS refiner scheduler'] = 'Simple (stale saved scheduler ignored)'
