"""Strict, non-persisting component selection for Ideogram 4 Refiner."""
from importlib import import_module
from pathlib import Path


def _header_kind(path, detect):
    try:
        header = detect.read_safetensors_header(path)
        keys = set(header) - {'__metadata__'}
    except (OSError, RuntimeError, ValueError):
        return None
    if detect._looks_like_qwen3vl_te(header):
        complete = (any(k.startswith(('model.layers.35.', 'language_model.layers.35.',
                                      'language_model.model.layers.35.', 'blk.35.')) for k in keys)
                    and any(k in keys for k in ('model.embed_tokens.weight',
                                                'language_model.embed_tokens.weight',
                                                'language_model.model.embed_tokens.weight',
                                                'token_embd.weight')))
        return 'te' if complete else None
    if detect._looks_like_flux2_vae(header):
        required = {'encoder.quant_conv.weight', 'decoder.post_quant_conv.weight'}
        native_towers = (any(k.startswith('encoder.down.0.block.') for k in keys)
                         and any(k.startswith('decoder.up.0.block.') for k in keys))
        return 'vae' if required <= keys and native_towers else None
    return None


def _validated_pair(paths, detect):
    found = {}
    for value in paths or ():
        path = Path(str(value))
        if path.is_file():
            kind = _header_kind(path, detect)
            if kind and kind not in found:
                found[kind] = str(path.resolve())
    if set(found) == {'vae', 'te'}:
        return [found['vae'], found['te']]
    return None


def _add_valid(found, paths, detect):
    for value in paths or ():
        path = Path(str(value))
        if path.is_file():
            kind = _header_kind(path, detect)
            if kind and kind not in found:
                found[kind] = str(path.resolve())


def _saved_modules():
    try:
        shared = import_module('modules.shared')
        return shared.opts.data.get('forge_additional_modules_ideogram4') or ()
    except (ImportError, AttributeError):
        return ()


def _autolink_modules(checkpoint):
    try:
        integration = import_module('model_autolink.integration')
        result = integration._resolver.resolve(checkpoint, persist=False)
        return result[0]
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError, TypeError):
        return ()


def _model_roots():
    roots = []
    for module_name in ('modules.paths', 'modules.shared'):
        try:
            module = import_module(module_name)
        except ImportError:
            continue
        value = getattr(module, 'models_path', None)
        if value and Path(value).is_dir() and Path(value) not in roots:
            roots.append(Path(value))
    return roots


def _local_candidates(detect, roots):
    dirs = detect.te_vae_search_dirs(roots)
    candidates = []
    for directory in dirs:
        candidates.extend(Path(directory).rglob('*.safetensors'))
    # Explicit Ideogram assets first; thereafter stable path ordering.
    def rank(path):
        name = path.name.casefold()
        return (0 if '--for-ideogram4.' in name else
                1 if 'ideogram4' in name else 2, str(path).casefold())
    candidates.sort(key=rank)
    return candidates


def resolve_modules(checkpoint=None, preferred=(), *, detect=None, roots=None,
                    autolink=None):
    """Return [native FLUX.2 VAE, complete Qwen3-VL-8B TE] without saving UI state."""
    detect = detect or import_module('pi_ideogram_lib.detect')
    found = {}
    for source in (preferred, _saved_modules()):
        _add_valid(found, source, detect)
        if set(found) == {'vae', 'te'}:
            return [found['vae'], found['te']]
    source = autolink if autolink is not None else _autolink_modules(checkpoint)
    _add_valid(found, source, detect)
    if set(found) == {'vae', 'te'}:
        return [found['vae'], found['te']]
    _add_valid(found, _local_candidates(detect, roots or _model_roots()), detect)
    if set(found) == {'vae', 'te'}:
        return [found['vae'], found['te']]
    raise ValueError(
        'Ideogram 4 Refiner requires a complete Qwen3-VL-8B text encoder and '
        'a native-layout FLUX.2 VAE; no validated pair was found.'
    )


def guidance_prompt(checkpoint, prompt, cfg, mode):
    """Choose Ideogram's full UNCOND twin or its replacement LoRA."""
    if float(cfg) == 1.0:
        return prompt, 'CFG 1 (UNCOND branch disabled)'
    detect = import_module('pi_ideogram_lib.detect')
    lora = import_module('pi_ideogram_forge.lora')
    roots = _model_roots()
    _clean, explicit = lora.split_lora_tags(str(prompt or ''), roots)
    if any(policy == 'uncond_replace' for _name, _strength, policy in explicit):
        return prompt, 'explicit uncond-replacement LoRA'

    mode = str(mode or 'Auto').strip().casefold()
    mode = {'full':'full transformer', 'ostris':'ostris uncond lora'}.get(mode, mode)
    if mode not in ('auto', 'full transformer', 'ostris uncond lora'):
        raise ValueError(f"Unknown Ideogram 4 UNCOND mode: {mode or '(empty)'}")
    if mode != 'ostris uncond lora':
        full = detect.find_matching_uncond(checkpoint, detect.model_search_dirs(roots))
        if full:
            return prompt, f'full transformer ({Path(full).name})'
        if mode == 'full transformer':
            raise ValueError(
                'Ideogram 4 Full transformer mode requires a matching UNCOND checkpoint '
                'with the same quantization; none is installed.'
            )

    adapter = lora.find_uncond_replacement_lora(roots)
    if not adapter:
        raise ValueError(
            'Ideogram 4 requires either a matching full UNCOND checkpoint or the '
            'Ostris uncond-replacement LoRA; neither is installed.'
        )
    tag = f'<lora:{Path(adapter).stem}:1>'
    _text, resolved = lora.split_lora_tags(tag, roots)
    if len(resolved) != 1 or resolved[0][2] != 'uncond_replace':
        raise ValueError('The installed Ideogram 4 uncond-replacement LoRA could not be resolved uniquely.')
    return f'{prompt.rstrip()} {tag}'.strip(), f'Ostris uncond LoRA ({Path(adapter).name})'
