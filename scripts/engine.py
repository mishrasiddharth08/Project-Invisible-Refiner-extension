"""Use Forge Neo's native Refiner controls for a decoded-image second pass."""
import copy
import logging
from contextlib import contextmanager, ExitStack
from modules import scripts, script_callbacks, processing, shared, sd_models

log = logging.getLogger('PI-Refiner')
_original_setups = {}
_original_uis = {}
_wrappers = []
_enabled = True
_lifecycle_token = object()
_DLSS_TARGET = 'DLSS 5 Neural Rendering'


def _dlss_backend():
    import importlib.util
    import sys
    from pathlib import Path
    name = 'pi_refiner_dlss5_backend'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / 'lib' / 'dlss5.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


def _checkpoint_choices(choices):
    return list(choices) + ([] if _DLSS_TARGET in choices else [_DLSS_TARGET])


def _setup(self, p, enabled, checkpoint, switch, cfg, replacements='', separate_prompt=False,
           target_prompt='', target_negative='', target_steps=0, *args, **kwargs):
    # No persistent option writes or hidden LoRA substitution rules.
    p._pi_refiner = (checkpoint, float(switch), cfg) if enabled and checkpoint not in (None, '', 'None') else None
    p.refiner_checkpoint = p.refiner_switch_at = p.refiner_checkpoint_info = None
    p.refiner_cfg = None
    p._pi_refiner_text = (str(target_prompt), str(target_negative)) if separate_prompt else None
    p._pi_refiner_steps = target_steps


def _ui(self, is_img2img):
    import gradio as gr
    from modules.infotext_utils import PasteField
    from modules.ui_common import create_refresh_button
    from modules.ui_components import InputAccordion
    self.refresh_checkpoints()
    with InputAccordion(False, label='Refiner', elem_id=self.elem_id('enable')) as enabled:
        with gr.Row():
            checkpoint = gr.Dropdown(value='None', label='Checkpoint', choices=_checkpoint_choices(self.ckpts),
                                     elem_id=self.elem_id('checkpoint'))
            create_refresh_button(checkpoint, self.refresh_checkpoints,
                                  lambda: {'choices': _checkpoint_choices(self.ckpts)}, self.elem_id('checkpoint_refresh'))
        with gr.Row():
            switch = gr.Slider(value=.75, minimum=0, maximum=1, step=.025,
                               label='Switch at', elem_id=self.elem_id('switch_at'),
                               info='1 skips refinement. DLSS 5 uses 1 minus this value as enhancement intensity; CFG is unused. FLUX3 uses full-image reference conditioning.')
            cfg = gr.Slider(value=0, minimum=0, maximum=24, step=.5,
                            label='Refiner CFG', elem_id=self.elem_id('cfg'), info='Below 1 uses fast CFG 1. Higher values can cost more memory and time.')
        # Retain argument positions for existing presets, without a textbox.
        replacements = gr.State('')
        with gr.Accordion('Refiner target settings', open=False):
            gr.Markdown('FLUX3 distilled requires CFG1; experimental Base accepts CFG0..20. Both require an empty target negative. AutoNeg can stay enabled for the base stage: explicitly enable separate refiner prompts and leave the target negative empty. FLUX3 does not gain negative-prompt support from this separation.')
            separate = gr.Checkbox(value=False, label='Use separate refiner prompts',
                info='Explicit target prompts leave base prompts unchanged. An empty target negative clears only the refiner negative. Remove foreign base-model LoRA tags here; they are never silently removed.')
            target_prompt = gr.Textbox(value='', label='Refiner positive prompt', lines=3)
            target_negative = gr.Textbox(value='', label='Refiner negative prompt', lines=2)
            target_steps = gr.Slider(value=0, minimum=0, maximum=100, step=1,
                label='Refiner steps (0 = target default)',
                info='FLUX3 defaults to16 distilled flow-map steps or25 Base CFG steps, independent of Switch at. Other targets retain their bounded native refinement step policy.')
    self.infotext_fields = [PasteField(enabled, lambda d: 'Refiner' in d),
        PasteField(checkpoint, 'Refiner', api='refiner_checkpoint'),
        PasteField(switch, 'Refiner switch at', api='refiner_switch_at'),
        PasteField(cfg, 'Refiner CFG scale', api='refiner_cfg')]
    return [enabled, checkpoint, switch, cfg, replacements, separate, target_prompt, target_negative, target_steps]


def _value(p, name, index, fallback):
    values = getattr(p, name, None)
    return values[index] if values is not None and index < len(values) else fallback


def _prompt(prompt):
    return prompt


class Script(scripts.Script):
    def title(self):
        return 'PROJECT INVISIBLE - Refiner'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def postprocess_image_after_composite(self, p, pp, *args):
        # Finish the base batch before changing models. This avoids reloading
        # both models for every image, and never switches during a live sampler.
        if getattr(p, '_pi_dispatching', False) and not getattr(p, '_pi_external_finish', False):
            # Keep the callback object: later callbacks may replace pp.image.
            # Native galleries can interleave masks between real samples.
            slots = getattr(p, '_pi_deferred_slots', None)
            if slots is not None:
                slots.append(pp)
            return
        if not getattr(p, '_pi_external_finish', False):
            p._pi_standard_callback = True
        spec = getattr(p, '_pi_refiner', None)
        if not spec or getattr(p, '_pi_refining', False) or shared.state.interrupted or shared.state.skipped or getattr(shared.state, 'stopping_generation', False):
            return
        checkpoint, switch, cfg = spec
        strength = max(0.0, min(1.0, 1.0 - switch))
        if strength == 0:
            return
        if checkpoint == _DLSS_TARGET:
            try:
                backend = _dlss_backend()
                backend.executable()
                if getattr(p, 'image_mask', None) is not None:
                    raise ValueError('DLSS 5 currently supports full images; masked refinement is unsupported.')
                cancelled = lambda: bool(shared.state.interrupted or shared.state.skipped or getattr(shared.state, 'stopping_generation', False))
                with _checkpoint_session(p, checkpoint):
                    image = backend.enhance_image(pp.image, strength=strength, cancelled=cancelled)
                if cancelled():
                    return
                pp.image = image
                p.extra_generation_params.pop('Refiner error', None)
                p.extra_generation_params.update({'Refiner': _DLSS_TARGET,
                    'Refiner switch at': switch, 'Refiner DLSS intensity': strength,
                    'Refiner backend': 'Visual Enhancer CLI; Neural Rendering only',
                    'Refiner DLSS style': 0})
            except Exception as exc:
                p.extra_generation_params['Refiner error'] = str(exc)
                log.exception('DLSS 5 failed; retaining original image')
            return
        if sd_models.get_closet_checkpoint_match(checkpoint) is None:
            p.extra_generation_params['Refiner error'] = 'Checkpoint unavailable: ' + checkpoint
            log.error('Checkpoint unavailable: %s', checkpoint)
            return
        state_keys = ('job', 'job_no', 'job_count', 'sampling_step', 'sampling_steps',
                      'current_latent', 'current_image', 'id_live_preview')
        state = {k: getattr(shared.state, k) for k in state_keys if hasattr(shared.state, k)}
        job = None
        session = ExitStack()
        p._pi_refining = True
        try:
            session.enter_context(_checkpoint_session(p, checkpoint))
            log.info('Refining image %s with %s (strength %.3f)', pp.index + 1, checkpoint, strength)
            print(f'[PI-Refiner] Refining image {pp.index + 1}: {checkpoint}')
            i = pp.index
            job = processing.StableDiffusionProcessingImg2Img(
                init_images=[pp.image], resize_mode=0, width=pp.image.width, height=pp.image.height,
                prompt=_prompt(_target_text(p, i)[0]),
                negative_prompt=_target_text(p, i)[1],
                seed=_value(p, 'all_seeds', i, p.seed), subseed=_value(p, 'all_subseeds', i, p.subseed),
                subseed_strength=p.subseed_strength, steps=_fast_steps(p, strength), denoising_strength=strength,
                cfg_scale=cfg if cfg is not None and cfg >= 1 else 1.0,
                image_cfg_scale=getattr(p, 'image_cfg_scale', None),
                distilled_cfg_scale=_target_setting(p, 'dcfg', p.distilled_cfg_scale),
                sampler_name=_target_setting(p, 'sampler', p.sampler_name),
                scheduler=_target_setting(p, 'scheduler', p.scheduler), batch_size=1, n_iter=1,
                restore_faces=False, do_not_reload_embeddings=True,
                do_not_save_samples=True, do_not_save_grid=True,
                override_settings={'img2img_fix_steps': True, 'live_previews_enable': False})
            # Preserve the full-resolution inpainting mask where supplied.
            mask = getattr(p, 'image_mask', None)
            if mask is not None:
                job.image_mask = mask
                for name in ('inpainting_mask_invert', 'mask_blur', 'inpainting_fill',
                             'mask_round', 'inpaint_full_res', 'inpaint_full_res_padding',
                             'initial_noise_multiplier', 'latent_mask'):
                    if hasattr(p, name):
                        setattr(job, name, getattr(p, name))
            job._pi_refining = True
            job.scripts = _OptionsOnlyRunner(getattr(p, 'scripts', None))
            job.script_args = getattr(p, 'script_args', ())
            job.disable_extra_networks = getattr(p, 'disable_extra_networks', False)
            output = processing.process_images(job)
            if shared.state.interrupted or shared.state.skipped or getattr(shared.state, 'stopping_generation', False):
                return
            first = getattr(output, 'index_of_first_image', 0)
            generation_error = _generation_error(output)
            if generation_error:
                raise RuntimeError(generation_error)
            if first >= len(output.images):
                raise RuntimeError(getattr(output, 'info', None) or 'Refiner returned no sample')
            pp.image = output.images[first]
            p.extra_generation_params.pop('Refiner error', None)
            extra = getattr(output, 'extra_generation_params', None) or {}
            actual_steps = extra.get('FLUX.3 steps', extra.get('Ideogram 4 refiner steps', job.steps))
            p.extra_generation_params.update({'Refiner': checkpoint,
                'Refiner switch at': switch, 'Refiner denoising strength': strength,
                'Refiner CFG scale': job.cfg_scale, 'Refiner steps': actual_steps})
            p.extra_generation_params['Refiner sampler'] = getattr(job,'sampler_name',_target_setting(p,'sampler',p.sampler_name))
            if getattr(p, '_pi_refiner_text', None) is not None:
                p.extra_generation_params['Refiner prompts'] = 'explicit target prompts; base unchanged'
                p.extra_generation_params['Refiner negative policy'] = 'explicit target negative; base/AutoNeg unchanged'
            for key, value in extra.items():
                if key.startswith('Ideogram 4 refiner'):
                    p.extra_generation_params['Refiner ' + key] = value
            if extra.get('FLUX.3 reference mode', '').startswith('direct conditioning'):
                p.extra_generation_params.pop('Refiner denoising strength', None)
                p.extra_generation_params['Refiner input mode'] = 'FLUX.3 direct reference; denoising strength unused'
        except Exception as exc:
            p.extra_generation_params['Refiner error'] = str(exc)
            log.exception('Refinement failed; retaining original image')
        finally:
            try:
                if job is not None:
                    job.close()
            finally:
                try:
                    session.close()
                finally:
                    for key, value in state.items():
                        setattr(shared.state, key, value)
                    p._pi_refining = False


def _install():
    # Forge loads built-in files under dynamic module names. Patch the actual
    # registered class, not a separately imported copy (which duplicates UI).
    for data in scripts.scripts_data:
        cls = data.script_class
        if cls.__name__ == 'ScriptRefiner' and cls not in _original_setups:
            _original_setups[cls] = cls.setup
            _patch(cls, 'setup', lambda original: _setup)
            _original_uis[cls] = cls.ui
            _patch(cls, 'ui', lambda original: _ui)


script_callbacks.on_before_ui(_install)


def _unload():
    global _enabled
    _enabled = False
    for cls, original in _original_setups.items():
        if cls.setup is _setup:
            cls.setup = original
    for cls, original in _original_uis.items():
        if cls.ui is _ui:
            cls.ui = original
    for owner, name, original, wrapped in reversed(_wrappers):
        if getattr(owner, name, None) is wrapped:
            setattr(owner, name, original)
    _wrappers.clear()
    _original_uis.clear()
    _original_setups.clear()


script_callbacks.on_script_unloaded(_unload)

def _release_for_switch(checkpoint):
    """Release loaded external engines; native engines stay Forge-managed."""
    import sys
    current = shared.opts.data.get('sd_model_checkpoint')
    if sd_models.get_closet_checkpoint_match(current) == sd_models.get_closet_checkpoint_match(checkpoint):
        return
    for name, method in (('pi_qwen21.lib.runtime', 'release'),
                         ('pi_boogu_image.lib.pipeline', 'release'),
                         ('pi_sensenova.run', 'release'), ('pi_h3.runtime', 'release'),
                         ('pi_hunyuan.runtime', 'release'), ('flux3_forge.runtime', 'unload')):
        runtime = sys.modules.get(name)
        release = getattr(runtime, method, None)
        if callable(release):
            release()
    ideogram = sys.modules.get('pi_ideogram_forge.run')
    lifecycle = sys.modules.get('pi_ideogram_forge.lifecycle')
    cache = getattr(ideogram, '_PIPELINE_CACHE', None)
    if cache and lifecycle is not None:
        lifecycle.release_changed(cache, checkpoint)


def _unload_native(p):
    if getattr(p, '_pi_external_finish', False):
        unload = getattr(sd_models, 'unload_model_weights', None)
        if callable(unload):
            unload()


def _generation_error(output):
    error = getattr(output, '_pi_generation_error', None)
    if error:
        return str(error)
    # Older Ideogram builds returned a nonempty white error card on failure.
    info = getattr(output, 'info', '')
    return info if isinstance(info, str) and info.startswith('Ideogram 4 error:') else None


def _report_refiner_error(p, result, error):
    """Expose retained-base failure in result metadata and the running UI."""
    import sys
    message = str(error)
    p.extra_generation_params['Refiner error'] = message
    extra = getattr(result,'extra_generation_params',None)
    if isinstance(extra,dict): extra['Refiner error'] = message
    result.comments = (getattr(result,'comments','') or '') + '\nRefiner failed; base images retained: ' + message
    notice = getattr(sys.modules.get('gradio'),'Warning',None)
    if callable(notice):
        try: notice('Refiner failed; base images retained: '+message)
        except Exception: log.warning('Could not display the Refiner warning; failure remains in metadata.')


def _preflight(checkpoint, p=None):
    import sys
    info = sd_models.get_closet_checkpoint_match(checkpoint)
    if info is None:
        raise ValueError('Checkpoint unavailable: ' + str(checkpoint))
    if getattr(info, 'refiner_supports_img2img', None) is False:
        raise ValueError('This checkpoint cannot accept images; select an image-editing checkpoint.')
    boogu = sys.modules.get('pi_boogu_image.forge')
    if boogu is not None:
        asset = boogu.selected(name=checkpoint)
        if asset is not None and 'edit' not in asset.trunk:
            raise ValueError('Boogu Base/Turbo cannot refine images. Select Boogu Edit or Edit-Turbo.')
    ideogram = sys.modules.get('pi_ideogram_forge.selection')
    routing = sys.modules.get('pi_ideogram_forge.routing')
    if ideogram is not None and routing is not None and ideogram._ci_is_ideogram4(info):
        routing._enforce_loadable_policy(info.filename)
        if p is not None and getattr(p, 'image_mask', None) is not None:
            raise ValueError('Ideogram 4 experimental refinement does not support masks.')
    flux = sys.modules.get('flux3_forge.bridge')
    if flux is not None and flux.selected(name=checkpoint):
        for setting, label in (('flux3_caption_acknowledged', 'full-caption'),
                               ('flux3_reference_acknowledged', 'reference-image')):
            if not flux._setting(setting, False):
                raise ValueError('Enable the FLUX.3 ' + label + ' acknowledgement in Settings before refinement.')
        if p is not None:
            positive, negative = _target_text(p)
            values = negative if isinstance(negative, list) else [negative]
            if any(str(value or '').strip() for value in values):
                raise ValueError('FLUX.3 refiner cannot use negative prompts; clear them or explicitly use separate refiner prompts with an empty target negative.')
            import math
            cfg = (getattr(p, '_pi_refiner', None) or (None,None,0))[2]
            if cfg is not None and not math.isfinite(float(cfg)):
                raise ValueError('Refiner CFG must be finite.')
            asset=flux.selected(name=checkpoint) or {}
            asset=asset if isinstance(asset,dict) else {}
            if asset.get('model_family')=='base':
                if cfg is not None and not 0 <= float(cfg) <= 20:
                    raise ValueError('Experimental FLUX.3 Base refiner CFG must be0..20.')
            elif cfg is not None and cfg >= 1 and cfg != 1:
                raise ValueError('FLUX.3 refiner requires Refiner CFG1; use0 or1, not the base model CFG.')
            if getattr(p, 'subseed_strength', 0):
                raise ValueError('FLUX.3 refiner does not support variation seed strength; use0 for this pairing.')
            if getattr(p, 'image_mask', None) is not None:
                raise ValueError('FLUX.3 refiner supports full-frame references, not masked/inpainting refinement.')
            steps = getattr(p, '_pi_refiner_steps', 0)
            if not isinstance(steps,(int,float)) or not 0 <= steps <= 100 or int(steps)!=steps:
                raise ValueError('Refiner steps must be an integer from0 to100.')
            # Resolve names first, then reject declared foreign adapters without
            # loading any tensor payload or removing the user's tags.
            resolve = getattr(flux, '_resolve_adapters', None)
            if callable(resolve):
                from safetensors import safe_open
                for prompt in positive if isinstance(positive,list) else [positive]:
                    _, adapters = resolve(str(prompt or ''))
                    for adapter in adapters:
                        with safe_open(adapter['path'],framework='pt',device='cpu') as reader:
                            family = (reader.metadata() or {}).get('ss_base_model_version','').lower()
                        if family and family not in {'flux3','flux-3','flux_3','flux3-klein','flux-3-image-klein'}:
                            raise ValueError('FLUX.3 refiner cannot reuse a '+family+' adapter. Explicitly supply target prompts/adapters; base LoRAs are not removed automatically.')
    # Do not gate by filename, bit width, or format. Forge/the backend owns
    # NVFP4, ConvRot, GGUF, FP8 and future-format capability checks.
    return info


def _target_text(p, index=None):
    explicit = getattr(p, '_pi_refiner_text', None)
    if explicit is not None:
        if not explicit[0].strip(): raise ValueError('Explicit refiner positive prompt is empty.')
        return explicit
    if index is None:
        return (getattr(p,'all_prompts',None) or p.prompt,
                getattr(p,'all_negative_prompts',None) or p.negative_prompt)
    return (_value(p,'all_prompts',index,p.prompt),
            _value(p,'all_negative_prompts',index,p.negative_prompt))


def _target_profile(checkpoint, info):
    import sys
    flux = sys.modules.get('flux3_forge.bridge')
    if flux is not None and flux.selected(name=checkpoint):
        asset=flux.selected(name=checkpoint)
        is_base=isinstance(asset,dict) and asset.get('model_family')=='base'
        return {'family':'flux3','preset':'flux3','modules':[],
                'sampler':'Euler','scheduler':'Automatic','steps':25 if is_base else 16}
    return {}


def _target_setting(p, suffix, fallback):
    profile = getattr(p, '_pi_target_profile', None) or {}
    if suffix in profile: return profile[suffix]
    if not getattr(p, '_pi_changed_preset', False):
        return fallback
    preset = shared.opts.data.get('forge_preset')
    value = shared.opts.data.get(f'{preset}_i2i_{suffix}')
    return fallback if value is None else value


def _refresh_for_refiner(main_entry):
    """Autolink must not persist temporary target components during refresh."""
    import sys
    integration = sys.modules.get('model_autolink.integration')
    state = getattr(integration, '_state', None)
    had_guard = state is not None and hasattr(state, 'checkpoint_change')
    previous = getattr(state, 'checkpoint_change', False)
    try:
        if state is not None:
            state.checkpoint_change = True
        main_entry.refresh_model_loading_parameters(refresh=True)
    finally:
        if state is not None:
            if had_guard:
                state.checkpoint_change = previous
            else:
                del state.checkpoint_change


def _fast_steps(p, strength):
    import math
    explicit = getattr(p, '_pi_refiner_steps', 0)
    if explicit:
        if not isinstance(explicit,(int,float)) or int(explicit)!=explicit or not 1 <= explicit <= 100:
            raise ValueError('Refiner steps must be an integer from1 to100.')
        return int(explicit)
    profile = getattr(p, '_pi_target_profile', None) or {}
    if profile.get('family') == 'flux3': return profile['steps']
    base = max(1, int(_target_setting(p, 'step', p.steps)))
    return min(base, 8, max(2, math.ceil(base * strength)))


def _release_base_conditioning(p):
    # All base sampling/postprocessing has returned; these GPU references are
    # no longer needed. Keep PIL source images and masks for refinement.
    for name in ('sampler', 'c', 'uc', 'hr_c', 'hr_uc', 'init_latent', 'image_conditioning'):
        if hasattr(p, name):
            setattr(p, name, None)
    clear = getattr(p, 'clear_prompt_cache', None)
    if callable(clear):
        clear()
    latents = getattr(p, 'latents_after_sampling', None)
    if isinstance(latents, list):
        latents.clear()
    if hasattr(shared.state, 'current_latent'):
        shared.state.current_latent = None


@contextmanager
def _checkpoint_session(p, checkpoint):
    """One target switch per batch, exact option rollback, no eager base load."""
    if getattr(p, '_pi_active_checkpoint', None) == checkpoint:
        yield
        return
    if checkpoint == _DLSS_TARGET:
        _dlss_backend().executable()
        from backend import memory_management
        if getattr(p, '_pi_external_finish', False):
            _release_base_conditioning(p)
        _release_for_switch(checkpoint)
        memory_management.unload_all_models()
        memory_management.soft_empty_cache()
        p._pi_active_checkpoint = checkpoint
        try:
            yield
        finally:
            p._pi_active_checkpoint = None
        return
    from modules_forge import main_entry
    target = _preflight(checkpoint, p)
    profile = _target_profile(checkpoint, target)
    old_profile = getattr(p, '_pi_target_profile', None)
    options = copy.deepcopy(shared.opts.data)
    original_info = sd_models.get_closet_checkpoint_match(options.get('sd_model_checkpoint'))
    switched = original_info != target
    model_data = getattr(sd_models, 'model_data', None)
    loading = copy.copy(getattr(model_data, 'forge_loading_parameters', None))
    if isinstance(loading, dict) and isinstance(loading.get('additional_modules'), list):
        loading['additional_modules'] = list(loading['additional_modules'])
    dynamic = getattr(main_entry, 'dynamic_args', None)
    online_lora = getattr(dynamic, 'online_lora', None)
    restore_keys = set()
    try:
        if switched:
            if getattr(p, '_pi_external_finish', False):
                _release_base_conditioning(p)
            _release_for_switch(checkpoint)
            try:
                _unload_native(p)
                # Packed target weights must not inherit the source model's
                # forced FP8/FP16 storage conversion setting.
                shared.opts.data['forge_unet_storage_dtype'] = 'Automatic'
                main_entry.checkpoint_change(checkpoint, preset=None, save=False, refresh=False)
                if profile:
                    shared.opts.data['forge_preset'] = profile['preset']
                    shared.opts.data['forge_additional_modules'] = list(profile['modules'])
                target_preset = shared.opts.data.get('forge_preset')
                if target_preset != options.get('forge_preset'):
                    shared.opts.data['forge_unet_storage_dtype'] = shared.opts.data.get(
                        f'forge_unet_storage_dtype_{target_preset}', 'Automatic')
                _refresh_for_refiner(main_entry)
            finally:
                # Autolink can also change per-family module and preset keys.
                restore_keys = {key for key in options.keys() | shared.opts.data.keys()
                                if options.get(key) != shared.opts.data.get(key)
                                or (key in options) != (key in shared.opts.data)}
        p._pi_active_checkpoint = checkpoint
        p._pi_target_profile = profile
        p._pi_changed_preset = options.get('forge_preset') != shared.opts.data.get('forge_preset')
        yield
    finally:
        try:
            if switched:
                try:
                    _release_for_switch(options.get('sd_model_checkpoint'))
                finally:
                    # Backends can change options during generation too.
                    restore_keys.update(key for key in options.keys() | shared.opts.data.keys()
                                        if options.get(key) != shared.opts.data.get(key)
                                        or (key in options) != (key in shared.opts.data))
                    for key in restore_keys:
                        if key in options:
                            shared.opts.data[key] = options[key]
                        else:
                            shared.opts.data.pop(key, None)
                    if model_data is not None and loading is not None:
                        model_data.forge_loading_parameters = loading
                    if dynamic is not None and online_lora is not None:
                        dynamic.online_lora = online_lora
                    if getattr(p, '_pi_external_finish', False):
                        _release_base_conditioning(p)
                    _unload_native(p)
                    # Restore without calling Autolink's persistence/refresh
                    # hook. The next normal job loads the base on demand.
                    processing.need_global_unload = True
                    if not getattr(p, '_pi_external_finish', False):
                        sd_models.forge_model_reload()
        finally:
            p._pi_active_checkpoint = None
            p._pi_changed_preset = False
            p._pi_target_profile = old_profile


class _OptionsOnlyRunner:
    """Expose saved model options without rerunning postprocessors in pass two."""
    def __init__(self, runner):
        self.alwayson_scripts = getattr(runner, 'alwayson_scripts', [])

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _arm(p, runner=None, ui_args=None):
    if getattr(p, '_pi_refining', False):
        return
    runner = runner or getattr(p, 'scripts', None)
    values = ui_args if ui_args is not None else getattr(p, 'script_args', ())
    for script in getattr(runner, 'alwayson_scripts', ()):
        if type(script).__name__ != 'ScriptRefiner':
            continue
        args = values[script.args_from:script.args_to]
        if len(args) >= 4:
            _setup(script, p, *args)
            return
    checkpoint = getattr(p, 'refiner_checkpoint', None)
    if checkpoint not in (None, '', 'None'):
        switch = getattr(p, 'refiner_switch_at', None)
        _setup(None, p, True, checkpoint, .75 if switch is None else switch,
               getattr(p, 'refiner_cfg', 0))


def _record_save(original, image, *args, **kwargs):
    p = kwargs.get('p')
    if p is None:
        import inspect
        try:
            p = inspect.signature(original).bind_partial(image, *args, **kwargs).arguments.get('p')
        except (TypeError, ValueError):
            p = None
    result = original(image, *args, **kwargs)
    records = getattr(p, '_pi_saved_files', None)
    if records is not None and not getattr(p, '_pi_refining', False) and result and result[0]:
        records.setdefault(id(image), []).append((result[0], result[1] if len(result) > 1 else None))
        grids = getattr(p, '_pi_saved_grids', None)
        if grids is not None and kwargs.get('grid', False):
            grids.append((id(image), kwargs.get('info')))
    return result


def _rewrite_saved(image, info, records):
    """Replace precisely the files the base pass saved; no duplicate filenames."""
    import os
    import tempfile
    import shutil
    from pathlib import Path
    from modules import images
    staged = []
    backups = []
    committed = []
    seen = set()
    retain = set()
    def temporary(target):
        fd, name = tempfile.mkstemp(prefix='.pi-refiner-', suffix=target.suffix, dir=target.parent)
        os.close(fd)
        staged.append(name)
        return name
    plans = []
    try:
        # Stage every image/sidecar before changing any existing output.
        # This keeps gallery and disk consistent if one format or text write fails.
        for filename, textfile in records:
            target = Path(filename)
            identity = os.path.normcase(os.path.abspath(target))
            if identity in seen:
                continue
            seen.add(identity)
            name = temporary(target)
            images.save_image_with_geninfo(image, info, name, existing_pnginfo=dict(image.info))
            plans.append((target, name))
            if textfile and Path(textfile).is_file():
                text_target = Path(textfile)
                text_name = temporary(text_target)
                Path(text_name).write_text(info, encoding='utf-8')
                plans.append((text_target, text_name))
        for target, name in plans:
            backup = temporary(target)
            os.unlink(backup)
            try:
                # A hard link keeps the original bytes without a RAM buffer or
                # extra disk copy. Filesystems without hard links use a copy.
                os.link(target, backup)
            except OSError:
                shutil.copy2(target, backup)
            backups.append((target, backup))
        for (target, name), (_, backup) in zip(plans, backups):
            os.replace(name, target)
            committed.append((target, backup))
    except Exception as error:
        for target, backup in reversed(committed):
            try:
                os.replace(backup, target)
            except OSError:
                retain.add(backup)
                log.exception('Original output backup retained at %s', backup)
        if retain:
            raise RuntimeError('Output rollback failed; original backups retained: ' + ', '.join(sorted(retain))) from error
        raise
    finally:
        for name in staged:
            if name not in retain and os.path.exists(name):
                try:
                    os.unlink(name)
                except OSError:
                    log.warning('Temporary refiner file retained: %s', name)
    if records:
        image.already_saved_as = records[0][0]


def _finish(p, result):
    spec = getattr(p, '_pi_refiner', None)
    if (result is None or not spec or not getattr(result, 'images', None)
            or getattr(p, '_pi_standard_callback', False) or float(spec[1]) >= 1
            or shared.state.interrupted or shared.state.skipped
            or getattr(shared.state, 'stopping_generation', False)):
        return result
    base_error = _generation_error(result)
    if base_error:
        p.extra_generation_params['Refiner error'] = 'Base generation failed: ' + base_error
        return result
    p._pi_external_finish = True
    try:
        with _checkpoint_session(p, spec[0]):
            return _finish_images(p, result)
    except Exception as exc:
        _report_refiner_error(p,result,exc)
        log.exception('Refinement session failed; generated images retained')
        return result
    finally:
        p._pi_external_finish = False


def _finish_images(p, result):
    if result is None or getattr(p, '_pi_standard_callback', False):
        return result
    samples = getattr(result, 'images', None)
    if not samples:
        return result
    from types import SimpleNamespace
    first = getattr(result, 'index_of_first_image', 0)
    grid_info_original = getattr(result, 'info', '')
    if first and getattr(result, 'infotexts', None):
        grid_info_original = result.infotexts[0]
    deferred = getattr(p, '_pi_deferred_slots', None)
    if deferred:
        identities = {id(pp.image): pp.index for pp in deferred}
        slots = [(index, identities[id(image)]) for index, image in enumerate(samples)
                 if index >= first and id(image) in identities]
        compact = list(getattr(result, 'infotexts', None) or [])
        if compact and len(compact) != len(samples):
            sample_at = dict(slots)
            expanded = []
            current_info = grid_info_original
            for position in range(len(samples)):
                if position in sample_at:
                    info_position = sample_at[position] + first
                    current_info = compact[info_position] if info_position < len(compact) else getattr(result, 'info', '')
                expanded.append(current_info)
            result.infotexts = expanded
    else:
        count = len(getattr(result, 'all_seeds', None) or getattr(result, 'all_prompts', None) or samples[first:])
        slots = [(index, index - first) for index in range(first, min(len(samples), first + count))]
    # Do not retain another batch of original images while replacing results.
    p._pi_deferred_slots = None
    deferred = None
    pixels = getattr(p, 'pixels_after_sampling', None)
    if isinstance(pixels, list) and pixels is not samples:
        pixels.clear()
    changed = False
    failures = []
    # Takeover engines may construct their own prompts/seeds without filling p.
    old = {k: (hasattr(p, k), getattr(p, k, None)) for k in
           ('all_prompts', 'all_negative_prompts', 'all_seeds', 'all_subseeds')}
    try:
        for key in old:
            values = getattr(result, key, None)
            if values:
                setattr(p, key, values)
        for index, sample_index in slots:
            if shared.state.interrupted or shared.state.skipped or getattr(shared.state, 'stopping_generation', False):
                break
            source = samples[index]
            pp = SimpleNamespace(image=source, index=sample_index)
            Script().postprocess_image_after_composite(p, pp)
            if pp.image is source:
                error = p.extra_generation_params.get('Refiner error')
                if error:
                    failures.append(f'Image {sample_index + 1}: {error}')
                continue
            info_list = getattr(result, 'infotexts', [])
            info_index = index if len(info_list) == len(samples) else sample_index + first
            info = info_list[info_index] if info_index < len(info_list) else getattr(result, 'info', '')
            details = ', '.join(f'{k}: {v}' for k, v in p.extra_generation_params.items() if k.startswith('Refiner') and k != 'Refiner error')
            info = (info or '') + ', ' + details
            pp.image.info.update(getattr(source, 'info', {}))
            pp.image.info['parameters'] = info
            try:
                _rewrite_saved(pp.image, info, getattr(p, '_pi_saved_files', {}).get(id(source), []))
            except Exception as exc:
                p.extra_generation_params['Refiner error'] = 'Saving refined image failed: ' + str(exc)
                failures.append(f'Image {sample_index + 1}: saving refined image failed: {exc}')
                log.exception('Could not replace saved base image')
                continue
            samples[index] = pp.image
            changed = True
            if info_index < len(info_list):
                info_list[info_index] = info
            if sample_index == 0:
                if not first:
                    result.info = info
        grid_records = list(getattr(p, '_pi_saved_grids', None) or [])
        if changed and (first == 1 or grid_records):
            from modules import images
            grid = images.image_grid([samples[index] for index, _ in slots], getattr(p, 'batch_size', 1))
            if first == 1:
                grid_records.append((id(samples[0]), grid_info_original))
                samples[0] = grid
            details = ', '.join(f'{k}: {v}' for k, v in p.extra_generation_params.items() if k.startswith('Refiner') and k != 'Refiner error')
            grid_info = (grid_info_original or '') + ', ' + details
            grid.info['parameters'] = grid_info
            if first:
                result.info = grid_info
                if getattr(result, 'infotexts', None):
                    result.infotexts[0] = grid_info
            seen_grids = set()
            for record in grid_records:
                identity, original_info = record if isinstance(record, tuple) else (record, grid_info_original)
                if identity in seen_grids:
                    continue
                seen_grids.add(identity)
                saved_info = (original_info if original_info is not None else grid_info_original) or ''
                _rewrite_saved(grid, saved_info + ', ' + details, getattr(p, '_pi_saved_files', {}).get(identity, []))
        if failures:
            p.extra_generation_params['Refiner error'] = '; '.join(failures)
        if hasattr(result, 'extra_generation_params'):
            result.extra_generation_params.update(p.extra_generation_params)
        error = p.extra_generation_params.get('Refiner error')
        if error:
            _report_refiner_error(p,result,error)
        return result
    finally:
        for key, (existed, value) in old.items():
            if existed:
                setattr(p, key, value)
            elif hasattr(p, key):
                delattr(p, key)


def _dispatch(original, p, args, kwargs, runner=None):
    if getattr(p, '_pi_refining', False) or getattr(p, '_pi_dispatching', False):
        return original(*args, **kwargs)
    _arm(p, runner, args[2:] if runner is not None else None)
    if not getattr(p, '_pi_refiner', None):
        return original(*args, **kwargs)
    p._pi_dispatching = True
    p._pi_standard_callback = False
    p._pi_saved_files = {}
    p._pi_deferred_slots = []
    p._pi_saved_grids = []
    try:
        return _finish(p, original(*args, **kwargs))
    finally:
        p._pi_dispatching = False
        p._pi_saved_files = None
        p._pi_deferred_slots = None
        p._pi_saved_grids = None


def _patch(owner, name, make):
    from functools import wraps
    original = getattr(owner, name)
    if getattr(original, '_pi_refiner_owner', None) is _lifecycle_token:
        return
    routed = make(original)
    token = _lifecycle_token
    @wraps(original)
    def wrapped(*args, **kwargs):
        if not _enabled or token is not _lifecycle_token:
            return original(*args, **kwargs)
        return routed(*args, **kwargs)
    wrapped._pi_refiner_route = True
    wrapped._pi_refiner_owner = token
    setattr(owner, name, wrapped)
    _wrappers.append((owner, name, original, wrapped))


def _install_routes(*unused):
    import sys
    from modules import images
    _patch(scripts.ScriptRunner, 'run', lambda original:
          lambda runner, p, *args, **kwargs: _dispatch(original, p, (runner, p, *args), kwargs, runner))
    # Patch aliases after all generation extensions have installed their routes.
    for owner in [processing] + [sys.modules.get(n) for n in ('modules.txt2img', 'modules.img2img', 'modules.api.api')]:
        if owner is not None and hasattr(owner, 'process_images'):
            _patch(owner, 'process_images', lambda original:
                  lambda p, *args, **kwargs: _dispatch(original, p, (p, *args), kwargs))
    _patch(images, 'save_image', lambda original:
          lambda image, *args, **kwargs: _record_save(original, image, *args, **kwargs))
    print('[PI-Refiner] Native controls and direct-pipeline refinement installed.')


script_callbacks.on_app_started(_install_routes)
