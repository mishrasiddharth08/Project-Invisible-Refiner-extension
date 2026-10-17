"""Use Forge Neo's native Refiner controls for a decoded-image second pass."""
import copy
import logging
from contextlib import contextmanager, ExitStack
from modules import scripts, script_callbacks, processing, shared, sd_models

log = logging.getLogger('PI-Refiner')
_original_setups = {}
_original_uis = {}
_wrappers = []


def _setup(self, p, enabled, checkpoint, switch, cfg, replacements='', *args, **kwargs):
    # No persistent option writes or hidden LoRA substitution rules.
    p._pi_refiner = (checkpoint, float(switch), cfg) if enabled and checkpoint not in (None, '', 'None') else None
    p.refiner_checkpoint = p.refiner_switch_at = p.refiner_checkpoint_info = None
    p.refiner_cfg = None


def _ui(self, is_img2img):
    import gradio as gr
    from modules.infotext_utils import PasteField
    from modules.ui_common import create_refresh_button
    from modules.ui_components import InputAccordion
    self.refresh_checkpoints()
    with InputAccordion(False, label='Refiner', elem_id=self.elem_id('enable')) as enabled:
        with gr.Row():
            checkpoint = gr.Dropdown(value='None', label='Checkpoint', choices=self.ckpts,
                                     elem_id=self.elem_id('checkpoint'))
            create_refresh_button(checkpoint, self.refresh_checkpoints,
                                  lambda: {'choices': self.ckpts}, self.elem_id('checkpoint_refresh'))
        with gr.Row():
            switch = gr.Slider(value=.75, minimum=0, maximum=1, step=.025,
                               label='Switch at', elem_id=self.elem_id('switch_at'),
                               info='1 skips refinement. Image-editing models may use their own strength.')
            cfg = gr.Slider(value=0, minimum=0, maximum=24, step=.5,
                            label='Refiner CFG', elem_id=self.elem_id('cfg'), info='0 uses fast CFG 1. Higher values can cost more memory and time.')
        # Retain argument positions for existing presets, without a textbox.
        replacements = gr.State('')
    self.infotext_fields = [PasteField(enabled, lambda d: 'Refiner' in d),
        PasteField(checkpoint, 'Refiner', api='refiner_checkpoint'),
        PasteField(switch, 'Refiner switch at', api='refiner_switch_at'),
        PasteField(cfg, 'Refiner CFG scale', api='refiner_cfg')]
    return [enabled, checkpoint, switch, cfg, replacements]


def _value(p, name, index, fallback):
    values = getattr(p, name, None)
    return values[index] if values is not None and index < len(values) else fallback


def _prompt(prompt):
    return prompt


class Script(scripts.Script):
    def title(self):
        return 'PROJECT INVISIBLE — Refiner'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def postprocess_image_after_composite(self, p, pp, *args):
        # Finish the base batch before changing models. This avoids reloading
        # both models for every image, and never switches during a live sampler.
        if getattr(p, '_pi_dispatching', False) and not getattr(p, '_pi_external_finish', False):
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
                prompt=_prompt(_value(p, 'all_prompts', i, p.prompt)),
                negative_prompt=_value(p, 'all_negative_prompts', i, p.negative_prompt),
                seed=_value(p, 'all_seeds', i, p.seed), subseed=_value(p, 'all_subseeds', i, p.subseed),
                subseed_strength=p.subseed_strength, steps=_fast_steps(p, strength), denoising_strength=strength,
                cfg_scale=cfg if cfg is not None and cfg >= 1 else 1.0,
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
                for name in ('inpainting_mask_invert', 'mask_blur', 'inpainting_fill'):
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
            if first >= len(output.images):
                raise RuntimeError('Refiner returned no sample')
            pp.image = output.images[first]
            p.extra_generation_params.pop('Refiner error', None)
            p.extra_generation_params.update({'Refiner': checkpoint,
                'Refiner switch at': switch, 'Refiner denoising strength': strength,
                'Refiner CFG scale': job.cfg_scale, 'Refiner steps': job.steps})
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
            cls.setup = _setup
            _original_uis[cls] = cls.ui
            cls.ui = _ui


script_callbacks.on_before_ui(_install)


def _unload():
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
    for name in ('pi_qwen21.lib.runtime', 'pi_boogu_image.lib.pipeline',
                 'pi_sensenova.run', 'pi_h3.runtime'):
        runtime = sys.modules.get(name)
        release = getattr(runtime, 'release', None)
        if callable(release):
            release()


def _preflight(checkpoint):
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
    # Do not gate by filename, bit width, or format. Forge/the backend owns
    # NVFP4, ConvRot, GGUF, FP8 and future-format capability checks.
    return info


def _target_setting(p, suffix, fallback):
    if not getattr(p, '_pi_changed_preset', False):
        return fallback
    preset = shared.opts.data.get('forge_preset')
    value = shared.opts.data.get(f'{preset}_i2i_{suffix}')
    return fallback if value is None else value


def _fast_steps(p, strength):
    import math
    base = max(1, int(_target_setting(p, 'step', p.steps)))
    return min(base, 8, max(2, math.ceil(base * strength)))


def _release_base_conditioning(p):
    # All base sampling/postprocessing has returned; these GPU references are
    # no longer needed. Keep PIL source images and masks for refinement.
    for name in ('sampler', 'c', 'uc', 'init_latent', 'image_conditioning'):
        if hasattr(p, name):
            setattr(p, name, None)
    clear = getattr(p, 'clear_prompt_cache', None)
    if callable(clear):
        clear()
    if hasattr(shared.state, 'current_latent'):
        shared.state.current_latent = None


@contextmanager
def _checkpoint_session(p, checkpoint):
    """One target switch per batch, exact option rollback, no eager base load."""
    if getattr(p, '_pi_active_checkpoint', None) == checkpoint:
        yield
        return
    from modules_forge import main_entry
    target = _preflight(checkpoint)
    options = copy.deepcopy(shared.opts.data)
    original_info = sd_models.get_closet_checkpoint_match(options.get('sd_model_checkpoint'))
    switched = original_info != target
    model_data = getattr(sd_models, 'model_data', None)
    loading = copy.copy(getattr(model_data, 'forge_loading_parameters', None))
    dynamic = getattr(main_entry, 'dynamic_args', None)
    online_lora = getattr(dynamic, 'online_lora', None)
    restore_keys = set()
    try:
        if switched:
            if getattr(p, '_pi_external_finish', False):
                _release_base_conditioning(p)
            _release_for_switch(checkpoint)
            try:
                # Packed target weights must not inherit the source model's
                # forced FP8/FP16 storage conversion setting.
                shared.opts.data['forge_unet_storage_dtype'] = 'Automatic'
                main_entry.checkpoint_change(checkpoint, preset=None, save=False, refresh=True)
            finally:
                # Autolink can also change per-family module and preset keys.
                restore_keys = {key for key in options.keys() | shared.opts.data.keys()
                                if options.get(key) != shared.opts.data.get(key)
                                or (key in options) != (key in shared.opts.data)}
        p._pi_active_checkpoint = checkpoint
        p._pi_changed_preset = options.get('forge_preset') != shared.opts.data.get('forge_preset')
        yield
    finally:
        try:
            if switched:
                try:
                    _release_for_switch(options.get('sd_model_checkpoint'))
                finally:
                    for key in restore_keys:
                        if key in options:
                            shared.opts.data[key] = options[key]
                        else:
                            shared.opts.data.pop(key, None)
                    if model_data is not None and loading is not None:
                        model_data.forge_loading_parameters = loading
                    if dynamic is not None and online_lora is not None:
                        dynamic.online_lora = online_lora
                    # Restore without calling Autolink's persistence/refresh
                    # hook. The next normal job loads the base on demand.
                    processing.need_global_unload = True
                    if not getattr(p, '_pi_external_finish', False):
                        sd_models.forge_model_reload()
        finally:
            p._pi_active_checkpoint = None
            p._pi_changed_preset = False


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
    return result


def _rewrite_saved(image, info, records):
    """Replace precisely the files the base pass saved; no duplicate filenames."""
    import os
    import tempfile
    from pathlib import Path
    from modules import images
    for filename, textfile in records:
        target = Path(filename)
        fd, temporary = tempfile.mkstemp(prefix='.pi-refiner-', suffix=target.suffix, dir=target.parent)
        os.close(fd)
        try:
            images.save_image_with_geninfo(image, info, temporary, existing_pnginfo=dict(image.info))
            os.replace(temporary, target)
            if textfile and Path(textfile).is_file():
                Path(textfile).write_text(info, encoding='utf-8')
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _finish(p, result):
    spec = getattr(p, '_pi_refiner', None)
    if (result is None or not spec or not getattr(result, 'images', None)
            or getattr(p, '_pi_standard_callback', False) or float(spec[1]) >= 1
            or shared.state.interrupted or shared.state.skipped
            or getattr(shared.state, 'stopping_generation', False)):
        return result
    p._pi_external_finish = True
    try:
        with _checkpoint_session(p, spec[0]):
            return _finish_images(p, result)
    except Exception as exc:
        p.extra_generation_params['Refiner error'] = str(exc)
        result.comments = (getattr(result, 'comments', '') or '') + '\nRefiner failed: ' + str(exc)
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
    # Takeover engines may construct their own prompts/seeds without filling p.
    old = {k: (hasattr(p, k), getattr(p, k, None)) for k in
           ('all_prompts', 'all_negative_prompts', 'all_seeds', 'all_subseeds')}
    try:
        for key in old:
            values = getattr(result, key, None)
            if values:
                setattr(p, key, values)
        for index in range(first, len(samples)):
            if shared.state.interrupted or shared.state.skipped or getattr(shared.state, 'stopping_generation', False):
                break
            source = samples[index]
            pp = SimpleNamespace(image=source, index=index - first)
            Script().postprocess_image_after_composite(p, pp)
            if pp.image is source:
                continue
            info_list = getattr(result, 'infotexts', [])
            info = info_list[index] if index < len(info_list) else getattr(result, 'info', '')
            details = ', '.join(f'{k}: {v}' for k, v in p.extra_generation_params.items() if k.startswith('Refiner'))
            info = (info or '') + ', ' + details
            pp.image.info.update(getattr(source, 'info', {}))
            pp.image.info['parameters'] = info
            try:
                _rewrite_saved(pp.image, info, getattr(p, '_pi_saved_files', {}).get(id(source), []))
            except Exception as exc:
                p.extra_generation_params['Refiner error'] = 'Saving refined image failed: ' + str(exc)
                log.exception('Could not replace saved base image')
                continue
            samples[index] = pp.image
            if index < len(info_list):
                info_list[index] = info
            if index == first:
                result.info = info
        if first == 1 and len(samples) > 1:
            from modules import images
            old_grid = samples[0]
            samples[0] = images.image_grid(samples[1:], getattr(p, 'batch_size', 1))
            grid_info = getattr(result, 'info', '')
            _rewrite_saved(samples[0], grid_info, getattr(p, '_pi_saved_files', {}).get(id(old_grid), []))
        if hasattr(result, 'extra_generation_params'):
            result.extra_generation_params.update(p.extra_generation_params)
        error = p.extra_generation_params.get('Refiner error')
        if error:
            result.comments = (getattr(result, 'comments', '') or '') + '\nRefiner failed: ' + error
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
    try:
        return _finish(p, original(*args, **kwargs))
    finally:
        p._pi_dispatching = False
        p._pi_saved_files = None


def _install_routes(*unused):
    import sys
    from functools import wraps
    from modules import images
    def patch(owner, name, make):
        original = getattr(owner, name)
        if getattr(original, '_pi_refiner_route', False):
            return
        wrapped = wraps(original)(make(original))
        wrapped._pi_refiner_route = True
        setattr(owner, name, wrapped)
        _wrappers.append((owner, name, original, wrapped))
    patch(scripts.ScriptRunner, 'run', lambda original:
          lambda runner, p, *args, **kwargs: _dispatch(original, p, (runner, p, *args), kwargs, runner))
    # Patch aliases after all generation extensions have installed their routes.
    for owner in [processing] + [sys.modules.get(n) for n in ('modules.txt2img', 'modules.img2img', 'modules.api.api')]:
        if owner is not None and hasattr(owner, 'process_images'):
            patch(owner, 'process_images', lambda original:
                  lambda p, *args, **kwargs: _dispatch(original, p, (p, *args), kwargs))
    patch(images, 'save_image', lambda original:
          lambda image, *args, **kwargs: _record_save(original, image, *args, **kwargs))
    print('[PI-Refiner] Native controls and direct-pipeline refinement installed.')


script_callbacks.on_app_started(_install_routes)
