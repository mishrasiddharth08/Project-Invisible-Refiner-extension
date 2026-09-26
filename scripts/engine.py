"""Use Forge Neo's native Refiner controls for a decoded-image second pass."""
import copy
import logging
from modules import scripts, script_callbacks, processing, shared, sd_models

log = logging.getLogger('PI-Refiner')
_original_setups = {}


def _setup(self, p, enabled, checkpoint, switch, cfg, replacements, *args, **kwargs):
    result = _original_setups[type(self)](self, p, enabled, checkpoint, switch, cfg, replacements, *args, **kwargs)
    p._pi_refiner = None
    if enabled and checkpoint not in (None, '', 'None'):
        p._pi_refiner = (checkpoint, float(switch), cfg)
        p.refiner_checkpoint = p.refiner_switch_at = p.refiner_checkpoint_info = None
    return result


def _value(p, name, index, fallback):
    values = getattr(p, name, None)
    return values[index] if values is not None and index < len(values) else fallback


def _prompt(prompt):
    from modules import extra_networks, sd_samplers_common
    text, networks = extra_networks.parse_prompt(prompt)
    loras = copy.deepcopy(networks.pop('lora', []))
    if loras:
        loras = sd_samplers_common.apply_lora_for_refiner(loras)
    else:
        _, _, appended = sd_samplers_common._parse_replacements(shared.opts.refiner_lora_replacement.strip())
        loras = [extra_networks.ExtraNetworkParams([n, w]) for n, w in appended]
    networks['lora'] = loras
    return text + ''.join(' <' + kind + ':' + ':'.join(map(str, param.items)) + '>'
                          for kind, params in networks.items() for param in params)


class Script(scripts.Script):
    def title(self):
        return 'PROJECT INVISIBLE — Refiner'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def postprocess_image_after_composite(self, p, pp, *args):
        spec = getattr(p, '_pi_refiner', None)
        if not spec or getattr(p, '_pi_refining', False) or shared.state.interrupted or shared.state.skipped:
            return
        checkpoint, switch, cfg = spec
        strength = max(0.0, min(1.0, 1.0 - switch))
        if strength == 0:
            return
        if sd_models.get_closet_checkpoint_match(checkpoint) is None:
            p.extra_generation_params['Refiner error'] = 'Checkpoint unavailable: ' + checkpoint
            log.error('Checkpoint unavailable: %s', checkpoint)
            return
        from modules_forge import main_entry
        keys = ('sd_model_checkpoint', 'forge_additional_modules', 'forge_unet_storage_dtype')
        options = {k: copy.deepcopy(shared.opts.data[k]) for k in keys if k in shared.opts.data}
        state_keys = ('job', 'job_no', 'job_count', 'sampling_step', 'sampling_steps',
                      'current_latent', 'current_image', 'id_live_preview')
        state = {k: getattr(shared.state, k) for k in state_keys if hasattr(shared.state, k)}
        job = None
        p._pi_refining = True
        try:
            main_entry.checkpoint_change(checkpoint, preset=None, save=False, refresh=True)
            i = pp.index
            job = processing.StableDiffusionProcessingImg2Img(
                init_images=[pp.image], resize_mode=0, width=pp.image.width, height=pp.image.height,
                prompt=_prompt(_value(p, 'all_prompts', i, p.prompt)),
                negative_prompt=_value(p, 'all_negative_prompts', i, p.negative_prompt),
                seed=_value(p, 'all_seeds', i, p.seed), subseed=_value(p, 'all_subseeds', i, p.subseed),
                subseed_strength=p.subseed_strength, steps=p.steps, denoising_strength=strength,
                cfg_scale=cfg if cfg is not None and cfg >= 1 else p.cfg_scale,
                distilled_cfg_scale=p.distilled_cfg_scale, sampler_name=p.sampler_name,
                scheduler=p.scheduler, batch_size=1, n_iter=1,
                do_not_save_samples=True, do_not_save_grid=True)
            # Preserve the full-resolution inpainting mask where supplied.
            mask = getattr(p, 'image_mask', None)
            if mask is not None:
                job.image_mask = mask
                for name in ('inpainting_mask_invert', 'mask_blur', 'inpainting_fill'):
                    if hasattr(p, name):
                        setattr(job, name, getattr(p, name))
            job._pi_refining = True
            job.scripts = None
            job.disable_extra_networks = getattr(p, 'disable_extra_networks', False)
            output = processing.process_images(job)
            if shared.state.interrupted or shared.state.skipped:
                return
            first = getattr(output, 'index_of_first_image', 0)
            if first >= len(output.images):
                raise RuntimeError('Refiner returned no sample')
            pp.image = output.images[first]
            p.extra_generation_params.pop('Refiner error', None)
            p.extra_generation_params.update({'Refiner': checkpoint,
                'Refiner denoising strength': strength, 'Refiner CFG scale': job.cfg_scale})
        except Exception as exc:
            p.extra_generation_params['Refiner error'] = str(exc)
            log.exception('Refinement failed; retaining original image')
        finally:
            try:
                if job is not None:
                    job.close()
            finally:
                try:
                    for key in keys:
                        if key in options:
                            shared.opts.data[key] = options[key]
                        else:
                            shared.opts.data.pop(key, None)
                    main_entry.refresh_model_loading_parameters(refresh=True)
                    sd_models.forge_model_reload()
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


script_callbacks.on_before_ui(_install)


def _unload():
    for cls, original in _original_setups.items():
        if cls.setup is _setup:
            cls.setup = original
    _original_setups.clear()


script_callbacks.on_script_unloaded(_unload)
