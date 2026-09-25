"""PROJECT INVISIBLE — Refiner.

A model-agnostic second detail pass for any checkpoint Forge can load
(SD 1.5, SDXL, Flux, Qwen-Image, and anything else). One dropdown, no knobs.

Any permutation of checkpoints is possible: the pass can run on the SAME
checkpoint that produced the image (zero extra memory) or on ANY other
installed checkpoint (Forge swaps it in, refines, and swaps the base model
back). Forge's own memory management handles the VRAM/RAM offloading and
loading of both models, which keeps the swap as fast as Forge can be.

Philosophy (Project Invisible): no Forge core edits, no extra model downloads.
The refiner runs a short img2img-style pass over each finished image through
Forge's own pipeline, so every sampler, scheduler, CFG, LoRA and preset
combination works unchanged. If the pass fails for any reason the base image
is kept.
"""
import gradio as gr
from modules import scripts, script_callbacks, processing, sd_models, shared

TAG = '[PI-Refiner]'
SAME = '(same checkpoint as the base run)'

# mode -> (denoise strength, step multiplier, force_cfg1)
MODES = {
    'Off': None,
    'Turbo (fast)': (0.25, 0.25, True),    # light pass, quarter of the steps
    'Balanced': (0.35, 0.5, False),
    'Quality (best)': (0.5, 1.0, False),
}


class Script(scripts.Script):
    _pi_universal_refiner = True

    def title(self):
        return 'PROJECT INVISIBLE — Refiner'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion('Refiner', open=False,
                          elem_classes=['pi-refiner-panel']):
            gr.Markdown(
                'A fast second detail pass over the finished image. Works with '
                'every model, quantization, VRAM size, sampler, CFG and LoRA '
                'combination. No downloads, memory-efficient by design.')
            refine = gr.Dropdown(list(MODES), value='Off', label='Refine result',
                                 info='Turbo adds a moment; Quality adds roughly one extra pass. Off restores stock behavior.')
            choices = [SAME] + sorted(sd_models.checkpoints_list)
            refiner_ckpt = gr.Dropdown(choices, value=SAME, label='Refiner checkpoint',
                                       info='Any permutation: same checkpoint (zero extra memory) or any other installed checkpoint — Forge swaps it in for the pass and restores your base model after.')
        return [refine, refiner_ckpt]

    def postprocess(self, p, processed, refine=None, refiner_ckpt=SAME):
        spec = MODES.get(str(refine or 'Off'))
        if spec is None or not getattr(processed, 'images', None):
            return
        strength, step_mult, force_cfg1 = spec
        # Only real samples carry PNG info; skip grid crops and UI previews.
        targets = [im for im in processed.images
                   if getattr(im, 'info', None) and not getattr(im, '_pi_refined', False)]
        if not targets:
            return
        base_steps = max(1, int(getattr(p, 'steps', 20) or 20))
        steps = max(2, min(40, round(base_steps * step_mult)))
        cfg = 1.0 if force_cfg1 else float(getattr(p, 'cfg_scale', 1.0) or 1.0)
        sampler = getattr(p, 'sampler_name', 'Euler') or 'Euler'
        # Any checkpoint permutation: same model (zero extra memory) or any
        # other installed checkpoint. Forge's loader swaps and restores it.
        wanted = str(refiner_ckpt or SAME).strip()
        swap = None
        if wanted and wanted != SAME and wanted in sd_models.checkpoints_list:
            current = shared.opts.sd_model_checkpoint
            if wanted != current:
                swap = (current, wanted)
        print(f'{TAG} refining {len(targets)} image(s): denoise {strength}, '
              f'{steps} steps, CFG {cfg}, sampler {sampler}, '
              f'checkpoint {wanted if swap else "same as base"}')
        refined = {id(im): None for im in targets}
        try:
            if swap:
                sd_models.reload_model_weights(shared.sd_model, info=sd_models.checkpoints_list.get(swap[1]))
                if shared.sd_model.sd_checkpoint_info is None or getattr(shared.sd_model.sd_checkpoint_info, 'name', None) != swap[1]:
                    raise RuntimeError('checkpoint swap failed')
        except Exception as exc:
            print(f'{TAG} falling back to the base checkpoint: {exc}')
            swap = None
        try:
            for image in targets:
                try:
                    job = processing.StableDiffusionProcessingImg2Img(
                        init_images=[image],
                        resize_mode=0,
                        denoising_strength=strength,
                        prompt=getattr(p, 'prompt', '') or '',
                        negative_prompt='' if force_cfg1 else (getattr(p, 'negative_prompt', '') or ''),
                        cfg_scale=cfg,
                        steps=steps,
                        sampler_name=sampler,
                        width=getattr(image, 'width', None) or int(getattr(p, 'width', 512)),
                        height=getattr(image, 'height', None) or int(getattr(p, 'height', 512)),
                        seed=int(getattr(p, 'seed', -1) or -1),
                        subseed=int(getattr(p, 'subseed', -1) or -1),
                        do_not_save_samples=True,
                        do_not_reload_embeddings=True,
                    )
                    # Match the base run's scheduler when the attribute exists.
                    scheduler = getattr(p, 'scheduler_name', None)
                    if scheduler:
                        try: job.scheduler_name = scheduler
                        except Exception: pass
                    job.disable_extra_networks = True
                    run = processing.process_images(job)
                    if not run.images:
                        raise RuntimeError('empty result')
                    out = run.images[0]
                    out.info = dict(getattr(image, 'info', {}) or {})
                    out.info['Refiner'] = f'{refine} (denoise {strength}, {steps} steps)'
                    out._pi_refined = True
                    refined[id(image)] = out
                except Exception as exc:
                    print(f'{TAG} refiner pass failed; keeping the base image: {exc}')
        finally:
            if swap:
                try:
                    sd_models.reload_model_weights(shared.sd_model, info=sd_models.checkpoints_list.get(swap[0]))
                except Exception as exc:
                    print(f'{TAG} base checkpoint restore failed; select it manually: {exc}')
        if any(v is not None for v in refined.values()):
            processed.images = [refined.get(id(im)) or im for im in processed.images]


script_callbacks.on_script_unloaded(lambda: None)