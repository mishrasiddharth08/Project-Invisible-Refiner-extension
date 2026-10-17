# Project Invisible — Refiner for Forge Neo

**A second detail pass for ANY checkpoint Forge can load — without a separate tab, extra downloads or a Forge core fork.**

[Installation](#beginner-installation) · [Usage](#step-by-step-usage) · [Philosophy](#the-project-invisible-philosophy) · [Troubleshooting](#troubleshooting-and-support)

> This is my first public project and I am still learning. Please forgive any mistakes or rough edges. Kind, complete bug reports will help improve the project for everyone.

## Short description
A decoded-image second pass using Forge Neo's **existing Refiner** controls. No extra panel, dependencies, model downloads, or Forge core edits.

## Project overview
The extension replaces the native mid-sampling latent switch with a separate image-to-image pass. This allows different model families to exchange pixels where both Forge loading and the receiving model's image input are supported.

## The Project Invisible philosophy
Use the ordinary checkpoint/preset UI, Refiner accordion, Generate button, gallery, and saving workflow. Existing sampler, scheduler, CFG, distilled guidance, prompts, per-image seeds, are forwarded. The LoRA rules textbox and explanatory panel are removed; prompt LoRAs are left unchanged. Use LoRAs compatible with the receiving model.

## Testing status
Thirty-two automated regression tests pass, including 64 simulated format pairings. They cover routing, batch switching, option restoration, target preset settings, cancellation, file replacement, and rejection of known text-only targets. These simulations do not load model weights or establish GPU compatibility.

An earlier live RTX 5090 test completed with Qwen 2.1 INT8 ConvRot as both base and refiner (512x512, 2 steps, seed 12345). Refined pixels differed from the matching unrefined run; the saved PNG matched the refined gallery and contained Refiner metadata. The October 17 speed and batch-switching changes have not been tested with real GPU generation; the GPU was already busy. Other pairings, NVFP4 generation, production image quality, and speed/memory improvements remain unmeasured.

The receiving checkpoint must support image-to-image in the installed Forge integration. Direct pipelines such as Qwen 2.1 are handled at the generation entry points, even when they bypass Forge image callbacks. Their finished images are refined after the base call returns, then the same saved filenames and gallery entries are updated. This avoids calling another model while the base pipeline is still rendering. Pipelines must return Forge-compatible image results; image saving outside Forge's save function is not intercepted. Text-only models cannot become image refiners through checkpoint selection alone. Cross-family external VAE/text-encoder selections must be valid for the target model; automatic component selection depends on the installed loader/preset integration.

## Main behavior
Both stock Forge and direct pipelines finish the base batch before refinement. The target is selected once for the batch, then images are refined sequentially. Saved images are replaced atomically after successful refinement and gallery entries are updated. If interrupted before refinement, original images remain. No dependence on PNG metadata. Each batch image keeps its own prompt and seed, including seed zero. Errors retain the original image and include a diagnostic message.

## Memory management
Forge owns loading, quantization, and offloading. The extension does not convert or dequantize weights or keep another model cache. Same-checkpoint refinement avoids selection/reload calls. Different checkpoints use one target selection per batch; the base model loads on demand for the next generation. Completed base conditioning, sampler, and latent references are released before cross-model refinement; same-model prompt caches are retained. Loaded Qwen, Boogu, SenseNova and H3 external engines are released through their own cleanup functions when switching. Other native engines remain Forge-managed. Selection changes, per-family module settings, preset keys, and loader parameters are restored after success or failure. Peak memory and speed still depend on the backend, model, resolution, and hardware.

## Quantization and model combinations

The refiner passes checkpoint identities to the existing loader without a format whitelist. INT8 ConvRot, NVFP4, ConvRot W4A4, FP8, MXFP8, GGUF and future formats therefore use the installed backend's support. A different target uses automatic storage precision to avoid inheriting a forced conversion setting from the base model; the original setting is restored afterwards. No checkpoints are converted or downloaded.

Any base can hand off decoded pixels, but the target must actually accept images and have compatible VAE/text-encoder components, adapters and kernels. Boogu Base/Turbo are text-only targets: use Edit/Edit-Turbo instead. Unknown format or architecture support must be supplied by Forge or the model extension, not invented by this refiner. If selection changes the preset, saved target img2img sampler, scheduler, steps and guidance defaults are used where available; an explicit Refiner CFG still wins.

Forge Neo's upstream supported-format list is maintained in its [README](https://github.com/Haoming02/sd-webui-forge-classic/blob/neo/README.md). NVIDIA describes [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) as a distinct floating-point format; it is not interchangeable with integer INT4. A filename alone is not evidence that a checkpoint's encoding, model architecture, and kernels are compatible.

## Speed and strength
Use **Switch at** as the preserved base-image fraction: 0.75 means 0.25 denoising; 1 means no refinement. This extension interprets it as a fraction even when native Forge is configured for sigma switching. Refinement uses a speed-first budget: ceil(base or target preset steps times denoising strength), normally 2 to 8 steps, or 1 if the source requests 1. Native img2img uses fixed refinement steps so denoising does not reduce this budget a second time. CFG 0 automatically uses CFG 1 to reduce guidance work; an explicit positive CFG is respected. Stock live previews, repeated face restoration, and embedding rescans are disabled for the refinement pass. These shortcuts can reduce image quality. Reference-image editing models such as Qwen 2.1 use their native image guidance and may not honor an img2img denoising fraction; a value of 1 always skips the pass. The LoRA rules UI is removed.

## Image honesty
Refinement can change text, faces, and details. Metadata records the target, strength, and CFG. Inspect output quality before relying on a pairing.

## Step-by-step usage
1. Select your base checkpoint/preset normally.
2. Enable Forge's existing **Refiner** accordion. If hidden, enable Refiner visibility in Forge settings.
3. Select the target checkpoint; select the base checkpoint again for same-model refinement.
4. Start with **Switch at 0.75**. Adjust CFG only if needed.
5. Press **Generate**. Check generation parameters and the terminal for errors.

## Beginner installation

1. Stop Forge completely.
2. Choose **one** way — never both: **Install from URL** with this repository's URL, or **Download ZIP** and extract into `sd-webui-forge-classic/extensions/`.
3. Avoid double nesting. The correct path ends with: `extensions/<folder>/scripts/engine.py`.
4. Start Forge normally. No extra dependencies are installed.
5. Refresh your browser with **Ctrl+F5** after updates.

## Troubleshooting and support

If a problem occurs, open a GitHub issue and paste the complete error from the DOS/terminal window. The report must begin at the first error line and continue through the final traceback line. A single final sentence is usually not enough to identify the cause.

Reports should also include:

- Windows and Forge Neo versions;
- GPU, VRAM and system RAM;
- base and refiner checkpoint filenames;
- resolution and sampling steps;
- whether Save GPU memory / offloading was enabled;
- the selected Refiner mode and checkpoint;
- exact reproduction steps.

Private usernames, paths, prompts, tokens and images should be removed before posting publicly.

Users may also paste the complete error into ChatGPT, Claude, Gemini or Grok and ask for a beginner-friendly explanation.

## Contributions

Bug reports, documentation corrections and focused code improvements are welcome. Contributors should state exactly what was tested, avoid describing untested modes as working, preserve upstream licenses and never commit model weights, generated images, dependency folders, logs, access tokens or private information.

## License and independence

This extension is independent and unofficial. It is not an official product of any model vendor or of Forge. The repository does not relicense model weights; every model keeps its own license. Extension code: [Apache License 2.0](LICENSE).

## A humble note from the author

This is a first public attempt by a non-programmer learning through experimentation and community help. Please forgive mistakes. Constructive feedback, patient explanations and complete error reports are welcomed with gratitude.

## Special Thanks

- [**r/sdforall**](https://www.reddit.com/r/sdforall/) - community discussion and testing
- [**r/SECourses**](https://www.reddit.com/r/SECourses/) - community discussion and testing
- [**r/malcolmrey**](https://www.reddit.com/r/malcolmrey/) - community discussion and testing
- [**Haoming02 / sd-webui-forge-classic (neo branch)**](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) - the Forge Neo tree this extension targets
- [**ComfyUI**](https://github.com/comfyanonymous/ComfyUI) - reference for upstream sampler/scheduler coverage
- The Forge / AUTOMATIC1111 community - for the extension ecosystem this plugs into

Special thanks to u/malcolmrey and the r/malcolmrey community for support and inspiration.

Thanks also to the Forge Neo, Diffusers, Qwen, DeGrid, Spectrum and wider open-source communities whose work made this project possible.

If you contribute, test, or report issues and would like to be named here, say so and you will be added.
