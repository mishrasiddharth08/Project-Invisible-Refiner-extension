# PROJECT INVISIBLE | GITHUB DESCRIPTION

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
Sixteen automated regression tests pass. They cover native controls, metadata-free samples, batch identity, cancellation, failure restoration, direct-pipeline routing, file replacement, and unloading. A live RTX 5090 test completed with Qwen 2.1 INT8 ConvRot as both base and refiner (512x512, 2 steps, seed 12345). Refined pixels differed from the matching unrefined run; the saved PNG matched the refined gallery and contained Refiner metadata. This verifies that route, not production visual quality, other pairings, or performance guarantees.

The receiving checkpoint must support image-to-image in the installed Forge integration. Direct pipelines such as Qwen 2.1 are handled at the generation entry points, even when they bypass Forge image callbacks. Their finished images are refined after the base call returns, then the same saved filenames and gallery entries are updated. This avoids calling another model while the base pipeline is still rendering. Pipelines must return Forge-compatible image results; image saving outside Forge's save function is not intercepted. Text-only models cannot become image refiners through checkpoint selection alone. Cross-family external VAE/text-encoder selections must be valid for the target model; automatic component selection depends on the installed loader/preset integration.

## Main behavior
For stock Forge, the pass runs before saving. For direct pipelines, the extension replaces the saved image atomically after successful refinement and updates the gallery. If interrupted before refinement, original images remain. No dependence on PNG metadata. Each batch image keeps its own prompt and seed, including seed zero. Interrupt retains the original image. Errors are logged with a traceback and included in generation parameters. Failed passes retain the original.

## Memory management
Forge owns loading, quantization, and offloading. The extension does not convert or dequantize weights and does not maintain a second model cache. Different checkpoints require switching, which costs time. The Qwen worker is released before switching to another checkpoint. Direct-pipeline base models are not forced through the stock Forge loader during restoration. Peak memory and speed depend on the backend, model, resolution, and available RAM/VRAM. No promise of universal quantization support or zero extra memory is made.

## Speed and strength
Use **Switch at** as the preserved base-image fraction: 0.75 means 0.25 denoising; 1 means no refinement. This extension interprets it as a fraction even when native Forge is configured for sigma switching. Steps follow the base job and Forge's img2img step rules. Refiner CFG below 1 inherits base CFG. Reference-image editing models such as Qwen 2.1 use their native image guidance and may not honor an img2img denoising fraction; a value of 1 always skips the pass. The LoRA rules UI is removed.

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
