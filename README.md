# Project Invisible — Refiner for Forge Neo

**A second detail pass for ANY checkpoint Forge can load — without a separate tab, extra downloads or a Forge core fork.**

[Installation](#beginner-installation) · [Usage](#step-by-step-usage) · [Philosophy](#the-project-invisible-philosophy) · [Troubleshooting](#troubleshooting-and-support)

> This is my first public project and I am still learning. Please forgive any mistakes or rough edges. Kind, complete bug reports will help improve the project for everyone.

## Short description
A decoded-image second pass using Forge Neo's **existing Refiner** controls. No extra panel, dependencies, model downloads, or Forge core edits.

## Project overview
The extension replaces the native mid-sampling latent switch with a separate image-to-image pass. This allows different model families to exchange pixels where both Forge loading and the receiving model's image input are supported.

## The Project Invisible philosophy
Use the ordinary checkpoint/preset UI, Refiner accordion, Generate button, gallery, and saving workflow. Existing sampler, scheduler, CFG, distilled guidance, prompts, per-image seeds, and LoRA replacement rules are forwarded. Incompatible LoRAs still require replacement/removal rules.

## Testing status
Six automated regression tests pass with simulated Forge objects. They cover native controls, metadata-free samples, batch identity, cancellation, failure restoration, and unloading. **GPU generation, visual quality, model pairings, and memory/speed measurements remain unverified.**

The receiving checkpoint must support image-to-image in the installed Forge integration. Custom pipelines that bypass Forge's image callbacks are not supported by this version. Text-only models cannot become image refiners through checkpoint selection alone. Cross-family external VAE/text-encoder selections must be valid for the target model; automatic component selection depends on the installed loader/preset integration.

## Main behavior
The pass runs before Forge saves samples and builds grids, so normal saved images and gallery samples use the refined result. No dependence on PNG metadata. Each batch image keeps its own prompt and seed, including seed zero. Interrupt retains the original image. Errors are logged with a traceback and included in generation parameters. Failed passes retain the original.

## Memory management
Forge owns loading, quantization, and offloading. The extension does not convert or dequantize weights and does not maintain a second model cache. Different checkpoints require swapping back before the base job continues, which costs time. Peak memory and speed depend on the backend, model, resolution, and available RAM/VRAM. No promise of universal quantization support or zero extra memory is made.

## Speed and strength
Use **Switch at** as the preserved base-image fraction: 0.75 means 0.25 denoising; 1 means no refinement. This extension interprets it as a fraction even when native Forge is configured for sigma switching. Steps follow the base job and Forge's img2img step rules. Refiner CFG below 1 inherits base CFG. LoRA Replacements remain available.

## Image honesty
Refinement can change text, faces, and details. Metadata records the target, strength, and CFG. Inspect output quality before relying on a pairing.

## Step-by-step usage
1. Select your base checkpoint/preset normally.
2. Enable Forge's existing **Refiner** accordion. If hidden, enable Refiner visibility in Forge settings.
3. Select the target checkpoint; select the base checkpoint again for same-model refinement.
4. Start with **Switch at 0.75**. Adjust CFG and LoRA Replacements only if needed.
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

## Special thanks

Special thanks to u/malcolmrey and the r/malcolmrey community for support and inspiration.

Thanks also to the Forge Neo, Diffusers, Qwen, DeGrid, Spectrum and wider open-source communities whose work made this project possible.

If you contribute, test, or report issues and would like to be named here, say so and you will be added.
