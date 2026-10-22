# Project Invisible - Refiner for Forge Neo

**A decoded-image second pass through Forge Neo Refiner, plus optional community DLSS enhancement in Refiner and Extras.**

[Installation](#beginner-installation) � [Usage](#step-by-step-usage) � [Philosophy](#the-project-invisible-philosophy) � [Troubleshooting](#troubleshooting-and-support)

> This is my first public project and I am still learning. Please forgive any mistakes or rough edges. Kind, complete bug reports will help improve the project for everyone.

## Short description
A decoded-image second pass using Forge Neo's **existing Refiner** controls. Checkpoint refinement reuses Forge dependencies. Optional DLSS enhancement requires a separately supplied external runtime; no Forge core edits are required.

## Project overview
The extension replaces the native mid-sampling latent switch with a separate image-to-image pass. This allows different model families to exchange pixels where both Forge loading and the receiving model's image input are supported.

## The Project Invisible philosophy
Use the ordinary checkpoint/preset UI, Refiner accordion, Generate button, gallery, and saving workflow. Existing sampler, scheduler, CFG, distilled guidance, prompts, per-image seeds, are forwarded. The LoRA rules textbox and explanatory panel are removed; prompt LoRAs are left unchanged. Use LoRAs compatible with the receiving model.

## Testing status
For this publication, all 81 mocked CPU regression tests passed. The October 22 implementation task also recorded those tests passing.
It also recorded a live RTX 5090/Qwen 2.1 INT8 ConvRot generation with changed
pixels and matching saved/gallery output, an Extras upload/save check preserving
size, alpha and generation parameters, and five distinct existing-image outputs
with unchanged original file hashes. These are prior task results, not a new GPU
run for this publication. They do not establish universal model compatibility,
visual improvement, speedup or reduced memory use.

JPEG testing exposed EXIF/DPI changes in Forge's export path. General preservation
of those fields remains unverified. A separate short-video backend check retained
eight frames and audio; the Forge video UI is unfinished.

## Optional DLSS 5 enhancement
Select **DLSS 5 Neural Rendering** in Refiner for generated images, or open
**Extras**, upload existing images, enable **DLSS 5 Neural Rendering** and press
**Generate**. Batch-image input uses native Extras. Refiner Switch at 0.75 means
intensity 0.25; 1 skips enhancement. Extras has its own intensity control.

This uses Merserk's unsigned community Visual Enhancer v18 runtime. It is
independent of NVIDIA and is not an NVIDIA-endorsed Forge integration. Obtain
the complete runtime separately and set `PI_REFINER_DLSS5_HOME`; nothing is
downloaded or installed automatically. See [setup, requirements and limits](docs/DLSS5.md).
No runtime binaries, model weights or test pictures are distributed here.

The receiving checkpoint must support image-to-image in the installed Forge integration. Direct pipelines such as Qwen 2.1 are handled at the generation entry points, even when they bypass Forge image callbacks. Their finished images are refined after the base call returns, then the same saved filenames and gallery entries are updated. This avoids calling another model while the base pipeline is still rendering. Pipelines must return Forge-compatible image results; image saving outside Forge's save function is not intercepted. Text-only models cannot become image refiners through checkpoint selection alone. Cross-family external VAE/text-encoder selections must be valid for the target model; automatic component selection depends on the installed loader/preset integration.

## Main behavior
Both stock Forge and direct pipelines finish the base batch before refinement. The target is selected once for the batch, then images are refined sequentially. Images and sidecars are staged together, with hard-link backups (copy fallback) and rollback on write failures. Gallery entries update only after saving succeeds. Returned masks/composites are excluded from refinement and rebuilt image grids. Grid metadata keeps its original text. If interrupted before refinement, original images remain. No dependence on PNG metadata. Each batch image keeps its own prompt and seed, including seed zero. Errors retain the original image and include a diagnostic message.

## Memory management
Forge owns loading, quantization, and offloading. The extension does not convert or dequantize weights or keep another model cache. Same-checkpoint refinement avoids selection/reload calls. Different checkpoints use one target selection per batch; the base model loads on demand for the next generation. Completed base conditioning, sampler, and latent references are released before cross-model refinement; same-model prompt caches are retained. Loaded Qwen, Boogu, SenseNova, H3, Hunyuan, Flux 3 and Ideogram external engines are released through their own cleanup functions when switching. Completed cross-model handoffs unload native weights too, preventing the source and target model graphs from staying resident together. Completed latent, hires conditioning, and duplicate pixel references are released. Same-model refinement retains its model and prompt cache. Selection changes, per-family module settings, preset keys, and loader parameters are restored after success or failure. Peak memory and speed still depend on the backend, model, resolution, and hardware.

## Quantization and model combinations

The refiner passes checkpoint identities to the existing loader without a format whitelist. INT8 ConvRot, NVFP4, ConvRot W4A4, FP8, MXFP8, GGUF and future formats therefore use the installed backend's support. A different target uses the resolved target preset storage precision, or automatic precision when no different target preset is resolved. This avoids inheriting a forced source conversion; the original settings are restored afterwards. No checkpoints are converted or downloaded.

Any base can hand off decoded pixels, but the target must actually accept images and have compatible VAE/text-encoder components, adapters and kernels. Boogu Base/Turbo are text-only targets: use Edit/Edit-Turbo instead. Unknown format or architecture support must be supplied by Forge or the model extension, not invented by this refiner. If selection changes the preset, saved target img2img sampler, scheduler, steps and guidance defaults are used where available; an explicit Refiner CFG still wins.

Forge Neo's upstream supported-format list is maintained in its [README](https://github.com/Haoming02/sd-webui-forge-classic/blob/neo/README.md). NVIDIA describes [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) as a distinct floating-point format; it is not interchangeable with integer INT4. A filename alone is not evidence that a checkpoint's encoding, model architecture, and kernels are compatible.

## Speed and strength
Use **Switch at** as the preserved base-image fraction: 0.75 means 0.25 denoising; 1 means no refinement. This extension interprets it as a fraction even when native Forge is configured for sigma switching. Refinement uses a speed-first budget: ceil(base or target preset steps times denoising strength), normally 2 to 8 steps, or 1 if the source requests 1. Native img2img uses fixed refinement steps so denoising does not reduce this budget a second time. CFG 0 automatically uses CFG 1 to reduce guidance work; an explicit CFG of 1 or higher is respected. Stock live previews, repeated face restoration, and embedding rescans are disabled for the refinement pass. These shortcuts can reduce image quality. Reference-image editing models such as Qwen 2.1 use their native image guidance and may not honor an img2img denoising fraction; a value of 1 always skips the pass. The LoRA rules UI is removed.

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
2. Choose **one** way - never both: **Install from URL** with this repository's URL, or **Download ZIP** and extract into `sd-webui-forge-classic/extensions/`.
3. Avoid double nesting. The correct path ends with: `extensions/<folder>/scripts/engine.py`.
4. Start Forge normally. Checkpoint refinement installs no extra dependencies; optional DLSS requires the separately configured runtime.
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


## FLUX3 target prompt separation (GPU validation pending)

FLUX3 has no negative-prompt guidance. AutoNeg may remain enabled for the base:
under Refiner target settings explicitly enable separate refiner prompts, supply
a complete target caption without foreign base LoRA tags, and leave its negative
prompt empty. The base prompt/AutoNeg output is retained. The second pass does not
rerun processing scripts. Inherited unsupported negatives still fail clearly;
they are never silently dropped. Distilled FLUX3 requires Refiner CFG 0 or 1; experimental Base accepts 0 through 20. Both require an empty target negative prompt.

FLUX3 receives the decoded base image as full-frame reference conditioning using
its packaged matching Qwen3VL/VAE and official flow-map sampler. A non-skipped
pass defaults to 16 distilled steps or 25 experimental Base steps; Switch at is not FLUX3 denoising strength. The
optional target steps control overrides this explicitly. Native mask/inpainting
and variation-seed requests are rejected. Separate prompts are opt-in for other
targets too; existing first-five script arguments retain their positions.

Retained-base failures now populate returned Refiner error metadata and issue an
available Gradio warning. CPU routing tests do not prove these model pairs on a GPU.
These changes require a coordinated restart and genuine FLUX3, MOSS and Ideogram4
Refiner tests; CPU tests and prior standalone generations do not prove those pairs.
