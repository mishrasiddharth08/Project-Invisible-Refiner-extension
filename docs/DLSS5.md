# DLSS 5 Neural Rendering

## Existing images

Open Forge's **Extras** tab, upload an image, enable **DLSS 5 Neural Rendering**,
and press **Generate**. Start with the default intensity and compare the result.
Native batch-image input is also supported. Leave other Extras effects disabled
when you want only DLSS processing. This does not generate a new image using a
diffusion checkpoint; it enhances the uploaded pixels. Original input files are
not overwritten. Improvement depends on the image and can include changed details.

## Newly generated images

Select **DLSS 5 Neural Rendering** in the existing Refiner checkpoint dropdown,
then press Generate normally. No second tab or Python environment is needed.
`Switch at` controls preserved strength: 0.75 applies DLSS intensity 0.25;
1 skips enhancement. CFG, prompts and diffusion sampling steps do not control
this backend. The adapter checks size and restores alpha and in-memory metadata. Forge controls the final export and filenames; JPEG EXIF/DPI preservation is not guaranteed.
Errors and cancellation keep the base image.

The backend calls the separately installed Visual Enhancer v18 command line
tool with only Neural Rendering enabled, native resolution, style 0, one pass.
This is the community integration supplied by Merserk, not an NVIDIA-supported
Forge plugin. It can change faces, text, color and detail.

Download Visual Enhancer from its author's release page:
https://github.com/Merserk/dlss5-visual-enhancer/releases/tag/v18.0
Keep the complete extracted distribution together. Set
`PI_REFINER_DLSS5_HOME` to its directory, or create `dlss5-runtime.json` in this
extension containing a `home` key with your own runtime directory. The adapter does
not download executables automatically. Vendor binaries are external dependencies
and must not be copied into a redistributed extension release.

The shared `lib/dlss5.py` file adapter also accepts video input and an output
filename through `enhance_file()`. A whole clip is sent in one invocation so the
upstream engine can own temporal state and scene-cut resets. The Forge video UI
is future work. Independent still-image calls must not be advertised as a
temporally consistent video workflow.

Full-frame images are supported. Masked/inpainting refinement is rejected to
preserve protected regions. Completed base models are offloaded using Forge's
memory manager before the native process runs.

## Requirements, provenance and validation boundaries

The tested configuration used Windows, an RTX 5090, driver 617.42 and Visual
Enhancer 18.0.0. That is a tested configuration, not a minimum hardware claim.
Follow the upstream release's GPU, driver and runtime requirements and licenses.
Keep the entire upstream distribution together. The binaries are unsigned and
are not included, downloaded or installed by this extension. Redistribution
permission for the runtime has not been established; obtain it from the author.
This extension does not confer a license to any third-party component.

The original October 22 task recorded 81 automated tests passing, live Refiner
and Extras image checks, five distinct existing-image results with originals
unchanged, and a separate eight-frame video backend check retaining audio.
Those are prior task results; this publication does not rerun the unsigned
runtime. Forge video UI integration is unfinished, and native framewise video
Extras processing is intentionally skipped. No temporal-quality guarantee is made.

JPEG tests exposed EXIF/DPI changes in Forge's export path. A general fix has not
been verified. Verify exported metadata yourself when those fields matter. Other
model pairings and GPUs, production quality, speed and memory benefits remain
unverified. Enhancement may alter faces, text, color and fine details.
