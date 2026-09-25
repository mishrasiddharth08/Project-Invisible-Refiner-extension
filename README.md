# Project Invisible — Refiner for Forge Neo

**A fast second detail pass for ANY checkpoint Forge can load — SD 1.5, SDXL, Flux, Qwen-Image and more. Any quantization. Any VRAM size. One dropdown, no knobs.**

[Installation](#beginner-installation) · [Usage](#step-by-step-usage) · [Combinations](#what-it-works-with) · [How it works](#how-it-works)

> This is an independent community extension, not an official product of any model vendor. Model licenses remain with their owners.

## The idea

Most checkpoints benefit from a short polish pass after the main generation: fine detail sharpens, small artifacts smooth out. This extension adds that second pass **without** a second refiner checkpoint, **without** extra downloads and **without** extra VRAM — it re-runs the model that is already loaded over its own finished image, through Forge's normal img2img path.

"Invisible" means familiar: no new tabs, no new workflow. Pick a mode, press the normal Generate button.

## Why it is memory-efficient and fast

- **Zero extra model memory** — no second checkpoint is ever loaded. The pass reuses the weights already on your GPU, whatever their quantization (bf16, fp8, int8, and everything else Forge loads).
- **Any VRAM size** — because no new weights appear, the pass uses the same VRAM envelope as your base generation. Forge's own memory management handles VRAM/RAM offloading and loading during the pass exactly as it does for any normal generation, so low-VRAM cards behave the same as always.
- **Turbo speed** — Turbo mode runs a quarter-strength, quarter-step pass: usually just a moment per image. Balanced halves the steps; Quality uses your full step count at half denoise for the strongest polish.
- **Fast VRAM↔RAM offloading** — the second pass flows through Forge's standard pipeline, so its streaming/offload/caching behavior (and anything your memory profile already does) applies to it automatically. Nothing new to configure.

## Features

- **Any model:** every checkpoint Forge loads, any quantization, any architecture.
- **Any checkpoint permutation:** the refiner can use the same checkpoint (zero extra memory) or any other installed checkpoint — Forge's loader swaps it in for the pass and restores your base model afterwards, with fast VRAM↔RAM offloading handled by Forge itself.
- **Three modes:** Turbo (fast), Balanced, Quality (best).
- **Safe by design:** if the refiner pass fails for any reason, your base image is kept and the reason is logged.
- **Honest metadata:** refined images record `Refiner: <mode>` in their PNG info.
- **No core edits:** the extension only observes Forge's normal pipeline and calls it again; nothing in Forge is modified. Set the mode back to Off and behavior is instantly stock.

## Beginner installation

1. Stop Forge completely.
2. Choose **one** way — never both: **Install from URL** with this repository's URL, or **Download ZIP** and extract into `sd-webui-forge-classic/extensions/`.
3. Avoid double nesting. The correct path ends with:
   `extensions/<folder>/scripts/engine.py`.
4. Start Forge normally. No extra dependencies are installed.
5. Refresh your browser with **Ctrl+F5** after updates.

## Step-by-step usage

![Step-by-step usage](docs/img/usage-steps.svg)

1. **Generate as always** — pick your model, sampler, CFG, LoRAs; press Generate.
2. **Open the Refiner panel** — it sits in the normal txt2img/img2img page.
3. **Pick a mode** — Off (default), Turbo (fast), Balanced, or Quality (best).
4. **(Optional) Pick a refiner checkpoint** — *Same as base* (default, zero extra memory) or any other installed checkpoint. Every permutation is possible: refine an SDXL image with another SDXL, a Flux image with Flux, mix freely.
5. **Press Generate again** — the finished image is automatically re-run through a short detail pass and replaces the result in the gallery. If you chose a different checkpoint, Forge swaps it in for the pass and restores your base model afterwards.

## What it works with

Everything, because it reuses your loaded checkpoint through Forge's own pipeline:

| Combination | Behavior |
|---|---|
| SD 1.5 / SDXL / SDXL refiner-less setups | Works |
| Flux, Qwen-Image, and other modern checkpoints | Works |
| Any quantization (bf16, fp8, int8, ...) | Works — refines with the loaded weights |
| Any sampler / scheduler / CFG | Works — the pass inherits your choices (Turbo forces CFG 1) |
| Any LoRA / embedding / preset | Works — they stay applied during the pass |
| txt2img and img2img | Works — refines whatever the base run produced |
| Any VRAM size / Save GPU memory / offload profiles | Works — no additional model memory, Forge's offloading applies |
| **Any checkpoint permutation** | Works — refine with the *same* checkpoint (zero extra memory) or with **any other installed checkpoint**; Forge swaps it in for the pass and restores your base model afterwards |

## How it works

After a normal generation finishes, the extension takes each finished sample and runs it back through Forge's standard img2img machinery with:

- a small **denoising strength** (0.25 for Turbo, 0.35 Balanced, 0.5 Quality),
- a reduced step count derived from your own steps (Turbo uses a quarter, Quality uses the full amount),
- the **same loaded checkpoint, sampler and LoRAs** as the base run.

Because the second pass is an ordinary Forge img2img job, anything the base run could do, the refiner can do too — that is why the combinations table above is simply "works".

Turbo forces CFG 1 to keep the pass as fast as possible; the other modes keep your CFG scale.

## Known limitations

- The pass re-renders pixels at low denoise; extremely fine text can shift slightly (use Turbo for the smallest change).
- Batches are refined one image at a time, so a Quality pass on a large batch takes proportionally longer.
- Forge updates may change extension hooks; keep a backup of the extension folder.

## If something goes wrong

1. Set the mode back to **Off** — generation returns to stock behavior immediately.
2. Check the terminal for `[PI-Refiner]` lines; failures keep the base image and log the reason.
3. Open a GitHub issue with the complete terminal error.

## License

Extension code: [Apache License 2.0](LICENSE). This is an independent Forge adapter; it is not affiliated with any model vendor.

## Special thanks

Special thanks to [u/malcolmrey and the r/malcolmrey community](https://www.reddit.com/r/malcolmrey/), [r/sdforall](https://www.reddit.com/r/sdforall/) and the [r/SECourses community](https://www.reddit.com/r/SECourses/) for support and inspiration.

Thank you to [Forge Neo (sd-webui-forge-classic, neo branch) by Haoming02](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo), and to the Diffusers, Qwen, DeGrid, Spectrum and wider open-source communities.

If you contribute, test, or report issues and would like to be named here, say so and you will be added.

Part of the [Project Invisible](https://github.com/mishrasiddharth08) family —
see also [Qwen-Image-2.1](https://github.com/mishrasiddharth08/Project-Invisible-Qwen2.1-extension)
and [Ideogram 4](https://github.com/mishrasiddharth08/Project-Invisible-Ideogram4-extension).
