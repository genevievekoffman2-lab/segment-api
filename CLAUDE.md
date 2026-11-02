# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Background-removal API: given an image and a text description of an item (e.g. "brown jacket"), return the item cut out on a transparent background. Being built step by step — currently only the standalone segmentation script exists (`segment.py`); the API layer has not been built yet.

## Environment

- Python 3.11, run everything through `.venv/bin/python`. The venv was created with `--system-site-packages` so it reuses the system torch 2.2.2.
- Intel Mac (x86_64): no MPS, inference runs on CPU. `pick_device()` will use CUDA/MPS if available elsewhere.
- **transformers is pinned `<4.48`** (tested 4.47.1). Newer releases require torch >= 2.6 and silently disable PyTorch on torch 2.2 ("PyTorch was not found"), so the models won't load. Don't upgrade transformers without upgrading torch.
- Setup: `python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt`

## Running

```
.venv/bin/python segment.py tests/samples/jacket.png "brown jacket" -o out/jacket_cutout.png
```

First run downloads models from Hugging Face (SAM-large is ~1.2GB). No test suite yet; `tests/samples/` holds input images and `*_expected.png` references. References are approximate targets, not pixel-exact — compare visually (e.g. composite the RGBA output on a solid color).

## Pipeline (`segment.py`)

1. **Grounding DINO** (`IDEA-Research/grounding-dino-tiny`) turns the text prompt into boxes; the highest-scoring box is used.
2. **SAM** (`facebook/sam-vit-large`) turns that box into a mask; the candidate with the best predicted IoU is used.
3. **`outer_outline`** reduces the mask to the item's outer silhouette so contents inside it are kept (e.g. the shirt visible through an open jacket): morphological closing with a disk of radius `CLOSING_RATIO * max(box side)`, then hole fill, then keep the largest connected region.
4. Mask becomes the alpha channel; output is cropped to the mask bbox.

Design decisions behind the current choices (from testing on the jacket sample):
- `sam-vit-base` dropped the gingham collar and cuffs; `vit-large` keeps the collar and one cuff.
- SAM splits an open jacket into two halves with an open-ended gap between them, so hole-filling alone can't recover the interior — the closing step is what bridges it. `CLOSING_RATIO = 0.045` was tuned on that sample; too large starts swallowing background in concavities.

Known limitation: on `jacket.png` the bottom-left gingham cuff is still missing — no SAM candidate for the full-jacket box includes it. Likely needs extra SAM prompts (points or per-part boxes).

## TODO (roadmap to a public API usable from another app)

1. **Build the API** — FastAPI app wrapping `segment.py`: `POST /segment` taking an image upload + prompt, returning the RGBA PNG. Load both models once at startup (not per request). Return a clear error when nothing matches the prompt. Test locally with `curl`.
2. **Containerize** — Dockerfile with the pinned requirements; download model weights at build time so cold starts don't fetch ~1.5GB.
3. **Choose hosting** — open decision; depends on where the consuming app runs and expected traffic. Measure CPU latency per request first (not yet measured).
   - GPU serverless (Modal / Replicate / RunPod): leading option — per-second billing, scales to zero, cold starts.
   - Hugging Face Spaces: easiest public URL, free CPU tier, slow and sleeps — demos only.
   - CPU VPS / Cloud Run: cheap, slow per request.
   - Always-on GPU: fast, pays 24/7.
4. **Harden before going public** — API key auth, upload size limit, rate limiting, request timeouts.
5. **Integrate with the other app** — plain HTTP call. If called from a browser, configure CORS and proxy through the app's backend so the API key isn't exposed client-side.

Optional later: faster segmentation model (e.g. SAM 2 / smaller variant); fix the missing-cuff limitation above.
