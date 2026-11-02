"""Cut a described item out of an image using Grounding DINO + SAM.

Grounding DINO turns the text prompt into a bounding box, SAM turns that box
into a pixel mask. The mask is then reduced to the item's outer outline so
whatever sits inside the item -- e.g. the shirt visible through an open jacket
-- is kept: a morphological closing bridges narrow gaps (like an open front),
holes are filled, and stray blobs are dropped. Output is an RGBA PNG cropped to the
item, with a transparent background.

Usage:
    python segment.py tests/samples/jacket.png "brown jacket" -o out/jacket.png
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)

DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
# vit-base misses the gingham collar/cuffs on the jacket sample; vit-large keeps them.
SAM_MODEL = "facebook/sam-vit-large"
# Closing radius as a fraction of the detected box's longer side. Big enough to
# bridge an open jacket front, small enough not to swallow background.
CLOSING_RATIO = 0.045


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def detect_box(image: Image.Image, prompt: str, device: str, box_threshold: float, text_threshold: float):
    """Return the highest-scoring [x0, y0, x1, y1] box for the prompt, or None."""
    processor = AutoProcessor.from_pretrained(DINO_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL).to(device)

    # Grounding DINO expects lowercase text ending with a period.
    text = prompt.lower().strip().rstrip(".") + "."
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]
    if len(result["scores"]) == 0:
        return None
    best = result["scores"].argmax()
    return result["boxes"][best].tolist(), result["scores"][best].item()


def segment_box(image: Image.Image, box: list[float], device: str) -> np.ndarray:
    """Return SAM's best boolean mask for the box."""
    processor = SamProcessor.from_pretrained(SAM_MODEL)
    model = SamModel.from_pretrained(SAM_MODEL).to(device)

    inputs = processor(image, input_boxes=[[box]], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0][0]  # (num_candidates, H, W) for our single box
    best = outputs.iou_scores[0, 0].argmax().item()
    return masks[best].numpy().astype(bool)


def outer_outline(mask: np.ndarray, radius: int) -> np.ndarray:
    """Bridge gaps narrower than ~2*radius, fill enclosed holes, keep the largest region."""
    if radius > 0:
        y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        disk = x * x + y * y <= radius * radius
        # Pad so the closing isn't clipped at the image border.
        padded = np.pad(mask, radius + 1)
        mask = ndimage.binary_closing(padded, disk)[radius + 1 : -(radius + 1), radius + 1 : -(radius + 1)]
    mask = ndimage.binary_fill_holes(mask)

    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    return labels == (np.argmax(sizes) + 1)


def cut_out(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Apply the mask as alpha and crop to its bounding box."""
    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray((mask * 255).astype(np.uint8)))
    ys, xs = np.nonzero(mask)
    return rgba.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("prompt")
    parser.add_argument("-o", "--output", type=Path, default=None, help="default: <image>_cutout.png")
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument(
        "--closing-ratio", type=float, default=CLOSING_RATIO, help="gap-bridging radius as a fraction of box size"
    )
    parser.add_argument("--keep-holes", action="store_true", help="use SAM's raw mask, no outline cleanup")
    args = parser.parse_args()

    output = args.output or args.image.with_name(f"{args.image.stem}_cutout.png")
    device = pick_device()
    image = Image.open(args.image).convert("RGB")

    detection = detect_box(image, args.prompt, device, args.box_threshold, args.text_threshold)
    if detection is None:
        raise SystemExit(f'No "{args.prompt}" found in {args.image}')
    box, score = detection
    print(f"Detected box {[round(v) for v in box]} (score {score:.2f})")

    mask = segment_box(image, box, device)
    if not args.keep_holes:
        radius = round(args.closing_ratio * max(box[2] - box[0], box[3] - box[1]))
        mask = outer_outline(mask, radius)

    output.parent.mkdir(parents=True, exist_ok=True)
    cut_out(image, mask).save(output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
