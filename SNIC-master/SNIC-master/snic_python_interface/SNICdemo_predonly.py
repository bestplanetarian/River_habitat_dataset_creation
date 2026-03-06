#!/usr/bin/env python3
"""
SNICdemo_predonly_full.py

Purpose
-------
Prediction-only superpixel refinement stage for your SegSP inference pipeline.

This script:
1) Reads an input image folder from environment variable:  data_path
2) Reads raw prediction masks from:
      ../run/{model}_{backbone}_{dataset}_predonly
   where each raw prediction is named:
      <base>_raw.png
3) Runs SNIC superpixel segmentation (on a downscaled version of the image),
   then refines the raw prediction via superpixel majority voting (mode within
   each superpixel).
4) Saves refined outputs to:
      superpixel_refinement/{model}_{backbone}_{dataset}_predonly

Outputs
-------
For each input image <base>:
- <base>_refined_raw.png   : refined class-id mask (same size as input image)
- <base>_refined.png       : color visualization (same size as input image)

Notes
-----
- No ground truth required.
- Downscale factor is fixed to 4 (matches your previous SNICdemo_clean behavior).
- Background leakage protection: refined pixels are forced to 0 wherever the raw
  prediction is 0 (after resizing to full size).

Environment
-----------
Required:
- data_path : directory that contains the original test images

Optional:
- SEgSP_MODEL : if you want to use model-only naming; not required here because
               we use CLI args (model/backbone/dataset) like your train/eval scripts.

Example
-------
export data_path=/path/to/test_images
python SNICdemo_predonly_full.py --model deeplabv3 --backbone resnet50 --dataset substrate
"""

from __future__ import annotations

import os
from pathlib import Path
from timeit import default_timer as timer

import cv2
import numpy as np
from PIL import Image
from cffi import FFI

from _snic.lib import SNIC_main
import doing_majority_voting



# -----------------------------
# Constants (match prior script)
# -----------------------------
Image.MAX_IMAGE_PIXELS = None  # allow very large images

DOWNSCALE = 4
NUM_SUPERPIXELS = 3000
COMPACTNESS = 1.0
DO_RGB_TO_LAB = False  # grayscale replicated to 3 channels


# -----------------------------
# Helpers
# -----------------------------
def find_image_path(image_dir: Path, base: str) -> Path | None:
    """Find an image file in image_dir matching base name with common extensions."""
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]
    for ext in exts:
        p = image_dir / f"{base}{ext}"
        if p.exists():
            return p
    return None


def segment_snic(
    img_path: str | Path,
    num_superpixels: int = NUM_SUPERPIXELS,
    compactness: float = COMPACTNESS,
    do_rgb_to_lab: bool = DO_RGB_TO_LAB,
    downscale: int = DOWNSCALE,
) -> np.ndarray:
    """
    Run SNIC to generate a superpixel label map on a downscaled image.

    Returns
    -------
    labels : (h, w) int32 array
        Superpixel id per pixel at the downscaled resolution.
    """
    img = Image.open(str(img_path)).convert("L")

    if downscale and downscale > 1:
        new_size = (max(1, img.width // downscale), max(1, img.height // downscale))
        img = img.resize(new_size, resample=Image.BILINEAR)

    # Replicate grayscale to 3 channels (matches your original SNIC wrapper)
    img_rgb = Image.merge("RGB", (img, img, img))
    img_arr = np.asarray(img_rgb)

    h, w, c = img_arr.shape
    chw = img_arr.transpose(2, 0, 1).reshape(-1).astype(np.double)

    labels = np.zeros((h, w), dtype=np.int32)
    numlabels = np.zeros(1, dtype=np.int32)

    ffibuilder = FFI()
    pinp = ffibuilder.cast("double*", ffibuilder.from_buffer(chw))
    plabels = ffibuilder.cast("int*", ffibuilder.from_buffer(labels.reshape(-1)))
    pnumlabels = ffibuilder.cast("int*", ffibuilder.from_buffer(numlabels))

    _t0 = timer()
    SNIC_main(pinp, w, h, c, num_superpixels, compactness, do_rgb_to_lab, plabels, pnumlabels)
    _ = timer() - _t0  # available for optional logging

    return labels


def colorize_refined(mask: np.ndarray) -> np.ndarray:
    """Colorize refined class-id mask into RGB image (uint8)."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    # Adjust these colors if you have a standard palette elsewhere
    rgb[mask == 1] = [0, 255, 0]       # class 1
    rgb[mask == 2] = [0, 0, 255]       # class 2
    rgb[mask == 3] = [0, 255, 255]     # class 3
    rgb[mask == 4] = [255, 165, 0]     # class 4
    return rgb

def make_exp_name(model_name: str, dataset: str = "substrate") -> str:
    """Match exp naming convention: <model>_<backbone>_<dataset>."""
    model_name = model_name.lower().strip()

    backbone_map = {
        "fcn8s": "vgg16",
        "fcn32s": "vgg16",
        "deeplabv3": "resnet50",
        "psp": "resnet50",
        "denseaspp": "densenet121",
    }
    if model_name not in backbone_map:
        raise ValueError(
            f"Unknown SEgSP_MODEL='{model_name}'. Expected one of: {sorted(backbone_map.keys())}"
        )
    backbone = backbone_map[model_name]
    return f"{model_name}_{backbone}_{dataset}_predonly"



def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    
    # env: image folder
    data_path = os.getenv("data_path", "").strip()
    model_name = os.getenv("SEgSP_MODEL", "").strip()
    if not data_path:
        raise EnvironmentError(
            "Environment variable 'data_path' is not set.\n"
            "Example:\n"
            "  export data_path=/path/to/test_images"
        )
    image_dir = Path(data_path).resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"data_path does not exist: {image_dir}")

    # prediction folder (produced by your folder-pred script)
    exp_name = make_exp_name(model_name, dataset="substrate")
    
    #pred_dir = Path(f"../runs/{args.model}_{args.backbone}_{args.dataset}_predonly").resolve()
    pred_dir = (repo_root / "runs" / exp_name).resolve()
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction folder not found: {pred_dir}")

    # output folder
    out_dir = (repo_root / "superpixel_refinement" / exp_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    

    print("Image folder:", image_dir)
    print("Prediction folder:", pred_dir)
    print("Output folder:", out_dir)

    pred_files = sorted([p for p in pred_dir.iterdir() if p.is_file() and p.name.endswith("_raw.png")])
    if not pred_files:
        raise RuntimeError(f"No *_raw.png found in: {pred_dir}")

    for pred_path in pred_files:
        base = pred_path.name.replace("_raw.png", "")
        img_path = find_image_path(image_dir, base)
        if img_path is None:
            print(f"[SKIP] image not found for base='{base}' in {image_dir}")
            continue

        # Load full-res inputs
        img_full = Image.open(str(img_path)).convert("L")
        raw_full = Image.open(str(pred_path)).convert("L")

        img_full_np = np.array(img_full, dtype=np.uint8)
        raw_full_np = np.array(raw_full, dtype=np.uint8)

        H, W = img_full_np.shape

        # Ensure raw pred matches image size (if not, resize to image size)
        if raw_full_np.shape != (H, W):
            raw_full_np = cv2.resize(raw_full_np, (W, H), interpolation=cv2.INTER_NEAREST)

        # Downscale both image and prediction (to match SNIC superpixel map)
        small_size = (max(1, W // DOWNSCALE), max(1, H // DOWNSCALE))
        raw_small = cv2.resize(raw_full_np, small_size, interpolation=cv2.INTER_NEAREST)

        # Superpixels on downscaled image
        sp_map = segment_snic(img_path, num_superpixels=NUM_SUPERPIXELS, compactness=COMPACTNESS)

        # Refine at downscaled resolution
        refined_small = doing_majority_voting.smooth_with_superpixel_mode(raw_small, sp_map).astype(np.uint8)

        # Upscale back to full resolution
        refined_full = cv2.resize(refined_small, (W, H), interpolation=cv2.INTER_NEAREST)

        # Prevent background leakage: keep background where raw is background
        refined_full[raw_full_np == 0] = 0

        # Save outputs
        #out_raw = out_dir / f"{base}_refined_raw.png"
        out_rgb = out_dir / f"{base}_refined.png"

        #Image.fromarray(refined_full).save(str(out_raw))
        Image.fromarray(colorize_refined(refined_full)).save(str(out_rgb))

        print(f"[OK] {base} -> {out_rgb.name}")

    print("Done.")


if __name__ == "__main__":
    main()
