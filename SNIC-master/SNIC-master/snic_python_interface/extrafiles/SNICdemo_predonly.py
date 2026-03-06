#!/usr/bin/env python3
"""
SNICdemo.py

Purpose
-------
Run SNIC superpixel segmentation on each test image, then refine a raw semantic
segmentation prediction using superpixel majority voting (mode within each
superpixel). Saves a color visualization of the refined predictions and prints
aggregate metrics (confusion matrix, per-class P/R/F1/F2, overall accuracy).

Assumptions / repository layout
-------------------------------
This script assumes it lives inside a repository where:
- runs/<exp_name> contains raw predictions named: <image>.png -> <image>_raw.png
- test_images/label_removed_images contains test grayscale .png images
- test_images/label_removed_labels contains grayscale label masks (0=background)
- superpixel_refinement/<exp_name> will be created for outputs

Environment
-----------
SEgSP_MODEL: one of {fcn8s, fcn32s, deeplabv3, psp, denseaspp}
"""

from __future__ import annotations

import os
from pathlib import Path
from timeit import default_timer as timer

import cv2
import numpy as np
from PIL import Image
from cffi import FFI
from sklearn.metrics import confusion_matrix

from _snic.lib import SNIC_main
import doing_majority_voting

# Environment flag: if true, run superpixel refinement + save only (skip evaluation metrics)
PRED_ONLY = os.getenv("PRED_ONLY", "").strip().lower() in {"1", "true", "yes", "y"}


# -----------------------------
# Configuration (defaults)
# -----------------------------
Image.MAX_IMAGE_PIXELS = None  # allow very large images

DEFAULT_NUM_SUPERPIXELS = 3000
DEFAULT_COMPACTNESS = 1.0
DEFAULT_DO_RGB_TO_LAB = False  # SNIC supports Lab if the image is 3-channel

# Class IDs used in your pipeline (background = 0)
LABELS_TO_INCLUDE = [1, 2, 3, 4]


# -----------------------------
# Utility functions
# -----------------------------
def segment_snic(
    img_path: str | Path,
    num_superpixels: int = DEFAULT_NUM_SUPERPIXELS,
    compactness: float = DEFAULT_COMPACTNESS,
    do_rgb_to_lab: bool = DEFAULT_DO_RGB_TO_LAB,
    downscale: int = 4,
) -> tuple[np.ndarray, int]:
    """
    Run SNIC to generate a superpixel label map.

    Notes
    -----
    Your original code:
    - Loads image in grayscale ("L")
    - Downscales by 4× for speed
    - Replicates to 3 channels before SNIC (Image.merge("RGB", ...))

    Returns
    -------
    labels : (H, W) int32 array
        Superpixel id per pixel.
    numlabels : int
        Number of superpixel labels returned by SNIC.
    """
    img = Image.open(str(img_path)).convert("L")

    if downscale and downscale > 1:
        new_size = (img.width // downscale, img.height // downscale)
        img = img.resize(new_size, resample=Image.BILINEAR)

    # Replicate grayscale to 3 channels (matches your original script)
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

    start = timer()
    SNIC_main(pinp, w, h, c, num_superpixels, compactness, do_rgb_to_lab, plabels, pnumlabels)
    _ = timer() - start  # available for optional logging

    return labels, int(numlabels[0])


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
    return f"{model_name}_{backbone}_{dataset}"


def colorize_refined(refined: np.ndarray, raw_pred: np.ndarray) -> np.ndarray:
    """Create an RGB visualization image of the refined prediction."""
    rgb = np.zeros((*refined.shape, 3), dtype=np.uint8)
    pred_mask = raw_pred > 0

    rgb[np.logical_and(pred_mask, refined == 1)] = [0, 255, 0]
    rgb[np.logical_and(pred_mask, refined == 2)] = [0, 0, 255]
    rgb[np.logical_and(pred_mask, refined == 3)] = [0, 255, 255]
    rgb[np.logical_and(pred_mask, refined == 4)] = [255, 165, 0]
    return rgb


def aggregate_metrics(all_gt: list[np.ndarray], all_pred: list[np.ndarray]) -> None:
    """Print confusion matrix + per-class metrics computed from the aggregated CM."""
    gt_concat = np.concatenate(all_gt)
    pred_concat = np.concatenate(all_pred)

    cm = confusion_matrix(gt_concat, pred_concat, labels=LABELS_TO_INCLUDE)

    print("Confusion Matrix (Aggregated):")
    print("Rows: Ground Truth, Columns: Prediction")
    print(cm)

    for i, label in enumerate(LABELS_TO_INCLUDE):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        f2 = 5 * precision * recall / (4 * precision + recall + 1e-6)

        print(
            f"Aggregate cls{label} → "
            f"P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}, F2: {f2:.4f}"
        )

    correct = np.trace(cm)
    total = cm.sum()
    accuracy = correct / (total + 1e-6)
    print(f"\nOverall Accuracy: {accuracy:.4f}")


def main() -> None:
    """Run refinement over all .png files in the test image folder."""
    repo_root = Path(__file__).resolve().parents[3]

    model_name = os.getenv("SEgSP_MODEL", "").strip()
    if not model_name:
        raise EnvironmentError(
            "SEgSP_MODEL is not set. Example:\n"
            "  export SEgSP_MODEL=deeplabv3\n"
            "  python SNICdemo.py"
        )

    exp_name = make_exp_name(model_name, dataset="substrate")

    out_dir = (repo_root / "superpixel_refinement" / exp_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_folder = (repo_root / "runs" / exp_name).resolve()

    image_folder = (repo_root / "test_images" / "label_removed_images").resolve()
    gt_folder = (repo_root / "test_images" / "label_removed_labels").resolve()

    test_filenames = sorted(f for f in os.listdir(image_folder) if f.lower().endswith(".png"))
    if not test_filenames:
        raise FileNotFoundError(f"No .png files found in: {image_folder}")

    all_gt: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    for test_filename in test_filenames:
        img_path = image_folder / test_filename
        gt_path = gt_folder / test_filename

        raw_pred_name = test_filename.replace(".png", "_raw.png")
        raw_pred_path = pred_folder / raw_pred_name

        if not raw_pred_path.exists():
            raise FileNotFoundError(
                f"Missing raw prediction: {raw_pred_path}\n"
                f"Expected naming: {test_filename} -> {raw_pred_name}"
            )

        # Load
        img = Image.open(str(img_path)).convert("L")
        gt = Image.open(str(gt_path)).convert("L")
        rawp = Image.open(str(raw_pred_path)).convert("L")

        img_np = np.array(img, dtype=np.uint8)
        gt_np = np.array(gt, dtype=np.uint8)
        rawp_np = np.array(rawp, dtype=np.uint8)

        # Resize everything to 1/4 scale (matches original behavior)
        h, w = rawp_np.shape
        target_size = (w // 4, h // 4)

        _img_small = np.array(Image.fromarray(img_np).resize(target_size, Image.BILINEAR))
        gt_small = cv2.resize(gt_np, target_size, interpolation=cv2.INTER_NEAREST)
        rawp_small = cv2.resize(rawp_np, target_size, interpolation=cv2.INTER_NEAREST)

        # Superpixels
        sp_map, _ = segment_snic(img_path, num_superpixels=DEFAULT_NUM_SUPERPIXELS)

        # Refine prediction via superpixel mode
        refined = doing_majority_voting.smooth_with_superpixel_mode(rawp_small, sp_map)

        # Save color visualization
        rgb = colorize_refined(refined, rawp_small)
        Image.fromarray(rgb).save(str(out_dir / test_filename))

        # Metrics: include pixels where either GT or refined has a label
        valid = (gt_small > 0) | (refined > 0)
        y_true = gt_small[valid]
        y_pred = refined[valid]
        all_gt.append(y_true)
        all_pred.append(y_pred)

    if not PRED_ONLY:
        aggregate_metrics(all_gt, all_pred)


if __name__ == "__main__":
    main()
