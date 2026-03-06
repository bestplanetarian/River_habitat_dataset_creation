# Probabilistic fusion of model predictions with texture features (entropy/variance) at the superpixel level.
# - Train phase: estimate per-class Gaussian stats over superpixel features from ground-truth
# - Inference: combine model softmax (averaged within superpixels) with texture likelihoods
#
# This is self-contained and uses only numpy + imageio (for I/O if you plug in real files).
# It computes per-superpixel features WITHOUT full GLCM:
#   - intensity mean
#   - intensity variance
#   - histogram entropy (levels=32) computed over pixels inside each superpixel
#
# You can optionally add GLCM-based features for ambiguous superpixels only.
#
# Replace the synthetic demo at the bottom with your real data:
#   - image: uint8 (H,W)
#   - spx:   int32 superpixel IDs (H,W)
#   - mask:  int32 ground-truth labels (H,W)
#   - model_probs: float (H,W,C) softmax per pixel (from your network)
#
# Then run: stats = fit_class_texture_stats(image, spx, mask, num_classes=C)
#           fused = fuse_with_texture(image, spx, model_probs, stats, alpha=0.7)
#
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import math

def _quantize_u8(image_u8: np.ndarray, levels: int) -> np.ndarray:
    #if image_u8.dtype != np.uint8:
    #    raise ValueError("image_u8 must be uint8")
    bin_size = max(1, 256 // levels)
    q = (image_u8 // bin_size)
    q[q >= levels] = levels - 1
    return q

def _masked_glcm(image_q: np.ndarray,
                 region_mask: np.ndarray,
                 distances=(1,),
                 angles=(0, np.pi/4, np.pi/2, 3*np.pi/4),
                 levels=32,
                 symmetric=True,
                 norm=True) -> np.ndarray:
    H, W = image_q.shape
    P = np.zeros((levels, levels), dtype=np.float64)
    region = region_mask.astype(bool)
    if not np.any(region):
        return P

    for d in distances:
        for theta in angles:
            dx = int(round(math.cos(theta) * d))
            dy = int(round(-math.sin(theta) * d))  # image coords (y down)
            if dx == 0 and dy == 0:
                continue

            y0_min = max(0, -dy); y0_max = min(H, H - dy)
            x0_min = max(0, -dx); x0_max = min(W, W - dx)
            if y0_min >= y0_max or x0_min >= x0_max:
                continue

            rr  = region[y0_min:y0_max, x0_min:x0_max]
            rr2 = region[y0_min+dy:y0_max+dy, x0_min+dx:x0_max+dx]
            valid = rr & rr2
            if not np.any(valid):
                continue

            I = image_q[y0_min:y0_max, x0_min:x0_max][valid]
            J = image_q[y0_min+dy:y0_max+dy, x0_min+dx:x0_max+dx][valid]
            idx = (I * levels + J).astype(np.int64)
            counts = np.bincount(idx, minlength=levels*levels)
            P += counts.reshape(levels, levels)

    if symmetric:
        P = P + P.T
    if norm:
        s = P.sum()
        if s > 0:
            P = P / s
    return P

def _glcm_props(P: np.ndarray):
    """Return (contrast, entropy, homogeneity) from normalized GLCM."""
    eps = 1e-12
    if P.sum() <= 0:
        return (np.nan, np.nan, np.nan)
    L = P.shape[0]
    i = np.arange(L).reshape(-1,1)
    j = np.arange(L).reshape(1,-1)
    contrast    = float(np.sum(P * (i - j)**2))
    homogeneity = float(np.sum(P / (1.0 + np.abs(i - j))))
    nz = P[P > 0]
    entropy     = float(-np.sum(nz * np.log2(nz + eps)))
    return (contrast, entropy, homogeneity)

def superpixel_stats_glcm(image_u8: np.ndarray,
                          spx: np.ndarray,
                          *,
                          levels=32,
                          distances=(1,),
                          angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    """Per-superpixel GLCM features: [contrast, entropy, homogeneity]."""
    #if image_u8.dtype != np.uint8:
    #    raise ValueError("image_u8 must be uint8")
    print("The length of superpixel list is:")

    print(spx.shape)
    if spx.shape != image_u8.shape:
        raise ValueError("spx must match image size")
    img_q = _quantize_u8(image_u8, levels=levels)
    feats = {}
    for sp_id in np.unique(spx):
        region = (spx == sp_id)
        if region.sum() < 20:  # skip tiny SPs
            feats[sp_id] = np.array([np.nan, np.nan, np.nan], dtype=float)
            continue
        P = _masked_glcm(img_q, region, distances=distances, angles=angles,
                         levels=levels, symmetric=True, norm=True)
        feats[sp_id] = np.array(_glcm_props(P), dtype=float)
    return feats


# ---------- NEW: batch utilities for split tiles ----------

'''

@dataclass
class GaussianStats:
    mean: np.ndarray  # (3,) for [contrast, entropy, homogeneity]
    var:  np.ndarray  # (3,)

def fit_class_texture_stats_batch_glcm_ignore_bg(
    images_u8_list,    # list[(H,W) uint8]
    spx_list,          # list[(H,W) int]
    masks_list,        # list[(H,W) int]  (GT)
    num_classes: int,
    *,
    background_id: int = 0,
    levels: int = 32,
    distances=(1,),
    angles=(0, np.pi/4, np.pi/2, 3*np.pi/4),
    classes_subset=None,      # e.g. [3,6,7]; still ignores background_id
    superpixel_stats_glcm_fn=None  # optional injection; defaults to your existing function
):
    """
    Learn diag-Gaussian per FOREGROUND class only.
    Superpixels whose GT majority == background_id are ignored.
    """
    if superpixel_stats_glcm_fn is None:
        superpixel_stats_glcm_fn = superpixel_stats_glcm  # your existing function

    per_class = defaultdict(list)
    for img, spx, msk in zip(images_u8_list, spx_list, masks_list):
        feats = superpixel_stats_glcm_fn(img, spx, levels=levels, distances=distances, angles=angles)
        for sp_id, f in feats.items():
            if np.any(~np.isfinite(f)):
                continue
            m = (spx == sp_id)
            labels, counts = np.unique(msk[m], return_counts=True)
            valid = (labels >= 0) & (labels < num_classes)
            if not np.any(valid):
                continue
            labels = labels[valid]; counts = counts[valid]
            c = int(labels[np.argmax(counts)])
            if c == background_id:
                continue  # <-- ignore background
            if classes_subset is not None and c not in classes_subset:
                continue
            per_class[c].append(f)

    class_stats = {}
    D = 3
    for c in range(num_classes):
        if c == background_id:
            continue  # no stats for background
        if len(per_class[c]) == 0:
            # fallback neutral stats for rare classes
            class_stats[c] = GaussianStats(mean=np.zeros(D), var=np.ones(D))
        else:
            X = np.vstack(per_class[c])
            class_stats[c] = GaussianStats(mean=X.mean(axis=0), var=X.var(axis=0) + 1e-6)

    return class_stats 

'''





@dataclass
class GaussianStats:
    mean: np.ndarray  # (D,)
    var:  np.ndarray  # (D,)

def fit_class_texture_stats_batch_glcm_soft(
    images_u8_list,     # list[(H,W) uint8]
    spx_list,           # list[(H,W) int]
    masks_list,         # list[(H,W) int]  (GT)
    num_classes: int,
    *,
    background_id: int = 0,
    levels: int = 96,
    distances=(1,),
    angles=(0, np.pi/4, np.pi/2, 3*np.pi/4),
    min_sp_area: int = 20,
    min_total_weight: float = 1e-6,
    temperature: float = 1.0,       # <1 sharpen, >1 smooth class proportions
    classes_subset=None,            # e.g., [3,6,7]; still ignores background
    superpixel_stats_glcm_fn=None   # optional; defaults to local function
):
    """
    Soft-label fitting: each superpixel contributes to class c with weight equal
    to its GT proportion for class c (optionally temperature-adjusted). Background ignored.
    Returns dict: class_id -> GaussianStats(mean, var) over [contrast, entropy, homogeneity].
    """
    # ---- fallback to local function if none provided ----
    if superpixel_stats_glcm_fn is None:
        try:
            superpixel_stats_glcm_fn = superpixel_stats_glcm  # noqa: F821 (must exist in module)
        except NameError:
            raise RuntimeError(
                "superpixel_stats_glcm_fn is None and no local superpixel_stats_glcm is defined."
            )

    D = 3  # [contrast, entropy, homogeneity]
    sum_w   = np.zeros(num_classes, dtype=np.float64)
    sum_wx  = np.zeros((num_classes, D), dtype=np.float64)
    sum_wxx = np.zeros((num_classes, D), dtype=np.float64)

    for img, spx, msk in zip(images_u8_list, spx_list, masks_list):
        feats = superpixel_stats_glcm_fn(img, spx, levels=levels, distances=distances, angles=angles)
        for sp_id, x in feats.items():
            # skip tiny/invalid SPs
            m = (spx == sp_id)
            if m.sum() < min_sp_area or x is None or np.any(~np.isfinite(x)):
                continue

            # class proportions in this SP from GT
            labels, counts = np.unique(msk[m], return_counts=True)
            total = float(m.sum())
            if total <= 0:
                continue
            props = {int(l): float(c)/total for l, c in zip(labels, counts)}

            # optional: restrict to subset (but still ignore background)
            if classes_subset is not None:
                props = {c: p for c, p in props.items() if c in classes_subset or c == background_id}

            # temperature scaling
            ps = np.zeros(num_classes, dtype=np.float64)
            invT = 1.0 / max(1e-6, temperature)
            for c, p in props.items():
                if 0 <= c < num_classes:
                    ps[c] = p ** invT

            # ignore background
            ps[background_id] = 0.0
            s = ps.sum()
            if s <= 0:
                continue
            ps /= s

            # accumulate weighted moments
            sum_w   += ps
            sum_wx  += ps[:, None] * x[None, :]
            sum_wxx += ps[:, None] * (x[None, :] ** 2)

    # finalize Gaussian parameters (diagonal)
    class_stats = {}
    for c in range(num_classes):
        if c == background_id:
            continue  # no stats for background
        w = sum_w[c]
        if w <= min_total_weight:
            class_stats[c] = GaussianStats(mean=np.zeros(D), var=np.ones(D))
        else:
            mu  = sum_wx[c] / w
            ex2 = sum_wxx[c] / w
            var = np.maximum(ex2 - mu**2, 1e-6)
            class_stats[c] = GaussianStats(mean=mu, var=var)

    return class_stats


def _gauss_loglike_diag(x: np.ndarray, stats: GaussianStats):
    mu, var = stats.mean, stats.var
    return -0.5 * (np.sum(np.log(2*np.pi*var)) + np.sum((x - mu)**2 / var))


def fuse_texture_over_hard_labels_glcm_noup_ignore_bg(
    image_u8: np.ndarray,     # (H,W) uint8
    spx: np.ndarray,          # (H,W) superpixel IDs
    hard_labels: np.ndarray,  # (H,W) int
    class_stats: dict,        # foreground-only stats from fit phase
    *,
    background_id: int = 0,
    levels=32,
    distances=(1,),
    angles=(0, np.pi/4, np.pi/2, 3*np.pi/4),
    restrict_classes=None,    # e.g., [3,6,7]; background ignored anyway
    margin_thresh_fg=1.0,     # min margin to flip into a FG class
    min_sp_area=80
):
    """
    Only flips superpixels to FOREGROUND classes using texture.
    Never flips into background based on texture (model remains source of background).
    """

    
    print("test dataset shape:")

    


    #if image_u8.dtype != np.uint8:
    #    raise ValueError("image_u8 must be uint8")
    if not (image_u8.shape == spx.shape == hard_labels.shape):
        raise ValueError("image, spx, hard_labels must have identical shapes")
    


    feats = superpixel_stats_glcm(image_u8, spx, levels=levels, distances=distances, angles=angles)
    out = hard_labels.copy()

    # Define the set of foreground classes that have stats
    fg_classes = sorted([c for c in class_stats.keys() if c != background_id])
    if restrict_classes is not None:
        fg_classes = [c for c in fg_classes if c in restrict_classes]
    if len(fg_classes) == 0:
        return out  # nothing to do

    for sp_id in np.unique(spx):
        m = (spx == sp_id)
        if m.sum() < min_sp_area:
            continue
        x = feats[sp_id]
        if np.any(~np.isfinite(x)):
            continue

        # Texture log-likelihoods ONLY over foreground classes
        lls = []
        for c in fg_classes:
            stats = class_stats.get(c, GaussianStats(np.zeros(3), np.ones(3)))
            mu, var = stats.mean, stats.var
            ll = -0.5 * (np.sum(np.log(2*np.pi*var)) + np.sum((x - mu)**2 / var))
            lls.append(ll)
        lls = np.array(lls)

        # Winner FG class + margin among FG only
        if lls.size >= 2:
            top2 = np.partition(lls, -2)[-2:]
            margin = float(top2.max() - top2.min())
        else:
            margin = float('inf')  # single class available
        fg_winner = fg_classes[int(np.argmax(lls))]

        # Current (model) majority label in this SP
        current = int(np.bincount(out[m].ravel()).argmax())

        # Never flip into background based on texture
        if current == background_id:
            # only promote to FG if texture is confident
            if margin >= margin_thresh_fg:
                out[m] = fg_winner
        else:
            # Current is FG:
            # Optionally flip to *another* FG if texture is confident and different
            if fg_winner != current and margin >= margin_thresh_fg:
                out[m] = fg_winner

    return out


# ----------------------------
# Synthetic demo (replace with your data)
# ----------------------------

    
