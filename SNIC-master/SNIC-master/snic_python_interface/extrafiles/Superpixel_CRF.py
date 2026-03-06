"""
Superpixel + GLCM + Graph Cut refinement (multi-class Potts)

Inputs:
  - image_gray: (H,W) uint8 or float grayscale image (0..255 recommended)
  - spx:        (H,W) int superpixel id map with ids in [0..K-1]
  - EITHER probs (H,W,C) float in [0,1], sum=1 per pixel
    OR     hard_pred (H,W) int labels in [0..C-1]

Key options:
  - ignore_index: background class id (excluded from competition if allow_background=False)
  - lam: smoothness strength (10~40 is common)
  - levels/distances/angles: GLCM parameters
  - prefer_pygco: try pygco first, fallback to PyMaxflow α-expansion

Outputs:
  - refined pixel labels (H,W) ints
"""

import numpy as np
from skimage.feature import graycomatrix, graycoprops

# -------------------- Superpixel adjacency & boundary length --------------------

def region_adjacency_and_boundary(spx):
    """Return edges (i,j) and boundary lengths L_ij using 4-neighborhood."""
    H, W = spx.shape
    edges = {}

    # horizontal adjacencies
    diff = spx[:, 1:] != spx[:, :-1]
    ys, xs = np.where(diff)
    for y, x in zip(ys, xs):
        a, b = int(spx[y, x]), int(spx[y, x+1])
        i, j = (a, b) if a < b else (b, a)
        edges[(i, j)] = edges.get((i, j), 0) + 1

    # vertical adjacencies
    diff = spx[1:, :] != spx[:-1, :]
    ys, xs = np.where(diff)
    for y, x in zip(ys, xs):
        a, b = int(spx[y, x]), int(spx[y-1, x])
        i, j = (a, b) if a < b else (b, a)
        edges[(i, j)] = edges.get((i, j), 0) + 1

    E, L = [], []
    for (i, j), Lij in edges.items():
        E.append((i, j))
        L.append(Lij)
    return E, np.asarray(L, dtype=np.float64)

# -------------------- GLCM features per superpixel --------------------

def quantize_image_u8(img_u8, levels=32):
    """Quantize [0..255] -> [0..levels-1]."""
    if img_u8.dtype != np.uint8:
        img_u8 = np.clip(img_u8, 0, 255).astype(np.uint8)
    q = (img_u8.astype(np.float32) * (levels / 256.0)).astype(np.int32)
    q[q == levels] = levels - 1
    return q.astype(np.uint8)

def glcm_features_for_region(qimg, mask, levels=32, distances=(1,2), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    """
    GLCM features inside a region mask.
    Returns vector [contrast, homogeneity, energy, entropy] averaged across distances/angles.
    Uses a sentinel trick so only pairs fully inside the mask contribute.
    """
    ys, xs = np.where(mask)
    if len(ys) < 20:
        # region too small: fallback to simple intensity stats
        region = qimg[mask].astype(np.float32)
        v = float(np.var(region)) if region.size else 0.0
        return np.array([v, 1.0/(1.0+v), 0.0, 0.0], dtype=np.float32)

    # crop to bbox for speed
    y0, y1 = ys.min(), ys.max()+1
    x0, x1 = xs.min(), xs.max()+1
    patch = qimg[y0:y1, x0:x1]
    mpatch = mask[y0:y1, x0:x1]

    # Sentinel trick: shift valid codes by +1, set outside to 0
    sentinel_shift = 1
    patch2 = patch.copy().astype(np.uint8)
    patch2[mpatch] = patch2[mpatch] + sentinel_shift
    patch2[~mpatch] = 0
    L = levels + sentinel_shift

    glcm = graycomatrix(patch2, distances=distances, angles=angles,
                        levels=L, symmetric=True, normed=True)
    # drop sentinel row/col
    glcm = glcm[1:, 1:, :, :]
    glcm = glcm / (glcm.sum(axis=(0,1), keepdims=True) + 1e-12)

    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()

    p = glcm + 1e-12
    entropy = (-p * np.log(p)).sum(axis=(0,1)).mean()

    return np.array([contrast, homogeneity, energy, entropy], dtype=np.float32)

def glcm_features_per_superpixel(img_u8, spx, levels=32, distances=(1,2), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    """Compute a 4D GLCM feature vector per superpixel; return z-scored features (K,4)."""
    qimg = quantize_image_u8(img_u8, levels=levels)
    K = int(spx.max()) + 1
    feats = np.zeros((K, 4), dtype=np.float32)
    for k in range(K):
        feats[k] = glcm_features_for_region(qimg, (spx == k), levels=levels,
                                            distances=distances, angles=angles)
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True) + 1e-6
    return (feats - mu) / sd

def cosine_distance_rows(a, b):
    """Cosine distance for corresponding rows of a and b (N,D)."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return 1.0 - np.einsum('nd,nd->n', an, bn)

def glcm_based_pairwise_weights(img_u8, spx, lam=20.0, levels=32, distances=(1,2), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    """
    Build Potts weights w_ij = lam * L_ij * exp(-beta * dtex^2),
    where dtex = cosine distance between GLCM feature vectors of neighboring superpixels.
    """
    E, Lij = region_adjacency_and_boundary(spx)
    if len(E) == 0:
        return [], np.zeros((0,), dtype=np.float64)

    feats = glcm_features_per_superpixel(img_u8, spx, levels=levels, distances=distances, angles=angles)

    i_idx = np.array([i for i, _ in E], dtype=int)
    j_idx = np.array([j for _, j in E], dtype=int)
    dtex = cosine_distance_rows(feats[i_idx], feats[j_idx])

    var = np.var(dtex) + 1e-6
    beta = 1.0 / (2.0 * var)
    w = lam * Lij * np.exp(-beta * (dtex ** 2))
    return E, w.astype(np.float64)

# -------------------- Unary construction (probs or hard labels) --------------------

def region_unaries_from_probs(spx, probs, ignore_index=None):
    """
    From pixel probabilities -> per-region mean probs -> unaries = -log(mean_p)
    spx:   (H,W) ints [0..K-1]
    probs: (H,W,C) float in [0,1], sum=1
    """
    H, W, C = probs.shape
    K = int(spx.max()) + 1
    sums = np.zeros((K, C), dtype=np.float64)
    counts = np.bincount(spx.ravel(), minlength=K).astype(np.float64)

    flat_spx = spx.ravel()
    for c in range(C):
        sums[:, c] = np.bincount(flat_spx, weights=probs[..., c].ravel(), minlength=K)

    eps = 1e-6
    mean_p = sums / (counts[:, None] + eps)
    if ignore_index is not None:
        mean_p[:, ignore_index] = np.minimum(mean_p[:, ignore_index], 1e-6)

    return -np.log(mean_p + eps)  # (K,C)

def region_unaries_from_hard(spx, hard_pred, num_classes=None, ignore_index=None, smoothing=1.0):
    """
    From pixel hard labels -> per-region label histograms (Laplace smoothing) -> unaries = -log(p)
    """
    if num_classes is None:
        num_classes = int(hard_pred.max()) + 1
    K = int(spx.max()) + 1
    counts = np.zeros((K, num_classes), dtype=np.float64)

    flat_spx = spx.ravel()
    flat_pred = hard_pred.ravel()
    for k in range(K):
        m = (flat_spx == k)
        if m.any():
            hist = np.bincount(flat_pred[m], minlength=num_classes).astype(np.float64)
        else:
            hist = np.zeros((num_classes,), dtype=np.float64)
        counts[k] = hist

    if ignore_index is not None:
        counts[:, ignore_index] = 0.0

    size = counts.sum(1, keepdims=True)
    probs = (counts + smoothing) / (size + smoothing * num_classes + 1e-6)
    if ignore_index is not None:
        probs[:, ignore_index] = np.minimum(probs[:, ignore_index], 1e-6)

    return -np.log(probs + 1e-6)  # (K,C)

# -------------------- Graph cut solvers (pygco preferred; pymaxflow fallback) --------------------

def cut_multilabel_potts(unaries, edges, weights, prefer_pygco=True):
    """
    Solve min-labeling with Potts model:
      E = sum_i U[i, L_i] + sum_(i,j) w_ij * [L_i != L_j]
    Returns labels (K,) int.
    """
    K, C = unaries.shape

    # Try pygco
    if prefer_pygco:
        try:
            import pygco
            scale = 1000.0
            U_int = (unaries * scale).astype(np.int32)
            pairwise_cost = (np.ones((C, C), dtype=np.int32) - np.eye(C, dtype=np.int32))
            edges_arr = np.array(edges, dtype=np.int32)
            w_int = (weights * scale).astype(np.int32)
            labels = pygco.cut_general_graph(edges_arr, w_int, U_int, pairwise_cost, n_iter=5)
            return labels.astype(np.int32)
        except Exception:
            pass

    # Fallback: minimal α-expansion using PyMaxflow (binary moves)
    import maxflow

    # init with unary argmin
    labels = unaries.argmin(axis=1).astype(np.int32)

    def energy(lbls):
        e = unaries[np.arange(K), lbls].sum()
        for (i, j), w in zip(edges, weights):
            if lbls[i] != lbls[j]:
                e += w
        return float(e)

    def binary_move(current, alpha):
        g = maxflow.Graph[float](K, len(edges))
        nodes = g.add_nodes(K)
        for k in range(K):
            ca = float(unaries[k, alpha])
            ck = float(unaries[k, current[k]])
            g.add_tedge(nodes[k], ca, ck)
        for (i, j), w in zip(edges, weights):
            if i == j or w <= 0:
                continue
            g.add_edge(nodes[i], nodes[j], float(w), float(w))
        g.maxflow()
        new = current.copy()
        for k in range(K):
            if g.get_segment(nodes[k]) == 0:  # source => choose alpha
                new[k] = alpha
        return new

    improved = True
    prevE = energy(labels)
    while improved:
        improved = False
        for alpha in range(C):
            cand = binary_move(labels, alpha)
            E_c = energy(cand)
            if E_c < prevE - 1e-6:
                labels, prevE = cand, E_c
                improved = True
    return labels.astype(np.int32)

# -------------------- Main entry: refine with superpixel + GLCM + graph cut --------------------

def refine_with_superpixel_graphcut_glcm(
    image_gray, spx, probs=None, hard_pred=None,
    ignore_index=0, allow_background=False,
    lam=25.0, levels=32, distances=(1,2), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4),
    prefer_pygco=True
):
    """
    Returns:
      refined_labels: (H,W) int
    """
    if probs is None and hard_pred is None:
        raise ValueError("Provide either `probs` (H,W,C) or `hard_pred` (H,W).")

    # standardize image to uint8 for GLCM
    img = image_gray
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # Unaries
    if probs is not None:
        unaries = region_unaries_from_probs(spx, probs, ignore_index=ignore_index)
        C = probs.shape[-1]
    else:
        C = int(hard_pred.max()) + 1
        unaries = region_unaries_from_hard(spx, hard_pred, num_classes=C, ignore_index=ignore_index, smoothing=1.0)

    # Label allow-list
    if allow_background:
        keep = np.arange(C, dtype=int)
    else:
        keep = np.array([c for c in range(C) if c != ignore_index], dtype=int)

    # Slice unaries to allowed labels and remember mapping
    label_map = np.full(C, -1, dtype=int)
    label_map[keep] = np.arange(len(keep))
    U_eff = unaries[:, keep]
    C_eff = U_eff.shape[1]

    # Pairwise GLCM-based Potts weights
    edges, weights = glcm_based_pairwise_weights(
        img_u8=img, spx=spx, lam=lam, levels=levels, distances=distances, angles=angles
    )

    # Solve
    lab_eff = cut_multilabel_potts(U_eff, edges, weights, prefer_pygco=prefer_pygco)

    # Map back to original label ids
    inv = np.where(label_map >= 0)[0]
    labels_region = inv[lab_eff].astype(np.int32)

    # Paint to pixels
    refined = labels_region[spx]
    return refined

# -------------------- Tiny usage example --------------------
if __name__ == "__main__":
    # Dummy shapes to illustrate usage:
    H, W, C = 512, 512, 5     # 5 classes (0=background)
    K = 800                   # ~800 superpixels

    # Fake inputs (replace with your real data)
    image_gray = (np.random.rand(H, W) * 255).astype(np.uint8)
    spx = np.random.randint(0, K, size=(H, W), dtype=np.int32)

    # Option A: with softmax probs
    probs = np.random.rand(H, W, C).astype(np.float32)
    probs /= probs.sum(axis=-1, keepdims=True)

    # Option B: with hard labels only
    # hard_pred = probs.argmax(axis=-1).astype(np.int32)

    refined = refine_with_superpixel_graphcut_glcm(
        image_gray=image_gray,
        spx=spx,
        probs=probs,                 # or hard_pred=hard_pred
        ignore_index=0,
        allow_background=False,
        lam=25.0,
        levels=32,
        distances=(1,2),
        angles=(0, np.pi/4, np.pi/2, 3*np.pi/4),
        prefer_pygco=True
    )
    print("Refined shape:", refined.shape, "labels:", np.unique(refined))
