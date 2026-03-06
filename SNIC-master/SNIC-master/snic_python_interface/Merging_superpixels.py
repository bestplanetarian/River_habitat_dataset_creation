import numpy as np
import cv2
from collections import defaultdict, deque
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

# ----------------------------
# Utilities (same as before)
# ----------------------------

def build_superpixel_adjacency(superpixel_map):
    adjacency = defaultdict(set)
    diff_r = superpixel_map[:, :-1] != superpixel_map[:, 1:]
    u = superpixel_map[:, :-1][diff_r]
    v = superpixel_map[:, 1:][diff_r]
    for a, b in zip(u.flat, v.flat):
        adjacency[a].add(b); adjacency[b].add(a)
    diff_d = superpixel_map[:-1, :] != superpixel_map[1:, :]
    u = superpixel_map[:-1, :][diff_d]
    v = superpixel_map[1:, :][diff_d]
    for a, b in zip(u.flat, v.flat):
        adjacency[a].add(b); adjacency[b].add(a)
    return adjacency

def _quantize_to_levels(img_u8, levels=32):
    if levels <= 1:
        return np.zeros_like(img_u8, dtype=np.uint8)
    scale = levels / 256.0
    q = np.floor(img_u8.astype(np.float32) * scale).astype(np.int32)
    q[q >= levels] = levels - 1
    return q.astype(np.int32)


# ----------------------------
# Main function with DBSCAN
# ----------------------------

import numpy as np
from collections import defaultdict, deque
from sklearn.cluster import KMeans

# assumes you already have:
# build_superpixel_adjacency(sp_map) -> dict {lab: set(neighbor_labs)}

import numpy as np
from collections import defaultdict, deque
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def build_superpixel_adjacency(sp_map: np.ndarray):
    """
    Fast 4-neighborhood adjacency from a label map.
    Returns: dict {label: set(neighbor_labels)}
    """
    sp_map = sp_map.astype(np.int32, copy=False)
    H, W = sp_map.shape
    adj = defaultdict(set)

    # Right neighbors
    a = sp_map[:, :-1]
    b = sp_map[:, 1:]
    mask = a != b
    if np.any(mask):
        u = a[mask].ravel()
        v = b[mask].ravel()
        for uu, vv in zip(u, v):
            adj[int(uu)].add(int(vv))
            adj[int(vv)].add(int(uu))

    # Down neighbors
    a = sp_map[:-1, :]
    b = sp_map[1:, :]
    mask = a != b
    if np.any(mask):
        u = a[mask].ravel()
        v = b[mask].ravel()
        for uu, vv in zip(u, v):
            adj[int(uu)].add(int(vv))
            adj[int(vv)].add(int(uu))

    return adj


def merge_superpixels_by_adjacency_kmeans(
    img: np.ndarray,
    superpixel_map: np.ndarray,
    min_sp_area: int = 50,
    black_thresh: float = 1.0,          # <=1 for uint8 "near black"
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 0,
    n_init: int = 20,
    max_iter: int = 300,
):
    """
    Pipeline:
      1) Compute mean intensity per superpixel.
      2) Select VALID superpixels: not near-black AND not tiny.
         - tiny superpixels are kept unchanged (NOT merged).
         - near-black superpixels are kept unchanged (NOT merged).
      3) Auto-pick K using silhouette score on valid superpixels.
      4) KMeans cluster valid superpixels by mean intensity.
      5) Merge adjacent valid superpixels with the same cluster label (BFS).
         - merged component id = min(original labels in that component),
           so IDs don't "start at 1" unless min-label is 1.
      6) Output merged_map:
         - valid merged components get new ids (min label in component)
         - invalid/tiny/near-black superpixels keep original labels
    """
    sp_map = superpixel_map.astype(np.int32, copy=False)
    if img.shape != sp_map.shape:
        raise ValueError(f"Shape mismatch: image {img.shape} vs superpixel_map {sp_map.shape}")

    # Ensure float for mean computation
    img_f = img.astype(np.float32, copy=False)

    super_ids = np.unique(sp_map)

    # Compute area + mean intensity for each superpixel id
    # (vectorized approach: loop over ids is usually fine for typical SP counts)
    area = {}
    mean_int = {}
    for lab in super_ids:
        mask = (sp_map == lab)
        a = int(mask.sum())
        area[int(lab)] = a
        mean_int[int(lab)] = float(img_f[mask].mean()) if a > 0 else 0.0

    # Valid = not near-black AND not tiny
    valid_ids = [lab for lab in super_ids
                 if (mean_int[int(lab)] > float(black_thresh)) and (area[int(lab)] >= int(min_sp_area))]

    # If too few valid superpixels, nothing meaningful to merge
    if len(valid_ids) < 2:
        return sp_map.copy()

    # Features for clustering: 1-D mean intensity
    X = np.array([[mean_int[int(lab)]] for lab in valid_ids], dtype=np.float64)

    # Auto choose K using silhouette score
    # Guard: silhouette requires k <= n_samples-1
    max_k_allowed = min(int(k_max), len(valid_ids) - 1)
    min_k_allowed = max(int(k_min), 2)

    if max_k_allowed < min_k_allowed:
        # Not enough samples to choose K robustly; use k=2
        best_k = 4
    else:
        best_k = None
        best_score = -np.inf
        for k in range(min_k_allowed, max_k_allowed + 1):
            km_tmp = KMeans(
                n_clusters=k,
                random_state=random_state,
                n_init=n_init,
                max_iter=max_iter
            )
            labels_tmp = km_tmp.fit_predict(X)
            # If KMeans collapses to 1 cluster due to duplicates, skip
            if len(np.unique(labels_tmp)) < 2:
                continue
            score = silhouette_score(X, labels_tmp)
            if score > best_score:
                best_score = score
                best_k = k

        if best_k is None:
            best_k = 4

    # Final KMeans
    km = KMeans(
        n_clusters=best_k,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter
    )
    valid_cluster = km.fit_predict(X)  # size = len(valid_ids)

    valid_id_to_cluster = {int(lab): int(valid_cluster[i]) for i, lab in enumerate(valid_ids)}

    # Adjacency
    adjacency = build_superpixel_adjacency(sp_map)

    # BFS merge among VALID superpixels only, restricted by same cluster
    visited = set()
    component_members = []  # list of lists

    valid_set = set(int(l) for l in valid_ids)

    for lab in valid_ids:
        lab = int(lab)
        if lab in visited:
            continue
        # Start BFS from this valid superpixel
        q = deque([lab])
        visited.add(lab)
        comp = [lab]

        cl = valid_id_to_cluster[lab]

        while q:
            cur = q.popleft()
            for nb in adjacency.get(cur, []):
                nb = int(nb)
                if nb in visited:
                    continue
                if nb not in valid_set:
                    continue
                if valid_id_to_cluster.get(nb, None) != cl:
                    continue
                visited.add(nb)
                q.append(nb)
                comp.append(nb)

        component_members.append(comp)

    # Build mapping: each component id = min original id in that component
    superpixel_to_merged = {}
    for comp in component_members:
        merged_id = int(min(comp))
        for lab in comp:
            superpixel_to_merged[int(lab)] = merged_id

    # Construct output:
    # - valid merged -> mapped id
    # - invalid/tiny/near-black -> keep original label
    merged_map = sp_map.copy()
    for lab, mid in superpixel_to_merged.items():
        merged_map[sp_map == lab] = mid

    return merged_map

