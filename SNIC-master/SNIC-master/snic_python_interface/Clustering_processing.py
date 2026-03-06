
import numpy as np
from skimage import io, segmentation







'''

if not features:
        return {}

    sp_ids, vecs = zip(*features)
    X_feat = np.stack(vecs)               # shape [N, D]
    # 1) Normalize CNN features
    X_norm = X_feat / (np.linalg.norm(X_feat, axis=1, keepdims=True) + 1e-10)

    # 2) Compute mean intensity per superpixel
    means = []
    for sp_id in sp_ids:
        mask = (sp_map == sp_id)
        means.append(image_arr[mask].mean() / 255.0)  # scale to [0,1]
    I = np.array(means)[:,None]  # shape [N,1]

    # 3) Weight that intensity dimension
    I_weighted = I * intensity_weight

    # 4) Concatenate
    X = np.hstack([X_norm, I_weighted])  # now shape [N, D+1]

    # 5) Run DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(X)
    labels = db.labels_

    # 6) Build clusters dict
    clusters = {}
    for sp_id, lbl in zip(sp_ids, labels):
        clusters.setdefault(int(lbl), []).append(sp_id)

    return clusters

'''

def split_superpixels_by_intensity(image_gray, sp_map):
    """
    Splits superpixels into two equal‐sized groups based on mean intensity.

    Returns
    -------
    final_map : 2D int array
        Pixels in the bottom 50% darkest SPs → label 1
        Pixels in the top 50% brightest SPs → label 2
    """
    '''
    # 1) Compute per‐superpixel mean intensity
    sp_ids = np.unique(sp_map)
    means = np.empty_like(sp_ids, dtype=float)
    for i, sp in enumerate(sp_ids):
        means[i] = image_gray[sp_map == sp].mean()

    # 2) Sort superpixels by mean
    order = np.argsort(means)
    # bottom half → class 1, top half → class 2
    cutoff = len(order) // 2
    class1_sp = sp_ids[order[:cutoff]]
    class2_sp = sp_ids[order[cutoff:]]

    # 3) Build the final label map
    final_map = np.zeros_like(sp_map, dtype=np.uint8)
    for sp in class1_sp:
        final_map[sp_map == sp] = 1
    for sp in class2_sp:
        final_map[sp_map == sp] = 2

    return final_map
    '''
    
    # 1) Compute per‐superpixel mean intensity
    sp_ids = np.unique(sp_map)
    means  = np.array([image_gray[sp_map == sp].mean() for sp in sp_ids])

    # 2) Sort superpixels by mean and split at the given ratio
    order  = np.argsort(means)
    cutoff = int(len(order) * 0.3)
    low_sps  = sp_ids[order[:cutoff]]
    high_sps = sp_ids[order[cutoff:]]

    # 3) Build the final label map
    final_map = np.zeros_like(sp_map, dtype=np.uint8)
    for sp in low_sps:
        final_map[sp_map == sp] = 1
    for sp in high_sps:
        final_map[sp_map == sp] = 2

    return final_map
    


    '''
     # 1) Flatten arrays and ensure numpy type
    sp_arr = np.asarray(sp_map, dtype=int)
    img_flat = np.asarray(image_gray, dtype=float).ravel()
    sp_flat = sp_arr.ravel()

    # 2) Mask out negative superpixel IDs
    valid_mask = sp_flat >= 0
    sp_valid = sp_flat[valid_mask]
    img_valid = img_flat[valid_mask]

    # 3) Count pixels per superpixel
    max_sp = sp_valid.max()
    counts = np.bincount(sp_valid, minlength=max_sp+1)

    # 4) Sum intensities per superpixel
    sums = np.bincount(sp_valid, weights=img_valid, minlength=max_sp+1)

    # 5) Compute mean intensity, avoid division by zero
    means = np.zeros_like(sums, dtype=float)
    nonzero = counts > 0
    means[nonzero] = sums[nonzero] / counts[nonzero]

    # 6) Determine cutoff for the darkest 'ratio' fraction
    sp_ids = np.nonzero(nonzero)[0]
    ordered = sp_ids[np.argsort(means[sp_ids])]
    cutoff = int(len(ordered) * 0.45)
    low_sps = ordered[:cutoff]
    high_sps = ordered[cutoff:]

    # 7) Build a lookup mapping from superpixel ID to class
    lookup = np.zeros(max_sp+1, dtype=np.uint8)
    lookup[low_sps] = 1
    lookup[high_sps] = 2

    # 8) Apply mapping to entire map (negative IDs stay 0)
    final_map = np.zeros_like(sp_arr, dtype=np.uint8)
    # Only label non-negative IDs
    mask = sp_arr >= 0
    final_map[mask] = lookup[sp_arr[mask]]

    return final_map
    '''






























