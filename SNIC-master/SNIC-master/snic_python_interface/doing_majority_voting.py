import numpy as np
from scipy.stats import mode
from PIL import Image


'''
def smooth_with_superpixel_threshold(pred_mask, superpixels,
                                     shadow_class=1,
                                     other_class=2,
                                     shadow_thresh=0.30):
    """
    For each superpixel:
      - compute the fraction of pixels labelled `shadow_class`
      - if frac >= shadow_thresh → assign entire superpixel to shadow
      - else if frac <= (1 - shadow_thresh) → assign entire to other_class
      - else → leave original per-pixel labels untouched (preserve detail)
    """
    
    """
    For each superpixel:
      - Collect only pixels where pred_mask != 0 (i.e., shadow or other).
      - If none, skip that superpixel.
      - Otherwise compute frac_shadow among those pixels and vote.
    """
    refined = pred_mask.copy()
    for sp_id in np.unique(superpixels):
        # Boolean mask for this SP
        sp_mask = (superpixels == sp_id)

        # Restrict to non-background pixels in the prediction
        vote_mask = sp_mask & (pred_mask != 0)
        if not np.any(vote_mask):
            # No shadow/other pixels here → skip voting
            continue

        # Extract those labels
        labels = pred_mask[vote_mask]
        # Compute fraction that are shadow
        frac_shadow = np.mean(labels == shadow_class)

        if frac_shadow >= shadow_thresh:
            refined[sp_mask] = shadow_class
        elif (1 - frac_shadow) >= shadow_thresh:
            refined[sp_mask] = other_class
        
        # else leave per-pixel predictions untouched

    return refined
'''

def smooth_with_superpixel_mode(pred_mask, superpixels):
    """
    For each superpixel:
      - Collect all non-zero prediction pixels.
      - If none, skip that superpixel.
      - Otherwise assign the entire superpixel to the mode label
        among those non-zero predictions.
    """
    refined = pred_mask.copy()
    for sp_id in np.unique(superpixels):
        sp_mask = (superpixels == sp_id)
        vote_mask = sp_mask & (pred_mask != 0)

        if not np.any(vote_mask):
            continue

        labels = pred_mask[vote_mask]
        most_common = mode(labels, keepdims=True)[0][0]
        refined[sp_mask] = most_common

    return refined





























    '''
    refined = pred_mask.copy()
    pm_flat = pred_mask.ravel()
    sp_flat = superpixels.ravel()

    # 1) Keep only foreground votes
    fg_mask = (pm_flat == shadow_class) | (pm_flat == other_class)
    ids = sp_flat[fg_mask]
    labs = pm_flat[fg_mask]

    # 2) Count total FG pixels per superpixel
    max_sp = sp_flat.max()
    fg_counts = np.bincount(ids, minlength=max_sp+1)

    # 3) Count shadow‐class pixels per superpixel
    shadow_counts = np.bincount(ids[labs==shadow_class], minlength=max_sp+1)

    # 4) Compute fraction of shadow for each SP id that appears
    #    To avoid division by zero, only compute where fg_counts>0
    sp_ids = np.nonzero(fg_counts)[0]
    frac_shadow = shadow_counts[sp_ids] / fg_counts[sp_ids]

    # 5) Decide class for each superpixel
    #    1=shadow where frac>=thresh; 2=other where (1-frac)>=thresh
    shadow_sps = sp_ids[frac_shadow >= shadow_thresh]
    other_sps  = sp_ids[(1-frac_shadow) >= shadow_thresh]

    # 6) Broadcast back in two passes
    mask = np.isin(superpixels, shadow_sps)
    refined[mask] = shadow_class
    mask = np.isin(superpixels, other_sps)
    refined[mask] = other_class

    return refined
    '''











def majority_vote(prediction, superpixels, outpath):
    """
    Apply majority voting within superpixel clusters, ignoring background pixels.
    
    Args:
        prediction (np.ndarray): 2D array of predicted labels.
        superpixels (np.ndarray): 2D array of superpixel cluster indices.
    
    Returns:
        np.ndarray: Updated prediction after applying majority voting.
    """
    # Ensure both arrays have the same shape
    assert prediction.shape == superpixels.shape, "Shape mismatch between prediction and superpixels"

    # Define background mask (black pixels in prediction)
    background_mask = (prediction == 0)

    # Initialize output prediction (copy original)
    updated_prediction = prediction.copy()

    # Get unique superpixel labels
    unique_clusters = np.unique(superpixels)

    
    '''
    for cluster_id in unique_clusters:
        # Get the mask for the current cluster
        cluster_mask = (superpixels == cluster_id)

        # Ignore clusters where the entire region is masked (background)
        if np.all(background_mask[cluster_mask]):
            continue
        
        # Extract valid prediction values (non-background) within the cluster
        cluster_values = prediction[cluster_mask & ~background_mask]

        if len(cluster_values) > 0:
            # Apply majority voting
            majority_label = mode(cluster_values, keepdims=True)[0][0]

            # Assign majority label back to the cluster (excluding background pixels)
            updated_prediction[cluster_mask & ~background_mask] = majority_label

    return updated_prediction
    '''


    for cluster_id in unique_clusters:
        # Mask for the current cluster
        cluster_mask = (superpixels == cluster_id)

    # Skip if entire cluster is background
        if np.all(background_mask[cluster_mask]):
           continue

    # Extract non-background labels in the cluster
        cluster_values = prediction[cluster_mask & ~background_mask]

        if len(cluster_values) > 0:
           # compute the fraction of class 1 in this cluster
            percent_class1 = np.sum(cluster_values == 1) / len(cluster_values)

           # apply your 55% rule
            if percent_class1 > 0.5:
               cluster_label = 1
            else:
               cluster_label = 2

            # assign back to all non-background pixels in the cluster
            updated_prediction[cluster_mask & ~background_mask] = cluster_label
    
    img_to_save = Image.fromarray((updated_prediction.astype(np.uint8)))
    img_to_save.save(outpath)

    return updated_prediction
