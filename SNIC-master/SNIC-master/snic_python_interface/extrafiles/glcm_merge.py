import numpy as np
import networkx as nx
from skimage import exposure
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops
from sklearn.preprocessing import MinMaxScaler
from skimage import graph
import matplotlib.pyplot as plt


def get_glcm_features(region_mask, image_uint8):
    """
    Extract Contrast, Energy, Correlation for a region.
    Returns [0,0,0] if invalid or uniform.
    image_uint8 must be uint8 (0..255).
    """
    coords = np.argwhere(region_mask)
    if coords.size == 0:
        return np.zeros(3, dtype=np.float32)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    crop = image_uint8[y0:y1, x0:x1]
    crop_mask = region_mask[y0:y1, x0:x1]

    # Mask out pixels outside region -> 0
    masked_crop = np.where(crop_mask, crop, 0).astype(np.uint8)

    glcm = graycomatrix(
        masked_crop,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    # Ignore masked-out zeros influence
    glcm[0, 0, :, :] = 0

    try:
        cont = graycoprops(glcm, 'contrast')[0, 0]
        ener = graycoprops(glcm, 'energy')[0, 0]
        corr = graycoprops(glcm, 'correlation')[0, 0]
        feat = np.array([cont, ener, corr], dtype=np.float32)
        return np.nan_to_num(feat).astype(np.float32)
    except Exception:
        return np.zeros(3, dtype=np.float32)


def weight_glcm(rag, src, dst, n):
    """
    Edge weight between two nodes = Euclidean distance of normalized feature vectors.
    Must return dict: {'weight': value}
    """
    diff = rag.nodes[dst]['features'] - rag.nodes[src]['features']
    return {'weight': float(np.linalg.norm(diff))}


def draw_adjacent_superpixel_distance_hist(rag, ignore_background=True, bg_id=0):
    """
    Draw histogram of Euclidean distances between adjacent superpixels.

    Parameters
    ----------
    rag : skimage.graph.RAG
        RAG with node attribute 'features' (normalized vectors).
    ignore_background : bool
        Whether to ignore edges connected to background.
    bg_id : int
        Background label id (usually 0).
    """

    distances = []

    for u, v in rag.edges():
        if ignore_background and (u == bg_id or v == bg_id):
            continue

        f_u = rag.nodes[u]['features']
        f_v = rag.nodes[v]['features']

        d = np.linalg.norm(f_u - f_v)
        distances.append(d)

    distances = np.asarray(distances)

    if distances.size == 0:
        print("No valid adjacent superpixel distances to plot.")
        return

    plt.figure()
    plt.hist(distances, bins=50)
    plt.xlabel("Euclidean distance (GLCM feature space)")
    plt.ylabel("Frequency")
    plt.title("Adjacent Superpixel Distance Distribution")

    plt.show()







def merge_glcm(rag, src, dst):
    """
    When merging src -> dst, update dst's features by pixel-count weighted average.
    Also update dst pixel_count.
    NOTE: rag.merge_nodes() will handle 'labels' attribute if it exists.
    """
    p_src = rag.nodes[src]['pixel_count']
    p_dst = rag.nodes[dst]['pixel_count']
    total = p_src + p_dst

    if total <= 0:
        return

    w_src = p_src / total
    w_dst = p_dst / total

    rag.nodes[dst]['features'] = rag.nodes[src]['features'] * w_src + rag.nodes[dst]['features'] * w_dst
    rag.nodes[dst]['pixel_count'] = total


def _collapse_black_to_background(image, labels, black_threshold=0.1):
    """
    Collapse any superpixel whose mean intensity < black_threshold into background label 0.
    This effectively removes black/background superpixels from merging.
    """
    labels = labels.astype(np.int32, copy=True)

    unique_ids = np.unique(labels)
    for sid in unique_ids:
        mask = (labels == sid)
        if mask.sum() == 0:
            continue
        if float(np.mean(image[mask])) < black_threshold:
            labels[mask] = 0  # background

    return labels


def _relabel_consecutive(labels, background_id=0):
    """
    Relabel all non-background labels to 1..K consecutively (keeps background_id as 0).
    This makes the RAG cleaner / consistent.
    """
    labels = labels.astype(np.int32, copy=True)
    uniq = np.unique(labels)

    # keep background
    uniq_fg = [u for u in uniq if u != background_id]

    mapping = {background_id: background_id}
    new_id = 1
    for u in uniq_fg:
        mapping[u] = new_id
        new_id += 1

    out = np.zeros_like(labels, dtype=np.int32)
    for old, new in mapping.items():
        out[labels == old] = new

    return out


def iterative_glcm_merge(image, labels, threshold=0.01, black_threshold=0.1, keep_background=True):
    """
    Merge adjacent superpixels based on GLCM texture similarity.

    Parameters
    ----------
    image : 2D array
        Grayscale image (uint8 recommended; if not, will be converted).
    labels : 2D int array
        Superpixel labels.
    threshold : float
        Merge threshold in feature-distance space (after normalization).
        Smaller => fewer merges. Larger => more merges.
    black_threshold : float
        A region with mean intensity < black_threshold is treated as background.
        For true black background, black_threshold=1 is reasonable.
    keep_background : bool
        If True, background remains label 0. If False, background is dropped by setting it to -1 (optional).

    Returns
    -------
    merged_labels : 2D int array
        Labels after merging.
    """
    if image.ndim != 2:
        raise ValueError("iterative_glcm_merge expects a single-channel (H,W) grayscale image.")

    # Convert image to uint8 for GLCM (0..255)
    if image.dtype != np.uint8:
        img = image.astype(np.float32)
        img = np.clip(img, 0, 255)
        image_u8 = img.astype(np.uint8)
    else:
        image_u8 = image

    labels_in = labels.astype(np.int32)

    # 1) Collapse black regions into background label 0
    labels_bg = _collapse_black_to_background(image_u8, labels_in, black_threshold=black_threshold)

    # 2) Make labels consecutive (0,1..K)
    labels_bg = _relabel_consecutive(labels_bg, background_id=0)

    # Optional: drop background entirely (not usually necessary)
    if not keep_background:
        labels_bg = labels_bg.copy()
        labels_bg[labels_bg == 0] = -1

    # 3) Build RAG (ignore bg=-1 automatically if used; if bg=0, it will be present as one node)
    rag = graph.RAG(labels_bg)

    # IMPORTANT: merge_hierarchical expects each node to have 'labels' attribute
    for n in rag.nodes:
        rag.nodes[n]['labels'] = [n]

    # 4) Extract features for every node
    nodes = list(rag.nodes)
    all_features = []

    for node in nodes:
        mask = (labels_bg == node)
        pix = int(np.count_nonzero(mask))

        rag.nodes[node]['pixel_count'] = pix

        # Background node 0: just give zero features and don't encourage merging
        if node == 0 and keep_background:
            feat = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            # For non-background, compute real GLCM features
            feat = get_glcm_features(mask, image_u8)

        rag.nodes[node]['features'] = feat
        all_features.append(feat)

    all_features = np.asarray(all_features, dtype=np.float32)

    # 5) Normalize features to 0..1
    scaler = MinMaxScaler()
    norm_feats = scaler.fit_transform(all_features)

    for i, node in enumerate(nodes):
        rag.nodes[node]['features'] = norm_feats[i].astype(np.float32)
    
    #draw_adjacent_superpixel_distance_hist(rag,True,0)

    # 6) Initialize edge weights
    for u, v, d in rag.edges(data=True):
        # Prevent background from merging (if keep_background)
        if keep_background and (u == 0 or v == 0):
            d['weight'] = np.inf
        else:
            d['weight'] = float(np.linalg.norm(rag.nodes[u]['features'] - rag.nodes[v]['features']))

    # 7) Hierarchical merge
    merged_labels = graph.merge_hierarchical(
        labels_bg,
        rag,
        thresh=threshold,
        rag_copy=False,
        in_place_merge=True,
        merge_func=merge_glcm,
        weight_func=weight_glcm
    )

    return merged_labels