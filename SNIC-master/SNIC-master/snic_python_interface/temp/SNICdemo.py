

import os
import subprocess
from PIL import Image
from skimage.io import imread,imshow
import numpy as np
from timeit import default_timer as timer
from _snic.lib import SNIC_main
from cffi import FFI
from scipy.stats import mode
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
import gc
import sys
#import Using_RAG
#import majority_voting_allimages
#import Superpixel_CRF

from pathlib import Path
rt = Path(__file__).resolve()
rt = rt.parents[3]

sys.path.insert(0, str(rt))
from scripts.runinference import parse_args

from skimage.color import rgb2lab
from skimage.feature import local_binary_pattern
from skimage import graph
from scipy.ndimage import sobel

Image.MAX_IMAGE_PIXELS = None

import torch
import torch.nn as nn
from torchvision import models, transforms
import doing_majority_voting

from skimage import io, segmentation, color

from skimage.filters import threshold_otsu
from skimage import color, filters

from skimage import color, segmentation, io, future

#import Clustering_processing

from skimage import io, segmentation, measure, color

import Merging_superpixels

from sklearn.metrics import confusion_matrix
#import GLCM_superpixel_relabeling
#import glcm_merge

try:
    from skimage import graph
    rag_mean_color = graph.rag_mean_color
    cut_threshold = graph.cut_threshold
except ImportError:
    from skimage.future import graph
    rag_mean_color = graph.rag_mean_color
    cut_threshold = graph.cut_threshold

try:
    import skimage.graph as graph
except ImportError:
    from skimage.future import graph




numsuperpixels = 3000
compactness = 5.0
doRGBtoLAB = False# only works if it is a three channel image








def drawBoundaries(imgname,labels,numlabels, bound_save):

    img = Image.open(imgname).convert("RGB")

    new_size = (img.width // 4, img.height // 4)
    img_resized = img.resize(new_size, resample=Image.NEAREST)

    # Convert resized image to NumPy array
    img_resized1 = np.array(img)
	#img = np.array(img)
	#print(img.shape)

    ht,wd = labels.shape
    
    radius = 1

    for y in range(radius, ht-radius):
        for x in range(radius, wd-radius):
            center_label = labels[y, x]
        
            # check if any neighbor in radius differs
            neighborhood = labels[y-radius:y+radius+1, x-radius:x+radius+1]
            if np.any(neighborhood != center_label):
                img_resized1[y, x] = [255, 0, 0]   # red boundary


    Image.fromarray(img_resized1).save(bound_save)

    return img_resized1



def merge_all_black_superpixels(merged_map, sp_map, image_gray):
    """
    Take your already‐merged label map and then force all superpixels
    that are entirely black (in image_gray) to share one label.
    
    Parameters
    ----------
    merged_map : 2D int array
        The post‐RAG merge label map (H×W).
    sp_map : 2D int array
        Original superpixel IDs (H×W).
    image_gray : 2D array
        Grayscale image (H×W) with zeros for black.
    
    Returns
    -------
    new_map : 2D int array
        Same as merged_map but with one unified label for all-black SPs.
    """
    new_map = merged_map.copy()

    print(sp_map.shape)
    
    # 1) Find which superpixels are completely black
    all_black = []
    for sp_id in np.unique(sp_map):
        mask = (sp_map == sp_id)
        # If every pixel in that SP is black (0)
        if np.all(image_gray[mask] == 0):
            all_black.append(sp_id)
    if not all_black:
        return new_map
    
    # 2) Pick a label to use for the unified black region
    #    (here: take the minimum merged label among those SPs)
    black_labels = np.unique(new_map[np.isin(sp_map, all_black)])
    target_label = int(black_labels.min())
    
    # 3) Overwrite all those pixels to the target_label
    new_map[np.isin(sp_map, all_black)] = target_label
    return new_map






def split_image(img, direction):
    if direction == 'horizontal':
        split_size = img.shape[1] // 3
        return [img[:, i*split_size:(i+1)*split_size] for i in range(3)]
    elif direction == 'vertical':
        split_size = img.shape[0] // 3
        return [img[i*split_size:(i+1)*split_size, :] for i in range(3)]
    elif direction == 'all':
        h_split = img.shape[0] // 3
        w_split = img.shape[1] // 3
        return [img[i*h_split:(i+1)*h_split, j*w_split:(j+1)*w_split] for i in range(3) for j in range(3)]
    else:
        raise ValueError("Invalid split direction")





'''
def merging_superpixel_with_rag(image_rgb, sp_map, pct=10):
    
    
    # 1) Global split
    th = filters.threshold_otsu(image_rgb)
    is_shadow = image_rgb < th  # True=shadow, False=bright

    # 2) Build RAG with initial mean‐color distances
    rag = graph.rag_mean_color(
        color.gray2rgb(image_rgb),
        sp_map,
        mode='distance'
    )

    # 3) Zero‐out (or ∞‐out) any cross‐group edges
    for u, v, d in rag.edges(data=True):
        su = is_shadow[sp_map == u].any()
        sv = is_shadow[sp_map == v].any()
        if su != sv:
            d['weight'] = np.inf

    # 4) Collect only the *finite* distances
    weights = [d['weight'] for _, _, d in rag.edges(data=True) if np.isfinite(d['weight'])]
    if not weights:
        return sp_map

    # 5) Choose a positive threshold (pct‐th percentile of finite weights)
    thresh = np.percentile(weights, pct)
    print(f"Merging within each group")

    rag = rag_mean_color(
          color.gray2rgb(image_rgb),  # needs 3-channel input
          sp_map,
          mode='similarity'            # <-- similarity instead of 'distance'
    )

    # 3) Decide a merge threshold in similarity units (e.g. –0.2..0 → 0 means identical)
    #    Since mode='similarity' produces negative distances, edges with weight closer to 0 are more similar.
    sim_thresh = -0.1

    # 4) Merge all edges whose similarity >= sim_thresh
    #    (i.e. weight >= sim_thresh)
    merged = cut_threshold(sp_map, rag, thresh=sim_thresh)
    #weights = [d['weight'] for _,_,d in rag.edges(data=True)]
    #print("Edge‐weight percentiles:")
    #for p in (1, 5, 10, 25, 50, 75):
    #    print(f"  {p:>2}th pct: {np.percentile(weights, p):.2f}")

    return mergedup to distance ≤ {thresh:.3f} (pct={pct})")

    # 6) Define weight_func returning a dict
    def weight_func(g, src, dst, n):
        w = g[src][dst]['weight']
        return {'weight': float(w)}

    merged = graph.cut_threshold(sp_map, rag, thresh=thresh)

    return merged
'''


all_train_image = []
all_train_labels = []
all_train_superpixels = []
all_test_image = []
all_test_prediction = []
all_test_superpixels = []
all_test_groundtruth = []







def flatten(nested):
    return [x for group in nested for x in group]


    
    
    

    





#SNIC method
def segment(imgname,numsuperpixels,compactness,doRGBtoLAB):
	#--------------------------------------------------------------
	# read image and change image shape from (h,w,c) to (c,h,w)
	#--------------------------------------------------------------


    img = Image.open(imgname).convert("L")  # ensure grayscale

	#Fast processing
    new_size = (img.width // 4, img.height // 4)
    img = img.resize(new_size, resample=Image.BILINEAR) 

    #If it is one channel, otherwise comment it out
    img = Image.merge("RGB", (img, img, img)) 

    img = np.asarray(img)
    #print(img.shape)

    dims = img.shape
    h,w,c = dims[0],dims[1],1
    if len(dims) > 1:
        c = dims[2]
        img = img.transpose(2,0,1)
	
	#--------------------------------------------------------------
	# Reshape image to a single dimensional vector
	#--------------------------------------------------------------
    img = img.reshape(-1).astype(np.double)
    labels = np.zeros((h,w), dtype = np.int32)
    numlabels = np.zeros(1,dtype = np.int32)
	#--------------------------------------------------------------
	# Prepare the pointers to pass to the C function
	#--------------------------------------------------------------
    ffibuilder = FFI()
    pinp = ffibuilder.cast("double*", ffibuilder.from_buffer(img))
    plabels = ffibuilder.cast("int*", ffibuilder.from_buffer(labels.reshape(-1)))
    pnumlabels = ffibuilder.cast("int*", ffibuilder.from_buffer(numlabels))

	
    start = timer()
    SNIC_main(pinp,w,h,c,numsuperpixels,compactness,doRGBtoLAB,plabels,pnumlabels)
    end = timer()

	#--------------------------------------------------------------
	# Collect labels
	#--------------------------------------------------------------


    return labels.reshape(h,w),numlabels[0]



#before_merge = '/home/swz45/Documents/Shiqilabelmodified/cropped_image/superpixel_beforemerge'
#after_merge = '/home/swz45/Documents/Shiqilabelmodified/cropped_image/Test_image_after_merge'

#merged_sp_majority_voting_path = '/home/swz45/Documents/Shiqilabelmodified/cropped_image/merged_kmean/pspnet/merged'


all_prediction = []
all_ground_truth = []
results = []
highest_intensities = {}





if __name__ == "__main__":
        all_gt  = []
        all_pred = []
        all_refined = []

        models = os.getenv("SEgSP_MODEL")

        REPO_ROOT = Path(__file__).resolve().parents[3]

        dataset = "substrate"
        
        if models =="fcn8s": backbone = "vgg16" 
        elif models == "fcn32s": backbone = "vgg16"
        elif models == "deeplabv3": backbone = "resnet50"
        elif models == "psp": backbone = "resnet50"
        elif models == "denseaspp": backbone = "densenet121"

        exp_name = f"{models}_{backbone}_{dataset}"

        # outputs
        majority_voting_path = (REPO_ROOT / "superpixel_refinement" / exp_name).resolve()

        PRED_FOLDER = (REPO_ROOT / "runs" / exp_name).resolve()

        #print(PRED_FOLDER)

        # test data
        IMAGE_FOLDER = (REPO_ROOT / "test_images" / "label_removed_images").resolve()
        ground_truth_folder = (REPO_ROOT / "test_images" / "label_removed_labels").resolve()

        Test_filenames = sorted(f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(".png"))
        

        for count1, test_filename in enumerate(Test_filenames, 1):
            # Paths
            
            img_path = os.path.join(IMAGE_FOLDER, test_filename)
            gt_path = os.path.join(ground_truth_folder, test_filename)
           
            raw_pred_name = test_filename.replace('.png', '_raw.png')

            rawp_path = os.path.join(PRED_FOLDER, raw_pred_name)


            # 1) Load
            img  = Image.open(img_path).convert("L")
            gt = Image.open(gt_path).convert("L")
            rawp = Image.open(rawp_path).convert("L")

            

            # convert to numpy arrays
            img1  = np.array(img, dtype=np.uint8)
            gt1 = np.array(gt, dtype=np.uint8)
            rawp1 = np.array(rawp, dtype=np.uint8)

            test_h, test_w = rawp1.shape

            
            img2 = np.array(Image.fromarray(img1).resize((test_w//4, test_h//4), Image.BILINEAR))
            gt2 = cv2.resize(gt1, (test_w//4, test_h//4), interpolation=cv2.INTER_NEAREST)
            rawp2 = cv2.resize(rawp1, (test_w//4, test_h//4), interpolation=cv2.INTER_NEAREST)

            
            all_test_groundtruth.append(gt2)
            all_test_prediction.append(rawp2)


            sp_map, _ = segment(img_path, 3000, 1, doRGBtoLAB)
            
            

            # 4) Merge all-black SPs(at this stage: No)
            #sp_map2 = merge_all_black_superpixels(sp_map, sp_map, img2)
            #print(np.unique(sp_map2))

            all_test_superpixels.append(sp_map)

            #merged_superpixels = Merging_superpixels.merge_superpixels_by_adjacency_kmeans(img2, sp_map2)

            # The following is for glcm+region ad
            #merged_superpixels = glcm_merge.iterative_glcm_merge(img2, sp_map2)

            
            refined = doing_majority_voting.smooth_with_superpixel_mode(rawp2, sp_map)

            all_refined.append(refined)

            rgb = np.zeros((*refined.shape, 3), dtype=np.uint8)

            pred_mask = (rawp2 > 0)

            rgb[np.logical_and(pred_mask, refined == 1)] = [0, 255, 0]  # class 1 → green
            rgb[np.logical_and(pred_mask, refined == 2)] = [0,  0, 255]  # class 2 → blue
            rgb[np.logical_and(pred_mask, refined == 3)] = [0, 255, 255]
            rgb[np.logical_and(pred_mask, refined == 4)] = [255, 165, 0]

            # 5) Save
            os.makedirs(majority_voting_path, exist_ok=True)

           

            #Saving the_prediction with superpixel refinement with visualization

            Image.fromarray(rgb).save(os.path.join(majority_voting_path, test_filename))
        
            valid = (gt2 > 0) | (refined > 0)
            y_true = gt2[valid]
            y_pred = refined[valid]
            all_gt.append(y_true)
            all_pred.append(y_pred)
        
      


        gt_concat = np.concatenate(all_gt) 
        pred_concat = np.concatenate(all_pred) 
        # Define included classes (excluding background = 0) 
        labels_to_include = [1, 2, 3, 4] 
        # Build the confusion matrix 
        cm = confusion_matrix(gt_concat, pred_concat, labels=labels_to_include) 
        # Print confusion matrix 
        print("Confusion Matrix (Aggregated):") 
        print("Rows: Ground Truth, Columns: Prediction")
        # Compute metrics per class from confusion matrix 
        print(cm) 
        for i, label in enumerate(labels_to_include):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP

            precision = TP / (TP + FP + 1e-6)
            recall = TP / (TP + FN + 1e-6)

            # F1
            f1 = 2 * precision * recall / (precision + recall + 1e-6)

            # F2 (beta=2)
            f2 = 5 * precision * recall / (4 * precision + recall + 1e-6)

            print(f"Aggregate cls{label} → "
                f"P: {precision:.4f}, R: {recall:.4f}, "
                f"F1: {f1:.4f}, F2: {f2:.4f}")
        
        # ---- Overall Accuracy ----
        correct = np.trace(cm)     # sum of diagonal
        total   = cm.sum()
        accuracy = correct / (total + 1e-6)

        print(f"\nOverall Accuracy: {accuracy:.4f}")

