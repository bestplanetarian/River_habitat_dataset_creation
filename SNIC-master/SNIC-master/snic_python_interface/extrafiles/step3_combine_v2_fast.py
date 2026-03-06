import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
from skimage import segmentation, color
from skimage.segmentation import slic, mark_boundaries
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from SNICdemo import segment
from scipy.stats import mode
import cv2
from build_mask import create_row_mask
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score



# Constants
n_segments = 50
compactness = 40
similarity_threshold = 0.95




class_colors1 = {

    0: (0,0,0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (128, 0, 128),
    6: (0, 255, 255),
    7: (255, 165, 0)
}
    
'''
    0: (0,0,0),
    1: (255,0,0),
    2: (0,0,255),
    3: (128,0,128),
    4: (0,255,255),
    5: (255,165,0),
    6: (255,192,203)
'''


#color generation
previous_colors = set()




# Preprocess image for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load pre-trained ResNet50 model and modify it to extract features
model = resnet50(pretrained=True)
model.eval()
model.fc = nn.Identity()


def extract_features_batch(images):
    """Extract features for a batch of images using ResNet-50."""
    images_tensor = torch.stack([transform(image) for image in images])

    #print('Image tensor looks like:')
    #print(images_tensor)
    with torch.no_grad():
        features = model(images_tensor)
    return features.cpu().numpy()

def calculate_superpixel_mean(resized_mask, image):
    
    """
    Calculate the mean color (or pixel value) for each superpixel area in the image.
    
    Args:
        resized_mask (np.ndarray): 2D array where each unique value corresponds to a superpixel segment.
        image (np.ndarray): The original image (2D for grayscale, 3D for RGB).
    
    Returns:
        dict: A dictionary where keys are segment IDs and values are the mean color (or pixel value) of each segment.
    """
    segment_ids = np.unique(resized_mask)
    segment_means = {}

    for segment_id in segment_ids:
        # Create a mask for the current segment
        mask = (resized_mask == segment_id)

        # Calculate the mean value for this segment
        if image.ndim == 3:  # For RGB images
            segment_mean = np.mean(image[mask], axis=0)
        else:  # For grayscale imagess
            segment_mean = np.mean(image[mask])

        # Store the mean value in the dictionary
        segment_means[segment_id] = segment_mean

    return segment_means
   

def downsize_image(image_path):
    """Downsize the input image."""
    image = Image.open(image_path).convert('RGB')
    resized_image = image.resize((image.width // 1, image.height // 1))
    return resized_image, image


def apply_slic_segmentation(image_array):
    """Perform SLIC segmentation on the downsized image."""
    segments = segmentation.slic(image_array, n_segments=n_segments, compactness=compactness, start_label=1)

    image_with_boundaries = mark_boundaries(image_array, segments, color=(1, 0, 0))



    #plt.imshow(image_with_boundaries)
    #plt.axis('off')
    #plt.show()

    return segments


def majority_vote(prediction, superpixels):
    """
    Apply majority voting within superpixel clusters, ignoring background pixels.
    
    Args:
        prediction (np.ndarray): 2D array of predicted labels.
        superpixels (np.ndarray): 2D array of superpixel cluster indices.
    
    Returns:
        np.ndarray: Updated prediction after applying majority voting.
    """
    """
    Apply majority voting within superpixel clusters for a grayscale prediction, ignoring background pixels.
    
    Args:
        prediction (np.ndarray): 2D array (H, W) of grayscale labels.
        superpixels (np.ndarray): 2D array of superpixel cluster indices.

    Returns:
        np.ndarray: Updated grayscale prediction after applying majority voting.
    """
    """
    Apply majority voting within superpixel clusters for a grayscale prediction, ignoring background pixels.
    
    Args:
        prediction (np.ndarray): 2D array (H, W) of grayscale labels.
        superpixels (np.ndarray): 2D array of superpixel cluster indices.

    Returns:
        np.ndarray: Updated grayscale prediction after applying majority voting.
    """
    # Ensure shape consistency
    #print(prediction.shape)
    #print(superpixels.shape)


    assert prediction.shape == superpixels.shape, "Shape mismatch between prediction and superpixels"

    # Define background mask (assumes black pixels [0] are background)
    background_mask = (prediction == 0)

    # Initialize output prediction (copy of the original)
    updated_prediction = prediction.copy()

    # Get unique superpixel labels
    unique_clusters = np.unique(superpixels)

    for cluster_id in unique_clusters:
        # Get mask for the current cluster
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


def grayscale_to_color(prediction, class_colors):
    """
    Convert a grayscale segmentation mask to a colorful RGB image based on class mapping.

    Args:
        prediction (np.ndarray): 2D array of class labels.
        class_colors (dict): Mapping of class indices to RGB color tuples.

    Returns:
        np.ndarray: 3D RGB image (H, W, 3).
    """
    # Create an empty RGB image
    height, width = prediction.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Assign colors based on class mapping
    for class_id, color in class_colors.items():
        color_image[prediction == class_id] = color  # Assign RGB tuple

    return color_image






def resize_segmentation_to_original(slic_mask, original_size):
    """Resize the segmentation mask to match the original image size."""
    resized_mask = Image.fromarray(slic_mask.astype(np.uint8))
    resized_mask = resized_mask.resize(original_size, resample=Image.NEAREST)
    return np.array(resized_mask)


def compute_accuracy(prediction, ground_truth):
    """
    Compute accuracy by comparing the prediction to the ground truth,
    while ignoring background pixels (value = 0).

    Args:
        prediction (np.ndarray): 2D array of predicted labels.
        ground_truth (np.ndarray): 2D array of ground truth labels.

    Returns:
        float: Accuracy (correct predictions / total non-background pixels).
    """
    assert prediction.shape == ground_truth.shape, "Shape mismatch between prediction and ground truth"

    # Mask to ignore background pixels (value 0)
    non_background_mask = (ground_truth != 0)

    # Count correct predictions (where prediction matches ground truth)
    correct_predictions = np.sum((prediction == ground_truth) & non_background_mask)

    # Count total non-background pixels
    total_non_background_pixels = np.sum(non_background_mask)

    # Compute accuracy (Avoid division by zero)
    accuracy = correct_predictions / total_non_background_pixels if total_non_background_pixels > 0 else 0

    return accuracy

def extract_segments(original_image, resized_mask):
    """Extract all segments as individual images and return them as a list."""
    segments = []
    segment_ids = np.unique(resized_mask)
    segment_dict = {}
    
    for segment_id in segment_ids:
        mask = (resized_mask == segment_id)
        upscaled_mask = Image.fromarray(mask.astype(np.uint8) * 255)
        segment_area = Image.composite(original_image, Image.new('RGB', original_image.size), upscaled_mask)
        
        # Crop based on non-zero mask area
        segment_array = np.array(upscaled_mask)
        non_zero_coords = np.argwhere(segment_array > 0)

        if len(non_zero_coords) == 0:
            continue

        top_left = non_zero_coords.min(axis=0)
        bottom_right = non_zero_coords.max(axis=0)
        cropped_segment = segment_area.crop((top_left[1], top_left[0], bottom_right[1] + 1, bottom_right[0] + 1))

        #print('The cropped segments are:')

        #segment_arr = np.array(cropped_segment)

        #print(segment_arr)

        # Resize to 224x224 and check if the segment is mostly black
        cropped_segment = cropped_segment.resize((224, 224))
        segment_array = np.array(cropped_segment)
        if np.mean(segment_array == 0) > 0.5:
            continue

        segments.append(cropped_segment)
        segment_dict[segment_id] = cropped_segment

    return segments, segment_dict


def find_neighbors(segment_mask, segment_id):
    """Find all neighboring segments that share a border with the given segment_id."""
    mask = (segment_mask == segment_id)
    neighbors = set()
    
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                if i > 0 and segment_mask[i-1, j] != segment_id and not mask[i-1, j]:
                    neighbors.add(segment_mask[i-1, j])
                if i < mask.shape[0] - 1 and segment_mask[i+1, j] != segment_id and not mask[i+1, j]:
                    neighbors.add(segment_mask[i+1, j])
                if j > 0 and segment_mask[i, j-1] != segment_id and not mask[i, j-1]:
                    neighbors.add(segment_mask[i, j-1])
                if j < mask.shape[1] - 1 and segment_mask[i, j+1] != segment_id and not mask[i, j+1]:
                    neighbors.add(segment_mask[i, j+1])

    return list(neighbors)


def generate_random_colors():
    """Generate random RGB colors."""

    '''
    colors = {}
    for i in range(1, num_colors + 1):
        colors[i] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    return colors
    '''
    color = (random.randint(10, 240), random.randint(10, 240), random.randint(10, 240))

    if color not in previous_colors:
       previous_colors.add(color)
       return color

    


def apply_transparent_overlay(original_image, segmentation_mask_org, segmentation_mask, output_path='colored_mask_result.jpg'):
    """Apply a half-transparent color overlay based on the segmentation mask."""
    mask_array = np.array(segmentation_mask)
    original_image_array = np.array(original_image)

    unique_ids = np.unique(np.array(segmentation_mask_org))

    valid_ids = []

    valid_mask1 = create_row_mask(original_image_array)

    print(len(valid_mask1))

    if len(valid_mask1.shape) == 3:  # If it's (H, W, 3), convert to grayscale
        valid_mask1 = np.mean(valid_mask1, axis=-1).astype(np.uint8)
    
    valid_mask_binary = (valid_mask1 == 255) 


    for unique_id in unique_ids:
        segment_mask = segmentation_mask_org == unique_id
        '''
        if np.any(original_image_array[mask]) > 0:
            valid_ids.append(unique_id)
        '''
        if np.any(segment_mask & valid_mask_binary):
            valid_ids.append(unique_id)

    print("valid Unique id in here is:")
    print(valid_ids)


    #mask = create_row_mask(original_image)


    #colors = generate_random_colors(len(valid_ids))

    colors = {valid_id: generate_random_colors() for valid_id in valid_ids}
    
    modified_color_image = original_image_array.copy()
    for value, color in colors.items():
        mask_area = (mask_array == value)
        modified_color_image[mask_area] = (0.5 * modified_color_image[mask_area] + 0.5 * np.array(color)).astype(int)

        

    modified_color_image = Image.fromarray(modified_color_image)
    modified_color_image.save(output_path)
    print(f"Half-transparent colored mask saved as '{output_path}'")

'''
def compute_eucludean_distance(points):
    threshold = 200
''' 


def process_image(image_path, pred_path, ground_truth_path, output_dir, similarity_threshold=0.75, class_colors=class_colors1, label_list=None):

    downsized_image, original_image = downsize_image(image_path)
    downsized_array = np.array(downsized_image)

    slic_downsize = apply_slic_segmentation(downsized_array)
    slic_original = resize_segmentation_to_original(slic_downsize, original_image.size)

    # ------------------------------------------------
    # 3) Extract segments and batch features
    # ------------------------------------------------
    segments, segment_dict = extract_segments(original_image, slic_original)
    if not segments:
        print(f"[Warning] No valid segments found for {image_path}.")
        return

    features_batch = extract_features_batch(segments)
    features_dict = {segment_id: features_batch[i] for i, segment_id in enumerate(segment_dict)}

    # ------------------------------------------------
    # 4) Merge segments based on cosine similarity
    # ------------------------------------------------
    slic_downsize_copy = np.copy(slic_downsize)

    
    '''
    #Here, we iteratively merge the superpixel segments
    iteration = 3

    for i in range(iteration):
        unique_segments = np.unique(slic_downsize_copy)  # Refresh unique segments at the start of each iteration
    
        for segment_id in unique_segments:
            if segment_id not in features_dict:
               continue
        
            segment_feature = features_dict[segment_id]
            neighbors = find_neighbors(slic_downsize_copy, segment_id)

            for neighbor_id in neighbors:
                if neighbor_id not in features_dict:
                   continue

                neighbor_feature = features_dict[neighbor_id]
                similarity = cosine_similarity([segment_feature], [neighbor_feature])[0, 0]

                if similarity >= similarity_threshold:
                # Merge neighbor_id into segment_id
                   slic_downsize_copy[slic_downsize_copy == neighbor_id] = segment_id
                   print(f"Merged {neighbor_id} into {segment_id}")

    # Step 5: Apply half-transparent color mask to the downsized image
        num_superpixels = len(np.unique(slic_downsize_copy))
        print("Number of superpixels in slic_downsize_copy:", num_superpixels)

# Update the original segmentation after all iterations
    slic_downsize = np.copy(slic_downsize_copy)
    '''
    if not os.path.isfile(pred_path):
        print(f"[Warning] Prediction file not found: {pred_path} — skipping.")
        return

    prediction_pil = Image.open(pred_path).convert("L")
    prediction = np.array(prediction_pil)

    modified_prediction = majority_vote(prediction, slic_downsize_copy)

    # ------------------------------------------------
    # 6) Save the updated grayscale prediction
    # ------------------------------------------------
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, base_name + "_slic.png")
    cv2.imwrite(output_path, modified_prediction)
    print(f"Updated grayscale prediction saved at {output_path}")

    # ------------------------------------------------
    # 7) Colorize & Save
    # ------------------------------------------------
    if class_colors is None:
        # Example color map for classes 1..6 if not provided
        class_colors = {
            
        }
    colorized_prediction = grayscale_to_color(modified_prediction, class_colors)
    output_color_path = os.path.join(output_dir, base_name + "_sliccolor.png")
    cv2.imwrite(output_color_path, cv2.cvtColor(colorized_prediction, cv2.COLOR_RGB2BGR))
    print(f"Colorized prediction saved at {output_color_path}")

    # ------------------------------------------------
    # 8) Evaluate Accuracy + Confusion Matrix + F1
    # ------------------------------------------------
    if os.path.isfile(ground_truth_path):
        gt_pil = Image.open(ground_truth_path).convert("L")
        gt = np.array(gt_pil)

        # 8.1) Overall Accuracy (if you have a custom function)
        acc = compute_accuracy(modified_prediction, gt)
        print(f"Accuracy (majority voting) for {image_path}: {acc:.4f}")

        # 8.2) Confusion Matrix and F1-Scores
        # Flatten arrays so that each pixel is treated as a single sample
        flat_pred = modified_prediction.flatten()
        flat_gt = gt.flatten()

        overlay_path = os.path.join(output_dir, base_name + "_overlay.jpg")
        #apply_transparent_overlay(downsized_image, slic_downsize, slic_downsize_copy, overlay_path)
        #print(f"Overlay saved at {overlay_path}")

        #print(f"[Done] Processed {image_path}")

        return flat_pred, flat_gt

        '''

        # If you have 6 classes: [1,2,3,4,5,6]
        if label_list is None:
            label_list = [1,2,3,4,5,6]

        # Compute confusion matrix
        cm = confusion_matrix(flat_gt, flat_pred, labels=label_list)
        print("Confusion Matrix (rows=GT, cols=Pred):")
        print(cm)

        # Compute F1-score per class
        f1_per_class = f1_score(flat_gt, flat_pred, labels=label_list, average=None)
        print("F1 per class:")
        for lbl, f1_val in zip(label_list, f1_per_class):
            print(f"   Class {lbl}: {f1_val:.4f}")

        # Compute average F1-score (macro)
        avg_f1 = f1_score(flat_gt, flat_pred, labels=label_list, average='macro')
        print(f"Average F1 (macro): {avg_f1:.4f}")
        '''

    else:
        print(f"[Warning] Ground truth file not found: {ground_truth_path} — skipping accuracy.")
        return None, None

    # ------------------------------------------------
    # 9) Apply transparent overlay (optional)
    # ------------------------------------------------
   
    


    




#Handling the whole dataset, the whole folder:






def main():    

    input_dir = '/home/swz45/Documents/awesome-semantic-segmentation-pytorch/awesome-semantic-segmentation-pytorch-master/runs/pred_pic_deeplabv31024x1024_wholeset/deeplabv3_resnet50_substrate'

    img_dir = '/home/swz45/Documents/Shadow_correction/Model_train_test/superpixel_used_testdataset'

    output_dir = '/home/swz45/Documents/awesome-semantic-segmentation-pytorch/awesome-semantic-segmentation-pytorch-master/runs/Deeplabv3_1024x1024Nomanmade_Superpix'

    label_list = [1, 2, 3, 4, 5, 6, 7]
    
    # Prepare empty lists for aggregating predictions & ground truths
    all_preds = []
    all_gts = []

    for filename in os.listdir(input_dir):
        #print(filename)  

        if 'horizon_gt_msk' in filename:

            ground_truth_path = os.path.join(input_dir, filename)
            print(ground_truth_path)

        # 1) Build the prediction path by replacing the last two tokens with 'raw'
            parts = filename.split('_')
        # Replace the *last two items* in `parts` with a single item 'raw'
        # e.g. [..., 'gt', 'msk.png'] becomes [..., 'raw']
            parts[-2:] = ['raw']
            prediction_name = '_'.join(parts) + '.png'
            prediction_path = os.path.join(input_dir, prediction_name)

        # 2) Build the image path by *removing* the last two tokens
            img_parts = filename.split('_')
        # This removes the last two tokens from `img_parts`
        # e.g. [..., 'gt', 'msk.png'] is dropped entirely
            img_parts = img_parts[:-2]
            image_name = '_'.join(img_parts) + '.png'
            image_path = os.path.join(img_dir, image_name)

            print("Ground Truth:", ground_truth_path)
            print("Prediction:  ", prediction_path)
            print("Image:       ", image_path)
        
           
            pred, gt = process_image(image_path, prediction_path, ground_truth_path, output_dir)

            valid_msk = (gt != 0) & (pred != 0)

            all_preds.append(pred[valid_msk])
            all_gts.append(gt[valid_msk])
    
    all_gts = np.concatenate(all_gts)
    all_preds = np.concatenate(all_preds)

    cm = confusion_matrix(all_gts, all_preds, labels=label_list)
    print("Confusion Matrix (dataset-level):")
    print(cm)

    
    cm_percentage = cm.astype(np.float32) / cm.sum() * 100.0
    print("\nConfusion Matrix as Percent of Total:")
    print(cm_percentage)
    

    '''
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Normalize to percentages
    cm_percentage = np.round(cm_percentage, decimals=2)  
    print(cm_percentage)
    '''
    class_label = ['Bedrock', 'Shadow', 'Fine','Bank', 'Woody', 'Boulder', 'Rocky Fine']

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_percentage,
        annot=True,
        fmt=".2f",      # 2 decimal places
        cmap="Blues",
        xticklabels=class_label,
        yticklabels=class_label
    )
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix (%)")
    plt.tight_layout()
    plt.show()

    # Compute precision, recall, F1 for each class
    per_class_precision = precision_score(all_gts, all_preds, labels=label_list, average=None, zero_division=0)
    per_class_recall = recall_score(all_gts, all_preds, labels=label_list, average=None, zero_division=0)
    per_class_f1 = f1_score(all_gts, all_preds, labels=label_list, average=None, zero_division=0)
    
    # Print them per class
    print("\nPer-Class Precision:")
    for lbl, val in zip(label_list, per_class_precision):
        print(f" Class {lbl}: {val:.4f}")
    
    print("\nPer-Class Recall:")
    for lbl, val in zip(label_list, per_class_recall):
        print(f" Class {lbl}: {val:.4f}")
    
    print("\nPer-Class F1:")
    for lbl, val in zip(label_list, per_class_f1):
        print(f" Class {lbl}: {val:.4f}")
    
    # Also compute macro-average or weighted-average metrics
    macro_precision = precision_score(all_gts, all_preds, labels=label_list, average='macro', zero_division=0)
    macro_recall = recall_score(all_gts, all_preds, labels=label_list, average='macro', zero_division=0)
    macro_f1 = f1_score(all_gts, all_preds, labels=label_list, average='macro', zero_division=0)
    
    print(f"\nMacro Precision: {macro_precision:.4f}")
    print(f"Macro Recall:    {macro_recall:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
        

        
        
if __name__ == "__main__":
    main()







