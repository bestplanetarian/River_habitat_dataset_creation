import os
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import shutil
from collections import defaultdict

# --- CONFIG ---
PATCH_SIZE = 1024
TARGET_PIXELS = 1024 * 1024  # Minimum required pixels
EXCLUDE_CLASSES = [5, 8]      # Classes to remove
TEST_RATIO = 0.2
#VAL_RATIO = 0.15

Image.MAX_IMAGE_PIXELS = None



def get_classes(label_file):
    """
    Extracts unique class values from a grayscale label image.
    Assumes label image is single-channel and pixel values represent class IDs.
    """
    label_img = Image.open(label_file).convert('L')  # Ensure grayscale
    label_array = np.array(label_img)
    return set(np.unique(label_array))

def process_dataset(root_dir, output_dir):
    """Full processing pipeline"""
    # 1. Load pairs and filter by size
    pairs = []

    image_dir = '/home/swz45/Documents/Shadow_correction/Scaling_method/1x1/original_size_image'

    mask_dir = '/home/swz45/Documents/Shadow_correction/Scaling_method/1x1/original_size_mask'

    new_mask_dir = '/home/swz45/Documents/Shadow_correction/Scaling_method/1x1/changed_mask'

    #for img_path, mask_path in sorted(os.listdir(image_dir)), sorted(os.listdir(mask_dir)):
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        #if img.width * img.height >= TARGET_PIXELS:
        pairs.append((img_path, mask_path))
    
    # 2. Filter classes and remap IDs
    filtered_pairs = []

    
    for img_path, mask_path in pairs:
        mask = np.array(Image.open(mask_path))
    
    # Create remapping dictionary
        remap_dict = {
            0: 0,
            1: 1,
            2: 2,
            3: 1,  # Original class 2 becomes 1
            4: 0,
            5: 0,
            6: 1,
            7: 1,
            8: 0  #Simply removing the 
        }
    # All classes in [1,3,4,5,6,7,8] become 2
        #for cls in [1,3,6,7]:
        #    remap_dict[cls] = 1
        
        #new_mask = np.zeros_like(mask)
    # Apply remapping
        new_mask = np.zeros_like(mask)
        for old_id, new_id in remap_dict.items():
            new_mask[mask == old_id] = new_id    
        # Apply remapping
        new_mask_path = os.path.join(new_mask_dir, mask_path.split('/')[-1])
        #filtered_mask_path = Path(mask_path).parent / f"{Path(mask_path).stem}.png"
        Image.fromarray(new_mask).save(new_mask_path)
        filtered_mask_path = new_mask_path
        filtered_pairs.append((img_path, filtered_mask_path))
    
    # 3. Create balanced splits
    train_pairs, test_pairs = create_splits(filtered_pairs)

    out_train = os.path.join(output_dir, 'train')
    #out_val = os.path.join(output_dir, 'val')
    out_test = os.path.join(output_dir, 'test')

    print(out_train)
    
    # 4. Generate patches
    generate_patches(train_pairs, out_train, False)
    #generate_patches(val_pairs, out_val, False)
    generate_patches(test_pairs, out_test, True)

def create_splits(pairs):
    """Stratified splitting by class presence"""
    seed = 42
    random.seed(seed)
    #val_ratio = 0.1
    test_ratio = 0.2
    
   
    #remaining_indices = list(set(range(len(pairs))) - required_indices)
    #random.shuffle(remaining_indices)

    random.shuffle(pairs)
    total = len(pairs)
    #print(total)
    #num_val = int(val_ratio * total)
    num_test = int(test_ratio * total)
    num_train = int(0.8 * total)

    train_set = pairs[:num_train]
    test_set = pairs[num_train:]
    
    return train_set, test_set








def generate_patches(pairs, output_dir, mark):
    """Generate non-overlapping patches"""
    #(output_dir/"images").mkdir(parents=True, exist_ok=True)
    #(output_dir/"masks").mkdir(exist_ok=True)
    
    #images_dir = os.path.join(output_dir, 'images')
    #masks_dir = os.path.join(output_dir, 'masks')



    for img_path, mask_path in pairs:
        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))

        img_name = img_path.split('.')[0].split('/')[-1]
        msk_name = mask_path.split('.')[0].split('/')[-1]


        #if mark == True:
        #    save_patch(img, mask, img_name, 0, 0, output_dir)
        #    continue
    
        if img.shape[0] * img.shape[1] < TARGET_PIXELS:
           save_patch(img, mask, img_name, 0, 0, output_dir)
        else:

           pad_y = (PATCH_SIZE - img.shape[0] % PATCH_SIZE) % PATCH_SIZE
           pad_x = (PATCH_SIZE - img.shape[1] % PATCH_SIZE) % PATCH_SIZE
        
        # Apply padding (mirror for image, zeros for mask)
        if pad_y > 0 or pad_x > 0:
           img = np.pad(img, 
                        ((0, pad_y), (0, pad_x)), 
                        mode='constant', constant_values=0)  # Mirror edge pixels
           mask = np.pad(mask, 
                         ((0, pad_y), (0, pad_x)), 
                         mode='constant', constant_values=0)  # Pad with 0

        #Doing overlapping in the training images, but do not do it in the test image
        if mark == False:
           stride = PATCH_SIZE // 2
        else:
           stride = PATCH_SIZE

        # Generate patches with 512-stride (50% overlap)
        for y in range(0, img.shape[0] - PATCH_SIZE + 1, stride):
            for x in range(0, img.shape[1] - PATCH_SIZE + 1, stride):
                    patch_img = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                    patch_mask = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

                    non_zero_pixels = np.count_nonzero(patch_mask)
                    total_pixels = PATCH_SIZE * PATCH_SIZE
                    non_zero_percentage = (non_zero_pixels / total_pixels) * 100
                    print(non_zero_percentage)
                
                # Only save patches with meaningful content
                    if non_zero_percentage > 5:  # At least some non-background
                       save_patch(patch_img, patch_mask, 
                              img_name, x, y, output_dir)

def save_patch(img, mask, base_name, x, y, output_dir):
    """Save patches with standardized naming"""
    patch_id = f"{base_name}_x{x}_y{y}.png"

    images_dir = os.path.join(output_dir, 'images')
    masks_dir = os.path.join(output_dir, 'masks')
        
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    Image.fromarray(img).save(images_dir + '/' + patch_id)
    Image.fromarray(mask).save(masks_dir + '/' + patch_id)

# Run processing
process_dataset("/path/to/orthomosaics", "/home/swz45/Documents/Shadow_correction/Scaling_method/1x1")
