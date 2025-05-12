import cv2
import numpy as np
import os
from typing import List, Tuple
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def read_image_and_label(image_path: str, label_path: str):
    image = np.array(Image.open(image_path).convert("L"))
    label = np.array(Image.open(label_path).convert("L"))
    return image, label

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

def find_best_tile(label_tiles, full_classes):
    best_tile_idx = -1
    max_class_count = 0
    print(len(label_tiles))
    for i, tile_label in enumerate(label_tiles):
        tile_classes = set(np.unique(tile_label))
        if tile_classes == full_classes:
            return i  # Perfect match
        elif len(tile_classes) > max_class_count:
            best_tile_idx = i
            max_class_count = len(tile_classes)
    return best_tile_idx

def crop_image_and_label(image, label, crop_size, stride, save_dir, prefix):
    h, w = image.shape[:2]
    os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'labels'), exist_ok=True)
    idx = 0
    min_foreground_ratio = 0.05  # 5% non-black pixels

    def is_valid_patch(lbl_patch: np.ndarray) -> bool:
        non_black_ratio = np.count_nonzero(lbl_patch) / lbl_patch.size
        return non_black_ratio >= min_foreground_ratio

    for y in range(0, h - crop_size + 1, stride):
        for x in range(0, w - crop_size + 1, stride):
            img_patch = image[y:y+crop_size, x:x+crop_size]
            lbl_patch = label[y:y+crop_size, x:x+crop_size]
            if is_valid_patch(lbl_patch):
                cv2.imwrite(os.path.join(save_dir, 'images', f"{prefix}_{idx}.png"), img_patch)
                cv2.imwrite(os.path.join(save_dir, 'labels', f"{prefix}_{idx}.png"), lbl_patch)
                idx += 1

    if h % crop_size != 0:
        for x in range(0, w - crop_size + 1, stride):
            img_patch = image[h-crop_size:h, x:x+crop_size]
            lbl_patch = label[h-crop_size:h, x:x+crop_size]
            if is_valid_patch(lbl_patch):
                cv2.imwrite(os.path.join(save_dir, 'images', f"{prefix}_{idx}.png"), img_patch)
                cv2.imwrite(os.path.join(save_dir, 'labels', f"{prefix}_{idx}.png"), lbl_patch)
                idx += 1

    if w % crop_size != 0:
        for y in range(0, h - crop_size + 1, stride):
            img_patch = image[y:y+crop_size, w-crop_size:w]
            lbl_patch = label[y:y+crop_size, w-crop_size:w]
            if is_valid_patch(lbl_patch):
                cv2.imwrite(os.path.join(save_dir, 'images', f"{prefix}_{idx}.png"), img_patch)
                cv2.imwrite(os.path.join(save_dir, 'labels', f"{prefix}_{idx}.png"), lbl_patch)
                idx += 1

    if h % crop_size != 0 and w % crop_size != 0:
        img_patch = image[h-crop_size:h, w-crop_size:w]
        lbl_patch = label[h-crop_size:h, w-crop_size:w]
        if is_valid_patch(lbl_patch):
            cv2.imwrite(os.path.join(save_dir, 'images', f"{prefix}_{idx}.png"), img_patch)
            cv2.imwrite(os.path.join(save_dir, 'labels', f"{prefix}_{idx}.png"), lbl_patch)

def process_orthomosaic(image_path, label_path, crop_size=1024, stride=1024, output_dir='/home/swz45/Documents/Shadow_correction/Scaling_method/1x1/Output'):
    
    image, label = read_image_and_label(image_path, label_path)
    h, w = image.shape[:2]
    full_classes = set(np.unique(label))

    test_tile_idx = -1
    test_image = None
    test_label = None
    train_tiles_img = []
    train_tiles_lbl = []

    split_options = []
    if w < 10000 and h >= 10000:
        split_options.append(('vertical', 3))
    elif h < 10000 and w >= 10000:
        split_options.append(('horizontal', 3))
    elif h >= 10000 and w >= 10000:
        split_options.append(('all', 9))
    #elif w < 10000 and h < 10000:
        

    found = False

    for direction, _ in split_options:
        image_tiles = split_image(image, direction)
        label_tiles = split_image(label, direction)

        idx = find_best_tile(label_tiles, full_classes)
        if idx != -1:
            test_tile_idx = idx
            test_image = image_tiles[idx]
            test_label = label_tiles[idx]
            for i in range(len(image_tiles)):
                if i != idx:
                    train_tiles_img.append(image_tiles[i])
                    train_tiles_lbl.append(label_tiles[i])
            found = True
            break

        print(len(image_tiles))

    if not found:
        crop_image_and_label(image, label, crop_size, stride, os.path.join(output_dir, "train"), "train")
        print("No large test tile found. Used full image for training patches.")
        return

    #basename = os.path.splitext(os.path.basename(image_path))[0]

    basename = image_path.split('.')[0].split('/')[-1]
    os.makedirs(os.path.join(output_dir, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'labels'), exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, 'test', 'images', f"{basename}_test.png"), test_image)
    cv2.imwrite(os.path.join(output_dir, 'test', 'labels', f"{basename}_test.png"), test_label)

    for i, (train_img, train_lbl) in enumerate(zip(train_tiles_img, train_tiles_lbl)):
        crop_image_and_label(train_img, train_lbl, crop_size, stride, os.path.join(output_dir, "train"), f"{basename}_{i}")

    print("Dataset creation complete.")



img = '/home/swz45/Documents/Shadow_correction/Scaling_method/1x1/original_size_image'
msk = '/home/swz45/Documents/Shadow_correction/Scaling_method/1x1/original_size_mask'

# Example usage:
for item in os.listdir(img):
    img_name = img + '/' + item
    msk_n = item.split('.')[0] + '_binarymask.png'
    msk_name = msk + '/' + msk_n
    print(img_name, msk_name)
    process_orthomosaic(img_name, msk_name)