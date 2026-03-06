import cv2
import numpy as np

def create_row_mask(img):
    # Read the image in grayscale
    #img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    
    # Create an empty mask with the same shape as the image
    mask = np.zeros_like(img, dtype=np.uint8)
    
    # Iterate through each row
    for y in range(img.shape[0]):
        row = img[y, :]
        
        # Find indices of non-black pixels (assuming black is 0)
        non_black_indices = np.where(row > 0)[0]
        
        if len(non_black_indices) > 0:
            start, end = non_black_indices[0], non_black_indices[-1]
            
            # Set pixels between start and end to white (255)
            mask[y, start:end+1] = 255
    
    return mask
