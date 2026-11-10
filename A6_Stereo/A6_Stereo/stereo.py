'''
    Prof. Seokju Lee @ EL4001
    Python code for stereo matching
    Dataset: https://vision.middlebury.edu/stereo
    
    pip install opencv-contrib-python
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('tkagg')
import pdb

# stereo matching with different window size options
def stereo_matching(left_image, right_image, window_size=5, max_disparity=64, metric='SSD'):
    # image size
    h, w = left_image.shape
    
    # initialize disparity
    disparity_map = np.zeros_like(left_image, dtype=np.float32)
    
    half_window = window_size // 2
    
    # generate the disparity map
    for y in range(half_window, h - half_window):
        print("metric: %s,  window size: %d,  scanline (y): %4d" %(metric, window_size, y))
        for x in range(half_window, w - half_window):
            best_offset = 0
            min_cost = float('inf')
            max_corr = -1.0  # For NCC
            
            # window from the left image
            left_patch = left_image[y-half_window:y+half_window+1, x-half_window:x+half_window+1]
            
            # search the right image
            for offset in range(max_disparity):
                if x - offset - half_window < 0:
                    continue;
                
                right_patch = right_image[y-half_window:y+half_window+1, x-offset-half_window:x-offset+half_window+1]
                
                # Compute cost based on the selected metric
                if metric == 'SSD':
                    diff = left_patch.astype(np.float64) - right_patch.astype(np.float64)
                    cost = np.sum(diff * diff)
                elif metric == 'SAD':
                    diff = left_patch.astype(np.float64) - right_patch.astype(np.float64)
                    cost = np.sum(np.abs(diff))
                elif metric == 'NCC':
                    left = left_patch.astype(np.float64)
                    right = right_patch.astype(np.float64)
                    left = left - np.mean(left)
                    right = right - np.mean(right)
                    num = np.sum(left * right)
                    den = np.sqrt(np.sum(left * left) * np.sum(right * right))
                    if den < 1e-8:
                        corr = -1.0
                    else:
                        corr = num / den

                    # Maximize NCC (instead of minimizing for SSD/SAD)
                    if corr > max_corr:
                        max_corr = np.copy(corr)
                        best_offset = np.copy(offset)
                    continue  # Skip to the next offset for NCC

                # update
                if cost < min_cost:
                    min_cost = np.copy(cost)
                    best_offset = np.copy(offset)
            
            disparity_map[y, x] = best_offset
    
    # visualization: normalize disparity map
    disparity_map = cv2.normalize(disparity_map, disparity_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_map = np.uint8(disparity_map)
    
    return disparity_map

# load images
left_image = cv2.imread('./cones/im2.png', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('./cones/im6.png', cv2.IMREAD_GRAYSCALE)

# downsample images
rfactor = 2
left_image = cv2.resize(left_image, (left_image.shape[1] // rfactor, left_image.shape[0] // rfactor))
right_image = cv2.resize(right_image, (right_image.shape[1] // rfactor, right_image.shape[0] // rfactor))


# different window sizes (list)
window_sizes = [3, 5, 7]
# window_sizes = [7]

# maximum disparity
max_disparity = 64

plt.figure(figsize=(15, 5))
plt.subplot(1, len(window_sizes)+1, 1)
plt.imshow(left_image, cmap='gray')
plt.title(f'Left Image')
plt.axis('off')

for i, window_size in enumerate(window_sizes):
    disparity_map = stereo_matching(left_image, right_image, window_size=window_size, max_disparity=max_disparity//rfactor, metric='NCC')
    plt.subplot(1, len(window_sizes)+1, i+2)
    plt.imshow(disparity_map, cmap='gray')
    plt.title(f'Window Size: {window_size}')
    plt.axis('off')

plt.tight_layout(); plt.ion(); plt.show()

pdb.set_trace()
