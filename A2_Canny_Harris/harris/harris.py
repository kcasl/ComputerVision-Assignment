'''
    Prof. Seokju Lee @ EL4001
    Python code for Harris Corner Detector
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt
import pdb

# read an image
img = cv2.imread('chess.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# convert to float32
gray = np.float32(gray)

# Harris Corner Detection
dst = cv2.cornerHarris(gray, blockSize=5, ksize=5, k=0.04)

# visualize with threshold
img[dst > 0.01 * dst.max()] = [255, 0, 255]


plt.close('all')
plt.imshow(img)
plt.tight_layout(); plt.ion(); plt.show()


pdb.set_trace()
'''
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(img); ax2.imshow(img);
    plt.tight_layout()
    plt.show()
'''

# save the image
cv2.imwrite('img.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
