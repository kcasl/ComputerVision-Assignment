'''
    Prof. Seokju Lee @ EL4001
    Python code for extracting edge points
'''

import numpy as np
import scipy.io
import cv2
from matplotlib import pyplot as plt
import pdb

'''
    Lane & Road Marking Dataset in VPGNet (ICCV'17) 
    #S: Sunny / #C: Cloudy / #R: Rainy / #N: Night
    20160503_1618_44: #C
    20160512_1341_43: #S
    20160512_1410_30: #S
    20160518_2023_14: #N
    20160808_1346_45: #C
    20160809_1433_15: #R
'''
mat = scipy.io.loadmat('../A2_Canny_Harris/canny/20160808_1346_45/000001.mat')   # Try different images
img = mat['rgb_seg_vp'][:, :, 0: 3]
seg_gt = mat['rgb_seg_vp'][:, :, 3]


# Try to tune the parameters
img_b = cv2.GaussianBlur(img, (5, 5), 1.4)  # kernel size & sigma
img_e = cv2.Canny(img_b, 50, 150)           # lower & upper threshold 

points = np.column_stack(np.where(img_e > 0))

plt.close('all')
plt.figure()
plt.imshow(img_e, cmap='gray')
plt.scatter(points[:, 1], points[:, 0], color='red', s=1)  # Points in (x, y) format
plt.title("Canny Edge Map with Extracted Points")
plt.xlim([0, img_e.shape[1]])
plt.ylim([img_e.shape[0], 0])  # Inverted y-axis for correct image coordinates
plt.tight_layout(); plt.ion(); plt.show()

# pdb.set_trace()
'''
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1,2)pi
    ax1.imshow(img); ax2.imshow(img);
    plt.tight_layout()
    plt.show()
'''