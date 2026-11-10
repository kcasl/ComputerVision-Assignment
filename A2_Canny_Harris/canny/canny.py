'''
    Prof. Seokju Lee @ EL4001
    Python code for Canny Edge Detector with VPGNet Dataset
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
mat = scipy.io.loadmat('20160809_1433_15/000001.mat')   # Try different images
img = mat['rgb_seg_vp'][:, :, 0: 3]
seg_gt = mat['rgb_seg_vp'][:, :, 3]


# Try to tune the parameters
img_b = cv2.GaussianBlur(img, (5, 5), 1.4)  # kernel size & sigma
img_e = cv2.Canny(img_b, 50, 150)           # lower & upper threshold 


plt.close('all')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
p1 = ax1.imshow(img); 
p2 = ax2.imshow(seg_gt); 
p3 = ax3.imshow(img_b); 
p4 = ax4.imshow(img_e);
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
