'''
    Prof. Seokju Lee @ EL4001

    Original code from:
    https://opencv24-python-tutorials.readthedocs.io/

    Prerequisite:
    which pip
    pip install opencv-contrib-python

    Visual application to understand camera parameters:
    If you want to study more, please try VirtualCam!
    https://github.com/kaustubh-sadekar/VirtualCam

    *Questions in the code are for self-understanding purposes, so answering them is optional.
'''

import numpy as np
import cv2 as cv
import glob, os
import pdb

np.set_printoptions(suppress=True)


################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS ################

chessboardSize = (10, 7)    # [Q] Please discuss the meaning of the paramter
frameSize = (960, 720)      # [Q] Please discuss the meaning of the paramter

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)    # [Q] Please discuss why we need this "criteria".

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 22
objp = objp * size_of_chessboard_squares_mm     # [Q] Please discuss the meaning of the paramter. Specify the unit of it.

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space  # [Q] How many points?
imgpoints = []  # 2d points in image plane.     # [Q] How many points?

images = glob.glob('./imgs_calib/*.jpg')        # [Q] How many images?
# pdb.set_trace()

for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)     # [Q] Please discuss the role of "findChessboardCorners()". Specify the meaning of its outputs.

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)
    # pdb.set_trace()

cv.destroyAllWindows()
# pdb.set_trace()



############################ CALIBRATION ############################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
# pdb.set_trace()
'''
    *retval: Average RMS re-projection error. This should be as close to zero as possible. 0.1 ~ 1.0 pixels in a good calibration.

    [Q] Please specify the focal length (fx, fy) and the principal point (cx, cy).

    [Q] Please specify the radial (k1, k2, k3) and tangential (p1, p2) distortion coefficients.

    *Tangential distortion occurs because the image-taking lens is not aligned perfectly parallel to the imaging plane.

    [Q] Please discuss the meaning of "rvecs" and "tvecs".

'''


############################ UNDISTORTION ############################
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, frameSize, 1, frameSize)
x, y, w, h = roi

### for calib samples ###
for image in images:
    print("saving undistorted calib sample: {}".format(image) )

    img = cv.imread(image)
    dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

    dst = dst[y:y+h, x:x+w]
    cv.imwrite('./imgs_calib/undist/' + os.path.basename(image), dst)
    # pdb.set_trace()

'''
### for SfM samples ###
for image in glob.glob('./imgs_sfm/*.jpg'):
    print("saving undistorted sfm sample: {}".format(image) )

    img = cv.imread(image)
    dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

    dst = dst[y:y+h, x:x+w]
    cv.imwrite('./imgs_sfm/undist/' + os.path.basename(image), dst)
    # pdb.set_trace()
'''

# Calculate reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )

'''
    [Q] What is the meaning of "Reprojection Error"? Please specify the unit of it.
    total error: 0.01946764118085742 (from 14 samples)

'''

print(cameraMatrix)
print(newCameraMatrix)
'''
    Original: 960 x 720
    [[699.3368201    0.         479.91420635]
     [  0.         698.95910137 363.49400524]
     [  0.           0.           1.        ]]

    After undistortion: 900 x 674
    [[707.08497205   0.         480.23402751]
     [  0.         704.13395517 363.45557819]
     [  0.           0.           1.        ]]
    
    (Pdb) dist
    array([[ 0.23901987, -1.09299926, -0.00007741,  0.00039854,  1.36042289]])

'''

pdb.set_trace()

