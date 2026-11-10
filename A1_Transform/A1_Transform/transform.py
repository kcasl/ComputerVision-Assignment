'''
    Prof. Seokju Lee @ EL4001 (Assignment #01)
    Python code for 2D projective geometry & transformation
    
    ssh -i "path_to_id_container" -p **** -R6000:127.0.0.1:6000 -L6006:127.0.0.1:6006 work@127.0.0.1
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda init bash
    
    conda create -n py310 python=3.10 numpy=1.26
    conda activate py310
    pip install matplotlib
    pip install opencv-python-headless

'''

import numpy as np
import cv2


def apply_transformation(image, transformation_matrix):
    # dimension of the image
    rows, cols, channels = image.shape
    
    # check the size 3x3
    if transformation_matrix.shape != (3, 3):
        raise ValueError("Transformation matrix must be 3x3 in size.")
    
    # initialize output
    transformed_image = np.zeros_like(image)
    
    # meshgrid function - how? and why?
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # make homogeneous coords - check the shape
    ones = np.ones_like(x)
    homogeneous_coords = np.stack([x, y, ones], axis=-1).reshape(-1, 3)
    
    # transformation
    new_coords = homogeneous_coords @ transformation_matrix.T
    
    # rasterization
    new_x = new_coords[:, 0].astype(int)
    new_y = new_coords[:, 1].astype(int)
    
    # Copy the value of valid pixels
    valid_indices = (new_x >= 0) & (new_x < cols) & (new_y >= 0) & (new_y < rows)
    transformed_image[new_y[valid_indices], new_x[valid_indices]] = image.reshape(-1, channels)[valid_indices]
    
    return transformed_image


def apply_projective_transformation(image, transformation_matrix):
    # dimension of the image
    rows, cols, channels = image.shape
    
    # check the size 3x3
    if transformation_matrix.shape != (3, 3):
        raise ValueError("Transformation matrix must be 3x3 in size.")

    transformed_image = np.zeros_like(image)

    # p2= H p1
    x1, y1 = np.meshgrid(np.arange(cols), np.arange(rows))
    p1 = np.stack(
        [x1.ravel(), y1.ravel(), np.ones(rows * cols, dtype=np.float64)],
        axis=0,
    )

    H = transformation_matrix.astype(np.float64)
    p2 = H @ p1
    z2 = p2[2]
    valid_z = np.abs(z2) > 1e-8

    new_x = np.zeros(z2.shape[0], dtype=int)
    new_y = np.zeros(z2.shape[0], dtype=int)
    new_x[valid_z] = (p2[0, valid_z] / z2[valid_z]).astype(int)
    new_y[valid_z] = (p2[1, valid_z] / z2[valid_z]).astype(int)

    valid = valid_z & (new_x >= 0) & (new_x < cols) & (new_y >= 0) & (new_y < rows)
    transformed_image[new_y[valid], new_x[valid]] = image.reshape(-1, channels)[valid]
    
    return transformed_image


if __name__ == "__main__":
    import matplotlib; matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import pdb

    # input image path
    img_0 = cv2.cvtColor(cv2.imread('mona.jpg'), cv2.COLOR_BGR2RGB) / 255.0

    pi = np.pi
    # th in degree, translation in pixel
    th1 = 0; tx1 = 100; ty1 = 100;
    th2 = +10; tx2 = 0; ty2 = 0;

    th1_ = +10; tx1_ = 0; ty1_ = 0;
    th2_ = 0; tx2_ = 100; ty2_ = 100;


    # CASE 1: 1st (0->1) transformation matrix
    mat_0_1 = np.array([[np.cos(th1 * pi/180), -np.sin(th1 * pi/180), tx1],
                        [np.sin(th1 * pi/180), np.cos(th1 * pi/180), ty1],
                        [0, 0, 1]])
    img_1 = apply_transformation(img_0, mat_0_1)

    # CASE 1: 2nd (1->2) transformation matrix
    mat_1_2 = np.array([[np.cos(th2 * pi/180), -np.sin(th2 * pi/180), tx2],
                        [np.sin(th2 * pi/180), np.cos(th2 * pi/180), ty2],
                        [0, 0, 1]])
    img_2 = apply_transformation(img_1, mat_1_2)


    # CASE 2: 1st (0->1) transformation matrix
    mat_0_1_ = np.array([[np.cos(th1_ * pi/180), -np.sin(th1_ * pi/180), tx1_],
                         [np.sin(th1_ * pi/180), np.cos(th1_ * pi/180), ty1_],
                         [0, 0, 1]])
    img_1_ = apply_transformation(img_0, mat_0_1_)

    # CASE 2: 2nd (1->2) transformation matrix
    mat_1_2_ = np.array([[np.cos(th2_ * pi/180), -np.sin(th2_ * pi/180), tx2_],
                         [np.sin(th2_ * pi/180), np.cos(th2_ * pi/180), ty2_],
                         [0, 0, 1]])
    img_2_ = apply_transformation(img_1_, mat_1_2_)



    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    p1 = ax1.imshow(img_0);
    p2 = ax2.imshow(img_2);
    p3 = ax3.imshow(img_2_);
    p4 = ax4.imshow(np.abs(img_2 - img_2_).mean(axis=2), cmap='turbo', vmin=0, vmax=1);
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(p4, cax=cax)
    plt.tight_layout(); plt.ion(); plt.show()

    pdb.set_trace()
    '''
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(img_0); ax2.imshow(img_1);
        plt.tight_layout()
        plt.show()

    '''

    # save the transformed image
    cv2.imwrite('mona_tr.jpg', img_1)
