import os

import matplotlib
matplotlib.use("Agg")

import cv2
import numpy as np
import scipy.io
from matplotlib import pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
CANNY_ROOT = os.path.join(os.path.dirname(ROOT), "canny")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

DEFAULT = dict(blockSize=5, ksize=5, k=0.04)
DILATE = 3


def hide(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)
    print("saved", path)


def grid(images, name, figsize):
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    for ax, im in zip(axes, images):
        ax.imshow(im)
        hide(ax)
    fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(fig, name)


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def harris_response(gray, blockSize=5, ksize=5, k=0.04):
    """R = λ1 λ2 - k (λ1 + λ2)^2  from eigenvalues of M."""
    gray = np.float32(gray)
    ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    win = (blockSize, blockSize)
    ixx = cv2.boxFilter(ix * ix, -1, win)
    iyy = cv2.boxFilter(iy * iy, -1, win)
    ixy = cv2.boxFilter(ix * iy, -1, win)

    tr = ixx + iyy
    det = ixx * iyy - ixy * ixy
    disc = np.sqrt(np.maximum(tr * tr - 4.0 * det, 0.0))
    lam1 = 0.5 * (tr + disc)
    lam2 = 0.5 * (tr - disc)
    r = lam1 * lam2 - k * (lam1 + lam2) ** 2
    return r


def mark_corners(img, dst, rel=0.01):
    # same rule as harris.py
    out = img.copy()
    out[dst > rel * dst.max()] = [255, 0, 255]
    return out


def overlay(bgr, dst):
    shown = mark_corners(bgr, dst)
    raw = dst > 0.01 * dst.max()
    n_pix = int(raw.sum())
    dil_dst = cv2.dilate(dst, np.ones((3, 3), np.float32))
    n_peak = int(((np.abs(dst - dil_dst) < 1e-8) & raw).sum())
    vis = cv2.dilate(raw.astype(np.uint8), np.ones((DILATE, DILATE), np.uint8))
    shown[vis > 0] = (255, 0, 255)
    return to_rgb(shown), n_peak, n_pix


def run(bgr, gray, **p):
    dst = harris_response(gray, **p)
    return overlay(bgr, dst)


def main():
    bgr = cv2.imread(os.path.join(ROOT, "chess.jpg"))
    gray = np.float32(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    orig = to_rgb(bgr)
    h, w = gray.shape

    rgb_def, n_def, pix_def = run(bgr, gray, **DEFAULT)
    grid([orig, rgb_def], "q3_original_default.png", (10.4, 5.1))
    print("default", DEFAULT, "peaks", n_def, "pixels", pix_def)

    block_vals = [2, 5, 11]
    block_imgs, block_n = [], []
    for bs in block_vals:
        im, n, pix = run(bgr, gray, blockSize=bs, ksize=5, k=0.04)
        block_imgs.append(im)
        block_n.append((n, pix))
    grid(block_imgs, "q3_blocksize.png", (14.4, 4.85))
    print("blockSize peaks/pix", list(zip(block_vals, block_n)))

    ksize_vals = [1, 5, 7]
    ksize_imgs, ksize_n = [], []
    for ks in ksize_vals:
        im, n, pix = run(bgr, gray, blockSize=5, ksize=ks, k=0.04)
        ksize_imgs.append(im)
        ksize_n.append((n, pix))
    grid(ksize_imgs, "q3_ksize.png", (14.4, 4.85))
    print("ksize peaks/pix", list(zip(ksize_vals, ksize_n)))

    k_vals = [0.01, 0.04, 0.20]
    k_imgs, k_n = [], []
    for kk in k_vals:
        im, n, pix = run(bgr, gray, blockSize=5, ksize=5, k=kk)
        k_imgs.append(im)
        k_n.append((n, pix))
    grid(k_imgs, "q3_k.png", (14.4, 4.85))
    print("k peaks/pix", list(zip(k_vals, k_n)))

    rng = np.random.default_rng(0)
    noise_imgs, noise_n = [], []
    for sig in [0, 15, 40]:
        if sig == 0:
            g, noisy_bgr = gray, bgr
        else:
            g = np.clip(gray + rng.normal(0, sig, gray.shape), 0, 255).astype(np.float32)
            noisy_bgr = cv2.cvtColor(np.uint8(g), cv2.COLOR_GRAY2BGR)
        im, n, pix = run(noisy_bgr, g, **DEFAULT)
        noise_imgs.append(im)
        noise_n.append((n, pix))
    grid(noise_imgs, "q4_noise.png", (14.4, 4.85))
    print("noise peaks/pix", list(zip([0, 15, 40], noise_n)))

    scale_imgs, scale_n = [], []
    for s in [1.0, 0.5, 0.25]:
        if s == 1.0:
            bb, g = bgr, gray
        else:
            bb = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            g = np.float32(cv2.cvtColor(bb, cv2.COLOR_BGR2GRAY))
        im, n, pix = run(bb, g, **DEFAULT)
        im_up = cv2.resize(im, (w, h), interpolation=cv2.INTER_NEAREST)
        scale_imgs.append(im_up)
        scale_n.append((n, pix))
    grid(scale_imgs, "q4_scale.png", (14.4, 4.85))
    print("scale peaks/pix", list(zip([1.0, 0.5, 0.25], scale_n)))

    mat = scipy.io.loadmat(os.path.join(CANNY_ROOT, "20160512_1341_43", "000001.mat"))
    rgb = mat["rgb_seg_vp"][:, :, 0:3]
    drive_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    drive_gray = np.float32(cv2.cvtColor(drive_bgr, cv2.COLOR_BGR2GRAY))
    drive_im, drive_n, drive_pix = run(drive_bgr, drive_gray, **DEFAULT)
    grid([rgb, drive_im], "q4_driving_compare.png", (10.4, 4.4))
    print("driving peaks/pix", drive_n, drive_pix)


if __name__ == "__main__":
    main()
