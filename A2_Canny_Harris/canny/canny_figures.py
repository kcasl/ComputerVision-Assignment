import os

import matplotlib
matplotlib.use("Agg")

import cv2
import scipy.io
from matplotlib import pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

DEFAULT = dict(ksize=5, sigma=1.4, low=50, high=150)

TUNED = {
    "20160503_1618_44": dict(ksize=5, sigma=1.4, low=40, high=120),
    "20160512_1341_43": dict(ksize=5, sigma=1.2, low=70, high=180),
    "20160512_1410_30": dict(ksize=7, sigma=1.6, low=35, high=110),
    "20160518_2023_14": dict(ksize=7, sigma=2.0, low=20, high=70),
    "20160808_1346_45": dict(ksize=5, sigma=1.6, low=35, high=105),
    "20160809_1433_15": dict(ksize=7, sigma=1.8, low=40, high=120),
}

VARIANTS = [
    DEFAULT,
    dict(ksize=3, sigma=0.8, low=30, high=90),
    dict(ksize=7, sigma=2.0, low=80, high=200),
]

Q1_TUNING = [
    "20160512_1341_43",
    "20160512_1410_30",
    "20160518_2023_14",
    "20160809_1433_15",
]

Q2_WEATHER = [
    "20160512_1341_43",
    "20160503_1618_44",
    "20160809_1433_15",
    "20160518_2023_14",
]


def hide(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print("saved", path)


def load_rgb(seq):
    mat = scipy.io.loadmat(os.path.join(ROOT, seq, "000001.mat"))
    return mat["rgb_seg_vp"][:, :, 0:3]


def run_canny(img, ksize, sigma, low, high):
    # same as canny.py
    img_b = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    img_e = cv2.Canny(img_b, low, high)
    return img_b, img_e


def main():
    loaded = {seq: load_rgb(seq) for seq in set(Q1_TUNING + Q2_WEATHER)}

    for seq in Q1_TUNING:
        img = loaded[seq]
        p = TUNED[seq]
        _, e_tuned = run_canny(img, **p)

        panels = [img]
        for vp in VARIANTS:
            _, e = run_canny(img, **vp)
            panels.append(e)
        panels.append(e_tuned)

        fig, axes = plt.subplots(1, len(panels), figsize=(16.5, 3.15))
        for ax, im in zip(axes, panels):
            ax.imshow(im, cmap=None if im.ndim == 3 else "gray")
            hide(ax)
        fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.01, right=0.99, top=0.99, bottom=0.01)
        save(fig, f"q1_{seq}_tuning.png")

    fig, axes = plt.subplots(4, 2, figsize=(7.2, 11.2))
    for i, seq in enumerate(Q2_WEATHER):
        img = loaded[seq]
        _, e = run_canny(img, **TUNED[seq])
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(e, cmap="gray")
        hide(axes[i, 0])
        hide(axes[i, 1])
    fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(fig, "q2_four_weather.png")


if __name__ == "__main__":
    main()
