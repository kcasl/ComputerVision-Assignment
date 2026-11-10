import os
import sys
import json

import matplotlib
matplotlib.use("Agg")

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from transform import apply_projective_transformation, apply_transformation

FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def savefig(fig, name):
    out = os.path.join(FIG_DIR, name)
    tmp = out + ".tmp.png"
    fig.savefig(tmp, dpi=160, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close(fig)
    os.replace(tmp, out)
    print("saved", out)
    return out


def hide_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def make_euclidean(th_deg, tx, ty):
    th = th_deg * np.pi / 180.0
    return np.array(
        [
            [np.cos(th), -np.sin(th), tx],
            [np.sin(th), np.cos(th), ty],
            [0.0, 0.0, 1.0],
        ]
    )


def overlay_marker(ax, xy, color, label, marker="o"):
    ax.scatter([xy[0]], [xy[1]], c=color, s=40, marker=marker, zorder=5, label=label)


def main():
    img_path = os.path.join(ROOT, "mona.jpg")
    img_0 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255.0
    rows, cols, _ = img_0.shape
    stats = {
        "image_shape_hwc": [int(rows), int(cols), 3],
        "origin_pixel": [0, 0],
        "center_pixel": [cols / 2.0, rows / 2.0],
    }

# Q1 meshgrid
    step = 80
    xs = np.arange(0, cols, step)
    ys = np.arange(0, rows, step)
    gx, gy = np.meshgrid(xs, ys)

    fig, ax = plt.subplots(figsize=(5.2, 7.2))
    ax.imshow(img_0)
    ax.plot(gx, gy, color="cyan", linewidth=0.7, alpha=0.85)
    ax.plot(gx.T, gy.T, color="cyan", linewidth=0.7, alpha=0.85)
    ax.scatter(gx, gy, s=8, c="yellow", zorder=4, edgecolors="k", linewidths=0.2)
    ax.scatter([0], [0], s=70, c="lime", marker="o", zorder=5, edgecolors="k", label="(x, y) = (0, 0)")
    ax.annotate(
        "(0, 0)",
        xy=(0, 0),
        xytext=(100, 140),
        textcoords="data",
        color="lime",
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="lime", lw=1.4),
    )
    ax.text(cols - 12, 18, f"(x, y) = ({cols - 1}, 0)", color="w", fontsize=8, ha="right", va="top")
    ax.text(12, rows - 18, f"(0, {rows - 1})", color="w", fontsize=8, va="bottom")
    ax.set_xlim(0, cols - 1)
    ax.set_ylim(rows - 1, 0)
    ax.set_xlabel("x  (column index from meshgrid)")
    ax.set_ylabel("y  (row index from meshgrid)")
    ax.set_title("Q1.  x, y = meshgrid(...)  forms the pixel lattice", pad=14)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    savefig(fig, "q1_meshgrid.png")

# Q2 CASE 1 / CASE 2
    th1, tx1, ty1 = 0, 100, 100
    th2, tx2, ty2 = +10, 0, 0
    th1_, tx1_, ty1_ = +10, 0, 0
    th2_, tx2_, ty2_ = 0, 100, 100

    mat_0_1 = make_euclidean(th1, tx1, ty1)
    mat_1_2 = make_euclidean(th2, tx2, ty2)
    mat_0_1_ = make_euclidean(th1_, tx1_, ty1_)
    mat_1_2_ = make_euclidean(th2_, tx2_, ty2_)

    img_1 = apply_transformation(img_0, mat_0_1)
    img_2 = apply_transformation(img_1, mat_1_2)
    img_1_ = apply_transformation(img_0, mat_0_1_)
    img_2_ = apply_transformation(img_1_, mat_1_2_)
    img_comp_case1 = apply_transformation(img_0, mat_1_2 @ mat_0_1)
    img_comp_case2 = apply_transformation(img_0, mat_1_2_ @ mat_0_1_)

    diff = np.abs(img_2 - img_2_).mean(axis=2)
    stats["case1_mean_abs_diff_vs_case2"] = float(diff.mean())
    stats["case1_max_abs_diff_vs_case2"] = float(diff.max())
    stats["mat_0_1"] = mat_0_1.tolist()
    stats["mat_1_2"] = mat_1_2.tolist()
    stats["mat_0_1_"] = mat_0_1_.tolist()
    stats["mat_1_2_"] = mat_1_2_.tolist()
    stats["case1_composite_mat_1_2_times_mat_0_1"] = (mat_1_2 @ mat_0_1).tolist()
    stats["case2_composite_mat_1_2__times_mat_0_1_"] = (mat_1_2_ @ mat_0_1_).tolist()

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6))
    axes[0, 0].imshow(img_0)
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(img_1)
    axes[0, 1].set_title("CASE 1-1  mat_0_1\ntranslation (100, 100)")
    axes[0, 2].imshow(img_2)
    axes[0, 2].set_title("CASE 1-2  then mat_1_2\nrotation $+10^\\circ$ about (0, 0)")
    axes[1, 0].imshow(img_1_)
    axes[1, 0].set_title("CASE 2-1  mat_0_1_\nrotation $+10^\\circ$ about (0, 0)")
    axes[1, 1].imshow(img_2_)
    axes[1, 1].set_title("CASE 2-2  then mat_1_2_\ntranslation (100, 100)")
    im = axes[1, 2].imshow(diff, cmap="turbo", vmin=0, vmax=1)
    axes[1, 2].set_title("|CASE 1 $-$ CASE 2|\norder changes the result")
    divider = make_axes_locatable(axes[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    for ax in axes.ravel():
        hide_ticks(ax)
    fig.suptitle("Q2. Translation then rotation  $\\neq$  rotation then translation")
    fig.tight_layout()
    savefig(fig, "q2_case1_case2.png")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    axes[0].imshow(img_comp_case1)
    axes[0].set_title("CASE 1 composed\nmat_1_2 @ mat_0_1")
    axes[1].imshow(img_comp_case2)
    axes[1].set_title("CASE 2 composed\nmat_1_2_ @ mat_0_1_")
    im = axes[2].imshow(np.abs(img_comp_case1 - img_comp_case2).mean(axis=2), cmap="turbo", vmin=0, vmax=1)
    axes[2].set_title("|composed CASE 1 $-$ CASE 2|")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        hide_ticks(ax)
    fig.suptitle("Q2. One-shot $H_2 H_1$ matches the two-step warp (order still matters)")
    fig.tight_layout()
    savefig(fig, "q2_composite_compare.png")

# Q3 rotation center
    th_rot = 25
    mat_rot_origin = make_euclidean(th_rot, 0, 0)
    img_rot_origin = apply_transformation(img_0, mat_rot_origin)

    cx, cy = cols / 2.0, rows / 2.0
    mat_img_cen = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    mat_rot_cen = make_euclidean(th_rot, 0, 0)
    mat_cen_img = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
    mat_rot_around_center = mat_cen_img @ mat_rot_cen @ mat_img_cen
    img_rot_center = apply_transformation(img_0, mat_rot_around_center)

    stats["q3_theta_deg"] = th_rot
    stats["mat_img_cen"] = mat_img_cen.tolist()
    stats["mat_rot_cen"] = mat_rot_cen.tolist()
    stats["mat_cen_img"] = mat_cen_img.tolist()
    stats["mat_rot_around_center"] = mat_rot_around_center.tolist()

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.6))
    axes[0].imshow(img_0)
    overlay_marker(axes[0], (12, 12), "lime", "origin (0, 0)")
    overlay_marker(axes[0], (cx, cy), "red", "image center", marker="x")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_title("Pivots on the original")
    axes[1].imshow(img_rot_origin)
    overlay_marker(axes[1], (12, 12), "lime", "pivot")
    axes[1].set_title(f"Rotate ${th_rot}^\\circ$ about (0, 0)\nface leaves the center")
    axes[2].imshow(img_rot_center)
    overlay_marker(axes[2], (cx, cy), "red", "pivot", marker="x")
    axes[2].set_title(f"Rotate ${th_rot}^\\circ$ about the center\nmat_cen_img @ mat_rot_cen @ mat_img_cen")
    for ax in axes:
        ax.set_xlim(0, cols - 1)
        ax.set_ylim(rows - 1, 0)
        hide_ticks(ax)
    fig.suptitle("Q3. Default rotation is about the top-left origin, not the image center")
    fig.tight_layout()
    savefig(fig, "q3_rotation_center.png")

# Q4 projective + parallel lines
    canvas = np.clip(img_0 * 0.55 + 0.45, 0, 1)
    y1, y2 = int(rows * 0.32), int(rows * 0.68)
    x_left, x_right = int(cols * 0.12), int(cols * 0.88)
    line_img = canvas.copy()
    cv2.line(line_img, (x_left, y1), (x_right, y1), (0.85, 0.1, 0.1), 6)
    cv2.line(line_img, (x_left, y2), (x_right, y2), (0.1, 0.2, 0.9), 6)
    cv2.circle(line_img, (x_left, y1), 8, (0.85, 0.1, 0.1), -1)
    cv2.circle(line_img, (x_right, y1), 8, (0.85, 0.1, 0.1), -1)
    cv2.circle(line_img, (x_left, y2), 8, (0.1, 0.2, 0.9), -1)
    cv2.circle(line_img, (x_right, y2), 8, (0.1, 0.2, 0.9), -1)


    src_quad = np.array(
        [[0.0, 0.0], [cols - 1.0, 0.0], [cols - 1.0, rows - 1.0], [0.0, rows - 1.0]],
        dtype=np.float32,
    )
    dst_quad = np.array(
        [
            [cols * 0.28, rows * 0.22],
            [cols * 0.90, rows * 0.06],
            [cols * 0.78, rows * 0.92],
            [cols * 0.06, rows * 0.78],
        ],
        dtype=np.float32,
    )
    mat_proj = cv2.getPerspectiveTransform(src_quad, dst_quad)
    img_proj = apply_projective_transformation(line_img, mat_proj)

    def warp_point(H, xy):
        p = np.array([xy[0], xy[1], 1.0])
        q = H @ p
        return np.array([q[0] / q[2], q[1] / q[2]])

    p11, p12 = warp_point(mat_proj, (x_left, y1)), warp_point(mat_proj, (x_right, y1))
    p21, p22 = warp_point(mat_proj, (x_left, y2)), warp_point(mat_proj, (x_right, y2))

    def slope(a, b):
        dx = b[0] - a[0]
        if abs(dx) < 1e-9:
            return float("inf")
        return float((b[1] - a[1]) / dx)

    s_before_1 = slope((x_left, y1), (x_right, y1))
    s_before_2 = slope((x_left, y2), (x_right, y2))
    s_after_1 = slope(p11, p12)
    s_after_2 = slope(p21, p22)
    stats["parallel_before_slopes"] = [s_before_1, s_before_2]
    stats["parallel_after_slopes"] = [s_after_1, s_after_2]
    stats["mat_proj"] = mat_proj.tolist()
    stats["warped_line1"] = [p11.tolist(), p12.tolist()]
    stats["warped_line2"] = [p21.tolist(), p22.tolist()]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 5.0))
    axes[0].imshow(line_img)
    axes[0].set_title("Before: two parallel lines\n(both slopes $= 0$)")
    axes[1].imshow(np.clip(img_proj, 0, 1))
    axes[1].plot([p11[0], p12[0]], [p11[1], p12[1]], color=(0.85, 0.1, 0.1), lw=3.0, zorder=5)
    axes[1].plot([p21[0], p22[0]], [p21[1], p22[1]], color=(0.1, 0.2, 0.9), lw=3.0, zorder=5)
    axes[1].scatter([p11[0], p12[0]], [p11[1], p12[1]], c=[(0.85, 0.1, 0.1)], s=28, zorder=6)
    axes[1].scatter([p21[0], p22[0]], [p21[1], p22[1]], c=[(0.1, 0.2, 0.9)], s=28, zorder=6)
    axes[1].set_title(
        f"After homography  "
        r"$\tilde{p}_2 = H\tilde{p}_1$"
        f"\nslope $= {s_after_1:.3f}$ vs ${s_after_2:.3f}$  (not parallel)"
    )
    for ax in axes:
        ax.set_xlim(0, cols - 1)
        ax.set_ylim(rows - 1, 0)
        hide_ticks(ax)
    fig.suptitle("Q4. Parallel lines are not preserved under a projective transform")
    fig.tight_layout()
    savefig(fig, "q4_projective_parallel_lines.png")

    # extra: vanishing / grid
    grid = np.full((rows, cols, 3), 0.95, dtype=np.float64)
    for yy in range(40, rows, 50):
        cv2.line(grid, (20, yy), (cols - 20, yy), (0.15, 0.15, 0.15), 2)
    for xx in range(40, cols, 50):
        cv2.line(grid, (xx, 20), (xx, rows - 20), (0.15, 0.15, 0.15), 2)
    grid_proj = apply_projective_transformation(grid, mat_proj)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 5.0))
    axes[0].imshow(grid)
    axes[0].set_title("Before: parallel grid")
    axes[1].imshow(np.clip(grid_proj, 0, 1))
    axes[1].set_title("After: lines meet at a vanishing point")
    for ax in axes:
        hide_ticks(ax)
    fig.suptitle("Q4. Same homography on a full grid")
    fig.tight_layout()
    savefig(fig, "q4_projective_grid.png")

    stats_path = os.path.join(FIG_DIR, "experiment_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("stats", stats_path)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    sys.exit(main())
