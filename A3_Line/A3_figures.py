import importlib.util
import os
import pdb

import matplotlib
matplotlib.use("Agg")

import cv2
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from scipy.optimize import minimize

# originals contain pdb.set_trace(); disable it so their functions can run
pdb.set_trace = lambda *a, **k: None

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figures")
A2 = os.path.join(os.path.dirname(ROOT), "A2_Canny_Harris", "canny")
os.makedirs(FIG, exist_ok=True)

SETS = [
    ("20160503_1618_44", dict(ksize=5, sigma=1.4, low=40, high=120)),
    ("20160512_1341_43", dict(ksize=5, sigma=1.2, low=70, high=180)),
    ("20160512_1410_30", dict(ksize=7, sigma=1.6, low=35, high=110)),
    ("20160518_2023_14", dict(ksize=7, sigma=2.0, low=20, high=70)),
    ("20160808_1346_45", dict(ksize=5, sigma=1.6, low=35, high=105)),
    ("20160809_1433_15", dict(ksize=7, sigma=1.8, low=40, high=120)),
]
COLS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]


def load_py(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lf, rs, hf = load_py("1_line_fitting"), load_py("2_ransac"), load_py("3_hough")


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.06)
    plt.close(fig)
    print("saved", path)


def hide(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def load_rgb(seq):
    return scipy.io.loadmat(os.path.join(A2, seq, "000001.mat"))["rgb_seg_vp"][:, :, :3]


def canny(img, ksize, sigma, low, high):
    # 0_edge_points.py
    return cv2.Canny(cv2.GaussianBlur(img, (ksize, ksize), sigma), low, high)


def edge_xy(img, img_e, y0=0.52, y1=0.86, vert=0.55, max_pts=1800):
    ys, xs = np.where(img_e > 0)
    h = img_e.shape[0]
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(np.float32(g), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(np.float32(g), cv2.CV_32F, 0, 1, ksize=3)
    keep = (ys >= int(h * y0)) & (ys <= int(h * y1)) & (np.abs(gx[ys, xs]) >= vert * np.abs(gy[ys, xs]))
    xs, ys = xs[keep].astype(float), ys[keep].astype(float)
    if len(xs) > max_pts:
        idx = np.random.default_rng(0).choice(len(xs), max_pts, replace=False)
        xs, ys = xs[idx], ys[idx]
    return xs, ys


def draw_polar(ax, theta, rho, x0, x1, y0, y1, color):
    c, s = np.cos(theta), np.sin(theta)
    px, py, dx, dy = rho * c, rho * s, -s, c
    ts = []
    if abs(dx) > 1e-8:
        ts += [(x0 - px) / dx, (x1 - px) / dx]
    if abs(dy) > 1e-8:
        ts += [(y0 - py) / dy, (y1 - py) / dy]
    pts = []
    for t in ts:
        x, y = px + t * dx, py + t * dy
        if x0 - 1 <= x <= x1 + 1 and y0 - 1 <= y <= y1 + 1:
            pts.append((x, y))
    if len(pts) < 2:
        return
    best, dmax = None, -1
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2
            if d > dmax:
                dmax, best = d, (pts[i], pts[j])
    ax.plot(*zip(*best), color=color, lw=2.0)


def draw_mb(ax, m, b, x0, x1, y0, y1, color):
    pts = []
    for x in (x0, x1):
        y = m * x + b
        if y0 - 1 <= y <= y1 + 1:
            pts.append((x, y))
    if abs(m) > 1e-8:
        for y in (y0, y1):
            x = (y - b) / m
            if x0 - 1 <= x <= x1 + 1:
                pts.append((x, y))
    if len(pts) >= 2:
        ax.plot(*zip(*pts[:2]), color=color, lw=2.0)


def panel3(img, img_e, drawer, name):
    h, w = img_e.shape
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.15))
    axes[0].imshow(img)
    axes[1].imshow(img_e, cmap="gray")
    axes[2].imshow(img)
    for ax in axes:
        hide(ax)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
    drawer(axes[2], w, h)
    fig.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(fig, name)


def grid6(rows, name):
    fig, axes = plt.subplots(6, 2, figsize=(6.2, 8.4))
    for i, (img, img_e, drawer) in enumerate(rows):
        h, w = img_e.shape
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(img)
        for ax in axes[i]:
            hide(ax)
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
        drawer(axes[i, 1], w, h)
    fig.subplots_adjust(wspace=0.03, hspace=0.03, left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(fig, name)


def x_at_y(th, rh, y):
    c = np.cos(th)
    if abs(c) < 1e-6:
        return None
    return (rh - y * np.sin(th)) / c


def looks_like_lane(th, rh, side, w, h):
    if abs(abs(th) % np.pi - np.pi / 2) < 0.38:
        return False
    xb = x_at_y(th, rh, 0.80 * h)
    xt = x_at_y(th, rh, 0.56 * h)
    if xb is None or xt is None:
        return False
    if side == "L":
        return (0.00 * w < xb < 0.55 * w) and (xt >= xb - 80) and (0.10 * w < xt < 0.70 * w)
    return (0.45 * w < xb < 1.02 * w) and (xt <= xb + 80) and (0.30 * w < xt < 0.90 * w)


def lane_wedge(x, y, w, h, side):
    t = np.clip((y - 0.52 * h) / max(0.34 * h, 1.0), 0.0, 1.0)
    if side == "L":
        lo = 0.28 * w * (1 - t) + 0.02 * w * t
        hi = 0.52 * w * (1 - t) + 0.50 * w * t
    else:
        lo = 0.48 * w * (1 - t) + 0.50 * w * t
        hi = 0.72 * w * (1 - t) + 0.98 * w * t
    return (x >= lo) & (x <= hi)


def split_lr(x, y, w, h, fit):
    lines = []
    for side in ("L", "R"):
        m = lane_wedge(x, y, w, h, side)
        hit = fit(x[m], y[m], side, w, h)
        if hit is not None:
            lines.append(hit)
    return lines


def fit_polar(x, y, side, w, h):
    xx, yy = np.array(x, float), np.array(y, float)
    best, best_n = None, -1
    for _ in range(5):
        if len(xx) < 25:
            break
        th0 = 0.85 if side == "L" else -0.85
        rho0 = float(np.mean(xx) * np.cos(th0) + np.mean(yy) * np.sin(th0))
        for g in ([0.0, 0.0], [th0, rho0]):
            th, rh = minimize(lf.total_perpendicular_distance, g, args=(xx, yy)).x
            d = np.abs(xx * np.cos(th) + yy * np.sin(th) - rh)
            n = int((d < 8.0).sum())
            if looks_like_lane(th, rh, side, w, h) and n >= 20 and n > best_n:
                best, best_n = (th, rh), n
        th, rh = lf.fit_line_optimize(xx, yy)
        d = np.abs(xx * np.cos(th) + yy * np.sin(th) - rh)
        xx, yy = xx[d >= 8.0], yy[d >= 8.0]
    return best


def fit_ransac(x, y, side, w, h):
    if len(x) < 28:
        return None
    xx, yy = np.array(x, float).copy(), np.array(y, float).copy()
    for _ in range(6):
        if len(xx) < 28:
            return None
        m, b = rs.ransac(xx, yy, num_iterations=800, threshold=10.0, min_inliers=28)
        if m is None:
            return None
        d = np.abs(m * xx - yy + b) / np.sqrt(m ** 2 + 1)
        inl = d < 10.0
        if abs(m) < 0.22:
            xx, yy = xx[~inl], yy[~inl]
            continue
        return m, b
    return None


def fit_hough(x, y):
    lines, acc, rhos, thetas = hf.hough_line_transform(
        x, y, num_rho=220, num_theta=180, threshold=30,
        min_distance=32, min_angle=np.deg2rad(12),
    )
    scored = []
    for rho, theta in lines:
        if abs(abs(theta) % np.pi - np.pi / 2) < 0.28:
            continue
        ri, ti = np.argmin(np.abs(rhos - rho)), np.argmin(np.abs(thetas - theta))
        scored.append((acc[ri, ti], rho, theta))
    scored.sort(reverse=True)
    kept = []
    for group in ([s for s in scored if s[2] > 0], [s for s in scored if s[2] < 0]):
        if group:
            kept.append(group[0])
    for item in scored:
        if len(kept) >= 3:
            break
        if all(abs(item[1] - r) >= 40 or abs(item[2] - t) >= np.deg2rad(12) for _, r, t in kept):
            kept.append(item)
    return [(rho, theta) for _, rho, theta in kept], acc, rhos, thetas


def fig_ls():
    np.random.seed(7)
    x, y, m, b = lf.generate_random_points_near_line(50)
    th, rh = lf.fit_line_optimize(x, y)
    xs = np.linspace(x.min() - 0.3, x.max() + 0.3, 100)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(x, y, s=18, c="0.25", zorder=2)
    ax.plot(xs, m * xs + b, "b--", lw=1.6)
    ax.plot(xs, (rh - xs * np.cos(th)) / np.sin(th), "r-", lw=1.8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    save(fig, "q1_synthetic.png")
    print(f"LS  true ({m:.3f},{b:.3f})  polar ({th:.3f},{rh:.3f})")


def fig_ransac():
    np.random.seed(11)
    x, y, m, b = rs.generate_random_points_near_line(80, noise_level=0.3, outlier_ratio=0.25)
    mr, br = rs.ransac(x, y)
    ml, bl = rs.fit_line_model(x, y)
    xs = np.linspace(x.min() - 0.3, x.max() + 0.3, 100)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(x, y, s=18, c="0.25", zorder=2)
    ax.plot(xs, m * xs + b, "b--", lw=1.6)
    ax.plot(xs, mr * xs + br, "r-", lw=1.8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    save(fig, "q1_ransac_synthetic.png")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1))
    for ax, (mm, bb, c) in zip(axes, ((ml, bl, "#d95f02"), (mr, br, "#e41a1c"))):
        ax.scatter(x, y, s=16, c="0.25", zorder=2)
        ax.plot(xs, m * xs + b, "b--", lw=1.5)
        ax.plot(xs, mm * xs + bb, color=c, lw=1.8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.35)
    fig.tight_layout()
    save(fig, "q1_ransac_vs_ls.png")
    print(f"RANSAC true ({m:.3f},{b:.3f})  ransac ({mr:.3f},{br:.3f})  ls ({ml:.3f},{bl:.3f})")


def fig_hough():
    np.random.seed(15)
    x, y, m1, b1, m2, b2 = hf.generate_random_points_near_two_lines(200)
    lines, acc, rhos, thetas = hf.hough_line_transform(
        x, y, num_rho=360, num_theta=360, threshold=16, min_distance=10, min_angle=np.deg2rad(5),
    )
    xs = np.linspace(x.min() - 0.2, x.max() + 0.2, 100)
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    ax.scatter(x, y, s=14, c="0.25", zorder=2)
    ax.plot(xs, m1 * xs + b1, "b--", lw=1.5)
    ax.plot(xs, m2 * xs + b2, "g--", lw=1.5)
    for rho, theta in lines:
        if abs(np.sin(theta)) > 1e-8:
            ax.plot(xs, (rho - xs * np.cos(theta)) / np.sin(theta), "r-", lw=1.4, alpha=0.85)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    save(fig, "q1_hough_synthetic.png")
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    im = ax.imshow(acc, extent=[thetas[0], thetas[-1], rhos[0], rhos[-1]], origin="lower", aspect="auto", cmap="hot")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save(fig, "q1_hough_accumulator.png")
    print("Hough true", (round(m1, 3), round(b1, 3)), (round(m2, 3), round(b2, 3)), "n", len(lines))


def _hough_overlay(ax, x, y, m1, b1, m2, b2, lines):
    xs = np.linspace(x.min(), x.max(), 80)
    ax.scatter(x, y, s=6, c="0.25", zorder=2)
    ax.plot(xs, m1 * xs + b1, "b--", lw=1.0)
    ax.plot(xs, m2 * xs + b2, "g--", lw=1.0)
    for rho, theta in lines:
        if abs(np.sin(theta)) > 1e-8:
            ax.plot(xs, (rho - xs * np.cos(theta)) / np.sin(theta), "r-", lw=1.1, alpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_hough_params():
    base = dict(num_rho=180, num_theta=180, threshold=16, min_distance=10, min_angle=np.deg2rad(5))
    trials = [
        ("num_points", [50, 200, 400], "n"),
        ("num_rho", [40, 180, 360], "h"),
        ("num_theta", [30, 180, 360], "h"),
        ("threshold", [8, 16, 40], "h"),
        ("min_distance", [0.2, 0.5, 10], "h"),
        ("min_angle", [1, 5, 30], "h"),
    ]
    fig, axes = plt.subplots(6, 3, figsize=(7.6, 9.2))
    stats = []
    for r, (key, values, kind) in enumerate(trials):
        for c, val in enumerate(values):
            np.random.seed(15)
            npts = val if key == "num_points" else 200
            x, y, m1, b1, m2, b2 = hf.generate_random_points_near_two_lines(npts)
            kw = dict(base)
            if key == "min_angle":
                kw["min_angle"] = np.deg2rad(val)
            elif key != "num_points":
                kw[key] = val
            lines, acc, rhos, thetas = hf.hough_line_transform(x, y, **kw)
            _hough_overlay(axes[r, c], x, y, m1, b1, m2, b2, lines)
            stats.append((key, val, len(lines)))
    fig.subplots_adjust(wspace=0.04, hspace=0.06, left=0.02, right=0.98, top=0.99, bottom=0.01)
    save(fig, "q2_hough_params.png")
    for s in stats:
        print("param", s)


def scenes(kind):
    rows = []
    first_acc = None
    for i, (seq, p) in enumerate(SETS):
        img = load_rgb(seq)
        img_e = canny(img, **p)
        if kind == "ls":
            xs, ys = edge_xy(img, img_e, y0=0.54, y1=0.86, vert=0.70)
            lines = split_lr(xs, ys, img_e.shape[1], img_e.shape[0], fit_polar)

            def drawer(ax, w, h, lines=lines):
                for (th, rh), c in zip(lines, COLS):
                    draw_polar(ax, th, rh, 0, w - 1, 0.52 * h, 0.93 * h, c)

            panel3(img, img_e, drawer, f"q2_{seq}.png")
            shown = [(round(float(th), 3), round(float(rh), 1)) for th, rh in lines]
        elif kind == "ransac":
            xs, ys = edge_xy(img, img_e, y0=0.52, y1=0.86, vert=0.55)
            np.random.seed(100 + i)
            lines = split_lr(xs, ys, img_e.shape[1], img_e.shape[0], fit_ransac)

            def drawer(ax, w, h, lines=lines):
                for (m, b), c in zip(lines, COLS):
                    draw_mb(ax, m, b, 0, w - 1, 0.52 * h, 0.93 * h, c)

            panel3(img, img_e, drawer, f"q2_ransac_{seq}.png")
            shown = []
            for m, b in lines:
                d = np.abs(m * xs - ys + b) / np.sqrt(m ** 2 + 1)
                shown.append((round(float(m), 3), round(float(b), 1), int((d < 10.0).sum())))
        else:
            xs, ys = edge_xy(img, img_e, y0=0.50, y1=0.88, vert=0.45)
            lines, acc, rhos, thetas = fit_hough(xs, ys)
            if first_acc is None:
                first_acc = (acc, rhos, thetas)

            def drawer(ax, w, h, lines=lines):
                for (rho, theta), c in zip(lines, COLS):
                    draw_polar(ax, theta, rho, 0, w - 1, 0.48 * h, h - 1, c)

            panel3(img, img_e, drawer, f"q3_hough_{seq}.png")
            shown = []
            for rho, theta in lines:
                ri, ti = np.argmin(np.abs(rhos - rho)), np.argmin(np.abs(thetas - theta))
                shown.append((round(float(rho), 1), round(float(theta), 3), int(acc[ri, ti])))
        rows.append((img, img_e, drawer))
        print(kind, seq, shown)

    names = {"ls": "q2_all.png", "ransac": "q2_ransac_all.png", "hough": "q3_hough_all.png"}
    grid6(rows, names[kind])
    if first_acc is not None:
        acc, rhos, thetas = first_acc
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        im = ax.imshow(acc, extent=[thetas[0], thetas[-1], rhos[0], rhos[-1]], origin="lower", aspect="auto", cmap="hot")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        save(fig, "q3_hough_accumulator.png")


def fig_all_points_vs_selected():
    seq, p = SETS[4]
    img = load_rgb(seq)
    img_e = canny(img, **p)
    h, w = img_e.shape
    ys, xs = np.where(img_e > 0)
    th0, rh0 = lf.fit_line_optimize(xs.astype(float), ys.astype(float))
    xs2, ys2 = edge_xy(img, img_e, y0=0.54, y1=0.86, vert=0.70)
    lines = split_lr(xs2, ys2, w, h, fit_polar)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2))
    for ax in axes:
        ax.imshow(img)
        hide(ax)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
    draw_polar(axes[0], th0, rh0, 0, w - 1, 0, h - 1, COLS[0])
    for (th, rh), c in zip(lines, COLS):
        draw_polar(axes[1], th, rh, 0, w - 1, 0.52 * h, 0.93 * h, c)
    fig.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(fig, "q2_all_points_vs_selected.png")
    print("all-points polar", float(th0), float(rh0))


if __name__ == "__main__":
    fig_ls()
    fig_all_points_vs_selected()
    fig_ransac()
    fig_hough()
    fig_hough_params()
    scenes("ls")
    scenes("ransac")
    scenes("hough")
