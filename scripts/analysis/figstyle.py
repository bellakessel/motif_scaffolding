"""Shared figure styling + helpers for the designability analysis notebook.
Import once at the top of the notebook:  from figstyle import *"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# MotifBench designability gates (arXiv:2502.12479)
MOTIF_RMSD_MAX = 1.0
SC_RMSD_MAX = 2.0

# fixed, reused palettes
SSE_PALETTE = {"C": "#7570b3", "E": "#1b9e77", "H": "#d95f02"}   # coil / sheet / helix
PASS_COLOR = "#0072B2"   # colorblind-safe blue
FAIL_COLOR = "#D55E00"   # colorblind-safe vermillion


def set_style():
    """Publication-ish defaults: paper context, no top/right spines, editable PDF text."""
    sns.set_theme(context="paper", style="ticks", palette="colorblind")
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,          # editable text in Illustrator
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "legend.frameon": False,
    })


def save_fig(fig, name, fig_dir="figures"):
    """Write both a vector PDF and a 300-dpi PNG."""
    d = Path(fig_dir); d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.pdf")
    fig.savefig(d / f"{name}.png", dpi=300)
    print(f"saved: {d}/{name}.pdf (+ .png)")


# ── ternary (coil / sheet / helix) plotting ───────────────────────────────────
# A design's DSSP composition is three fractions summing to 1, which a triangle shows without
# loss. Hand-rolled rather than mpltern / python-ternary because those make 2-D density and KDE
# awkward, and density is the point: tens of thousands of overlapping points would hide exactly
# the sparse regions we care about. Corners: coil (0,0), sheet (1,0), helix (0.5, sqrt3/2).
import numpy as np
import matplotlib.patches as _mpatches

SQ3 = np.sqrt(3) / 2
TRI = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, SQ3]])


def tern_xy(C, E, H):
    """Barycentric (coil, sheet, helix) -> 2-D. Renormalised, so it tolerates rounding."""
    C, E, H = np.asarray(C, float), np.asarray(E, float), np.asarray(H, float)
    s = C + E + H
    s = np.where(s == 0, 1.0, s)
    E, H = E / s, H / s
    return E + 0.5 * H, SQ3 * H


def tri_patch(ax):
    """An invisible triangle attached to `ax`, used purely as a clip path."""
    p = _mpatches.Polygon(TRI, closed=True, transform=ax.transData)
    ax.add_patch(p); p.set_visible(False)
    return p


def white_to(color):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("bg", ["white", color])


def draw_tri(ax, title=None, labels=True):
    """Triangle frame, 20% gridlines, corner labels coloured by SSE_PALETTE."""
    ax.add_patch(_mpatches.Polygon(TRI, closed=True, fill=False, lw=0.9, ec="0.25", zorder=5))
    for t in np.arange(0.2, 1.0, 0.2):
        for a, b in (((1 - t, 0, t), (0, 1 - t, t)),
                     ((1 - t, t, 0), (0, t, 1 - t)),
                     ((t, 1 - t, 0), (t, 0, 1 - t))):
            x, y = tern_xy(*zip(a, b))
            ax.plot(x, y, lw=0.4, color="0.88", zorder=0)
    if labels:
        ax.text(-0.02, -0.045, "coil", ha="center", va="top", fontsize=8, color=SSE_PALETTE["C"])
        ax.text(1.02, -0.045, "sheet", ha="center", va="top", fontsize=8, color=SSE_PALETTE["E"])
        ax.text(0.5, SQ3 + 0.035, "helix", ha="center", va="bottom", fontsize=8,
                color=SSE_PALETTE["H"])
        for t in np.arange(0.2, 1.0, 0.2):
            x, y = tern_xy(1 - t, t, 0)
            ax.text(x, y - 0.03, f"{t:.1f}", ha="center", va="top", fontsize=6, color="0.55")
    ax.set_xlim(-0.10, 1.10); ax.set_ylim(-0.12, SQ3 + 0.10)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)


def tern_hex(ax, d, cmap="viridis", gridsize=32, **kw):
    """Log-scaled hexbin density of a table with C_frac/E_frac/H_frac, clipped to the triangle."""
    x, y = tern_xy(d["C_frac"], d["E_frac"], d["H_frac"])
    hb = ax.hexbin(x, y, gridsize=gridsize, bins="log", cmap=cmap, mincnt=1,
                   linewidths=0, zorder=2, **kw)
    hb.set_clip_path(tri_patch(ax))
    return hb


def tern_bg(ax, d, color="0.86", gridsize=32):
    """Faint pooled cloud, so each small multiple is read against the same reference."""
    x, y = tern_xy(d["C_frac"], d["E_frac"], d["H_frac"])
    hb = ax.hexbin(x, y, gridsize=gridsize, mincnt=1, linewidths=0, zorder=1,
                   cmap=white_to(color))
    hb.set_clip_path(tri_patch(ax))


def tern_kde(ax, d, color, levels=(0.9, 0.5), bw=0.20, nmax=4000, lw=(0.9, 1.6),
             label=None, rng=None):
    """Contours enclosing the given fractions of a subset's designs."""
    from scipy.stats import gaussian_kde
    rng = rng or np.random.default_rng(0)
    x, y = tern_xy(d["C_frac"], d["E_frac"], d["H_frac"])
    x, y = np.asarray(x), np.asarray(y)
    if len(x) > nmax:                       # KDE is O(n*m); subsample, seeded
        i = rng.choice(len(x), nmax, replace=False); x, y = x[i], y[i]
    if len(x) < 20 or np.ptp(x) < 1e-6 or np.ptp(y) < 1e-6:
        return
    k = gaussian_kde(np.vstack([x, y]), bw_method=bw)
    gx, gy = np.mgrid[-0.05:1.05:180j, -0.05:SQ3 + 0.05:180j]
    Z = k(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    zs = k(np.vstack([x, y]))
    lv = sorted({float(np.percentile(zs, 100 * (1 - f))) for f in levels})
    cs = ax.contour(gx, gy, Z, levels=lv, colors=[color], zorder=3, linewidths=list(lw)[:len(lv)])
    clip = tri_patch(ax)
    try:
        cs.set_clip_path(clip)              # mpl >= 3.8: ContourSet IS a Collection
    except AttributeError:
        for c in cs.collections:
            c.set_clip_path(clip)
    if label:
        ax.plot([], [], color=color, lw=1.6, label=label)
