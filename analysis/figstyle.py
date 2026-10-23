"""Shared publication style, semantic palette, modular millimetre layout and export/QA pipeline for all figures.

Every figure is assembled from rectangular modules on a millimetre grid (origin top-left). Plot areas in one row
share top and bottom edges, plot areas in one column share left and right edges, and the render-time alignment
audit enforces this before export. Concept modules sit on a pale card; data modules sit on white.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MPath

SKILL_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts" / "figure_qa"
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)
MM = 1 / 25.4
WIDTH = 165.0  # manuscript text width (6.5 in): figures are placed 1:1 and stay below the 7-in ACS double column
DOUBLE = WIDTH * MM
SINGLE = 80 * MM

FS = dict(letter=8.5, title=6.6, label=6.5, tick=6.0, note=5.8, small=5.4)
plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": FS["label"],
    "axes.titlesize": FS["title"],
    "axes.labelsize": FS["label"],
    "xtick.labelsize": FS["tick"],
    "ytick.labelsize": FS["tick"],
    "legend.fontsize": FS["note"],
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#1A1A1A",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.labelcolor": "#1A1A1A",
    "ytick.labelcolor": "#1A1A1A",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.45,
    "ytick.minor.width": 0.45,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "xtick.minor.size": 1.4,
    "ytick.minor.size": 1.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.handletextpad": 0.4,
    "legend.borderaxespad": 0.2,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "axes.labelpad": 2.2,
    "xtick.major.pad": 1.6,
    "ytick.major.pad": 1.6,
    "hatch.linewidth": 0.5,
    "mathtext.default": "regular",
    "axes.facecolor": "none",
    "savefig.facecolor": "white",
})
LS = 1.3  # leading for multi-line labels; tighter leading makes consecutive line boxes overlap in the PDF audit

# ---------------------------------------------------------------- semantic palette (identical meaning in every figure)
INK = "#1A1A1A"
GREY_D = "#4D4D4D"
GREY_M = "#8C8C8C"
GREY_L = "#BDBDBD"
GREY_XL = "#E4E4E4"
CARD = "#F1F4F8"
CARD_EDGE = "#D9E0E9"

HERO = "#B2182B"          # HiFT, the proposed method
HERO_MID = "#D6604D"
HERO_LIGHT = "#F4A582"
HERO_PALE = "#FBE1D6"
K_COLORS = {0: "#F4A582", 1: "#D6604D", 2: "#B2182B", 3: "#67001F"}   # number of anchors is ordinal

STUDY = "#5E4FA2"         # source-paper / laboratory effect and the reproducibility limit it implies
STUDY_MID = "#9C92CC"
STUDY_PALE = "#E8E4F4"
RESIDUAL = "#CFCFCF"

RANDOM = "#8C8C8C"        # record-level random split (interpolation)
SOLVENT = "#92C5DE"       # unseen solvent
PAPER = "#2166AC"         # unseen paper (new laboratory), the deployment case
PAPER_PALE = "#D1E5F0"
CLASS = "#0B3C6E"         # unseen material class
FUTURE = "#1B7F79"        # papers published after the training period
GOOD = "#2E9E5B"
BAD = "#C0392B"

PROTOCOL_COLORS = {
    "Random 5-fold": RANDOM,
    "Unseen solvent": SOLVENT,
    "Unseen paper": PAPER,
    "Unseen material class": CLASS,
    "Future papers (≥2021)": FUTURE,
}

# Okabe-Ito, colour-vision-safe material classes (unchanged from the first submission draft)
MATERIAL_COLORS = {
    "Ceramic": "#E69F00",
    "PDMS": "#56B4E9",
    "Glassy ISA": "#009E73",
    "TFC/TFN": "#0072B2",
    "2D/framework": "#CC79A7",
    "Commercial polymer": "#999999",
}
MATERIAL_ORDER = list(MATERIAL_COLORS)

METHOD_STYLE = {
    "Zero-shot GBM":             dict(color=GREY_D, ls=(0, (4, 2)), marker="o", lw=0.9, z=3, family="zero-shot"),
    "Viscosity scaling":         dict(color="#8C6D46", ls="-", marker="v", lw=0.9, z=3, family="physics"),
    "Vm/η scaling":              dict(color="#C9A97E", ls="-", marker="^", lw=0.9, z=2, family="physics"),
    "Per-membrane η fit":        dict(color="#E0CDB0", ls="-.", marker="x", lw=0.8, z=2, family="physics"),
    "Additive solvent offset":   dict(color="#7B8FA8", ls="-", marker="s", lw=0.9, z=3, family="statistical"),
    "GBM + mean shift":          dict(color="#A9BCD3", ls="-", marker="D", lw=0.9, z=3, family="statistical"),
    "Bayesian level shift":      dict(color="#4F6584", ls="-", marker="h", lw=0.9, z=3, family="statistical"),
    "Linear random-coefficient": dict(color="#2E3D55", ls=(0, (3, 1.5)), marker="h", lw=0.9, z=3,
                                      family="statistical"),
    "Pairwise transfer GBM":     dict(color="#E7A39C", ls="-", marker="P", lw=0.9, z=4, family="hift-part"),
    "HiFT − pairwise learner":   dict(color=HERO_LIGHT, ls="-", marker="o", lw=0.9, z=4, family="ablation"),
    "HiFT − auxiliary data":     dict(color=HERO_MID, ls=(0, (3, 1.5)), marker="o", lw=0.9, z=4, family="ablation"),
    "HiFT + zero-shot prior shrinkage": dict(color="#F2B8A2", ls=":", marker="o", lw=0.9, z=4, family="ablation"),
    "HiFT":                      dict(color=HERO, ls="-", marker="o", lw=1.7, z=6, family="hift"),
}
METHOD_LABEL = {"Vm/η scaling": "V$_m$/η scaling", "Linear random-coefficient": "Linear random coefficients",
                "Zero-shot GBM": "Zero-shot LightGBM", "GBM + mean shift": "LightGBM + mean shift"}

# aliases kept for the legacy names used in the first draft
VIOLET, GREEN, BLUE, BLUE_MID, BLUE_LIGHT = STUDY, GOOD, PAPER, "#4393C3", PAPER_PALE


# ---------------------------------------------------------------- modular sheet
class Sheet:
    """A figure laid out in millimetres from the top-left corner."""

    def __init__(self, height_mm, width_mm=WIDTH):
        self.w, self.h = float(width_mm), float(height_mm)
        self.fig = plt.figure(figsize=(self.w * MM, self.h * MM))
        self.panels = {}

    def rect(self, x, y, w, h):
        return [x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h]

    def ax(self, pid, x, y, w, h, **kw):
        a = self.fig.add_axes(self.rect(x, y, w, h), **kw)
        if pid:
            self.panels[pid] = a
        return a

    def blank(self, x, y, w, h, xlim=(0, 1), ylim=(0, 1)):
        """Axes in data units equal to millimetres when xlim/ylim span w/h, used for schematics."""
        a = self.fig.add_axes(self.rect(x, y, w, h))
        a.set_xlim(*xlim)
        a.set_ylim(*ylim)
        a.set_axis_off()
        a.patch.set_alpha(0)
        return a

    def card(self, x, y, w, h, fc=CARD, ec=CARD_EDGE, r=1.6):
        p = FancyBboxPatch((x / self.w, 1 - (y + h) / self.h), w / self.w, h / self.h,
                           boxstyle=f"round,pad=0,rounding_size={r / self.w}", transform=self.fig.transFigure,
                           fc=fc, ec=ec, lw=0.5, zorder=-10, mutation_aspect=self.w / self.h)
        self.fig.patches.append(p)
        return p

    def letter(self, ax, letter, x, y, title=None, title_dx=3.6, color=INK):
        """Bold panel letter at (x, y) mm from the top-left corner, with an optional one-line headline."""
        tr = self.fig.transFigure
        ax.text(x / self.w, 1 - y / self.h, letter, transform=tr, fontsize=FS["letter"], fontweight="bold",
                ha="left", va="top", color=color)
        if title:
            ax.text((x + title_dx) / self.w, 1 - (y + 0.55) / self.h, title, transform=tr, fontsize=FS["title"],
                    ha="left", va="top", color=INK)

    def text(self, x, y, s, **kw):
        kw.setdefault("fontsize", FS["note"])
        kw.setdefault("color", INK)
        return self.fig.text(x / self.w, 1 - y / self.h, s, **kw)


def num(x, d=2, plus=False):
    """Number with a typographic minus sign (U+2212)."""
    s = f"{abs(x):.{d}f}"
    if round(x, d) < 0:
        return "−" + s
    return ("+" + s) if plus and round(x, d) > 0 else s


def repel_1d(ys, min_gap, lo=None, hi=None):
    """Spread 1-D label positions so that neighbours are at least min_gap apart, keeping their order."""
    order = np.argsort(ys)
    y = np.asarray(ys, float)[order].copy()
    for _ in range(500):
        moved = False
        for i in range(1, len(y)):
            if y[i] - y[i - 1] < min_gap - 1e-12:
                d = (min_gap - (y[i] - y[i - 1])) / 2
                y[i - 1] -= d
                y[i] += d
                moved = True
        if lo is not None and y[0] < lo:
            y += lo - y[0]
        if hi is not None and y[-1] > hi:
            y -= y[-1] - hi
        if not moved:
            break
    out = np.empty_like(y)
    out[order] = y
    return out


def mm_offset(ax, dx_mm=0.0, dy_mm=0.0):
    from matplotlib.transforms import ScaledTranslation
    return ScaledTranslation(dx_mm * MM, dy_mm * MM, ax.figure.dpi_scale_trans)


def panel_label(ax, letter, x_pt=-24, y_pt=4, x=0.0, y=1.0):
    from matplotlib.transforms import ScaledTranslation
    offset = ScaledTranslation(x_pt / 72, y_pt / 72, ax.figure.dpi_scale_trans)
    ax.text(x, y, letter, transform=ax.transAxes + offset, fontsize=FS["letter"], fontweight="bold",
            ha="left", va="bottom", color=INK)


def despine_all(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def frame_only(ax, *sides):
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(s in sides)


def legend_marker(ax, color, marker="o", ms=3.4, ls="none", lw=1.0, mfc=None, label=None, **kw):
    return ax.plot([], [], marker=marker, color=color, ms=ms, ls=ls, lw=lw, mfc=mfc or color, label=label, **kw)[0]


def place_labels(ax, xs, ys, texts, fontsize=FS["small"], color=GREY_D, lines=(), marker_pt=4.0, pad_pt=1.0,
                 colors=None):
    """Greedy label placement: per point choose the offset whose rendered box avoids other labels, markers, the
    given line segments (data coordinates) and the axes boundary."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    to_disp = ax.transData.transform
    pts = to_disp(np.column_stack([xs, ys]))
    px = fig.dpi / 72
    mk = [(p[0] - marker_pt * px, p[1] - marker_pt * px, p[0] + marker_pt * px, p[1] + marker_pt * px) for p in pts]
    segs = [to_disp(np.asarray(s, float)) for s in lines]
    abox = ax.get_window_extent(r)
    placed = []

    def overlap(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    def seg_hits(bb, s):
        x0, y0, x1, y1 = bb
        for t in np.linspace(0, 1, 60):
            x, y = s[0] + t * (s[1] - s[0])
            if x0 <= x <= x1 and y0 <= y <= y1:
                return 1
        return 0

    for i, (p, txt) in enumerate(zip(pts, texts)):
        tmp = ax.text(0, 0, txt, fontsize=fontsize)
        ext = tmp.get_window_extent(r)
        tmp.remove()
        w, h = ext.width, ext.height
        g = 3.2 * px
        cands = [(g, -h / 2), (g, g * 0.4), (g, -h - g * 0.4), (-g - w, -h / 2), (-g - w, g * 0.4),
                 (-g - w, -h - g * 0.4), (-w / 2, g), (-w / 2, -h - g), (g * 1.5, g * 1.6), (-g * 1.5 - w, g * 1.6),
                 (g * 1.5, -h - g * 1.6), (-g * 1.5 - w, -h - g * 1.6)]
        best, best_s = None, None
        for dx, dy in cands:
            bb = (p[0] + dx - pad_pt * px, p[1] + dy - pad_pt * px, p[0] + dx + w + pad_pt * px,
                  p[1] + dy + h + pad_pt * px)
            s = sum(overlap(bb, q) for q in placed) * 10
            s += sum(overlap(bb, m) for j, m in enumerate(mk) if j != i) * 4
            s += sum(seg_hits(bb, sg) for sg in segs) * 2e4
            if bb[0] < abox.x0 or bb[2] > abox.x1 or bb[1] < abox.y0 or bb[3] > abox.y1:
                s += 1e6
            if best_s is None or s < best_s:
                best, best_s = (dx, dy, bb), s
        dx, dy, bb = best
        placed.append(bb)
        x_d, y_d = ax.transData.inverted().transform((p[0] + dx, p[1] + dy))
        ax.text(x_d, y_d, txt, fontsize=fontsize, color=(colors[i] if colors is not None else color), ha="left",
                va="bottom", zorder=6)


# ---------------------------------------------------------------- schematic primitives (axes in mm-like units)
def arrow(ax, p, q, color=GREY_D, lw=0.8, ms=6, ls="-", style="-|>", rad=0.0, z=3, shrink=0.0):
    a = FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls, zorder=z,
                        shrinkA=shrink, shrinkB=shrink, connectionstyle=f"arc3,rad={rad}", joinstyle="miter",
                        capstyle="butt")
    ax.add_patch(a)
    return a


def box(ax, x, y, w, h, fc="white", ec=GREY_L, lw=0.6, r=0.8, z=2, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z,
                       ls=ls)
    ax.add_patch(p)
    return p


def paper_icon(ax, x, y, w, h, color=STUDY, fc="white", lw=0.6, z=3, lines=3):
    """Document with a folded corner; (x, y) is the lower-left corner."""
    f = 0.28 * w
    pts = [(x, y), (x + w, y), (x + w, y + h - f), (x + w - f, y + h), (x, y + h)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=color, lw=lw, zorder=z, joinstyle="round"))
    ax.add_patch(Polygon([(x + w - f, y + h), (x + w - f, y + h - f), (x + w, y + h - f)], closed=True,
                         fc=color, ec=color, lw=lw * 0.8, zorder=z + 0.1, alpha=0.35))
    for i in range(lines):
        yy = y + h * (0.62 - 0.18 * i)
        ax.plot([x + 0.18 * w, x + 0.82 * w - (f if i == 0 else 0)], [yy, yy], color=color, lw=lw * 0.9,
                alpha=0.75, zorder=z + 0.2, solid_capstyle="round")


def membrane_icon(ax, x, y, w, h, color=STUDY_MID, top=None, lw=0.5, z=3):
    """Flat-sheet membrane coupon: porous support with a thin selective layer on top; (x, y) lower-left."""
    top = top or color
    ax.add_patch(FancyBboxPatch((x, y), w, h * 0.72, boxstyle=f"round,pad=0,rounding_size={h * 0.18}",
                                fc="white", ec=color, lw=lw, zorder=z))
    for i in range(5):
        cx = x + w * (0.15 + 0.175 * i)
        ax.add_patch(Circle((cx, y + h * 0.36), h * 0.09, fc=color, ec="none", alpha=0.35, zorder=z + 0.1))
    ax.add_patch(FancyBboxPatch((x, y + h * 0.74), w, h * 0.26, boxstyle=f"round,pad=0,rounding_size={h * 0.1}",
                                fc=top, ec=top, lw=lw, zorder=z + 0.2))


def droplet(ax, x, y, s, color=PAPER, ec=None, lw=0.5, z=4, alpha=1.0, aspect=1.0):
    """Solvent droplet (teardrop, tip up) centred at (x, y) with height s; aspect rescales x for unequal axes."""
    t = np.linspace(0, 2 * np.pi, 80)
    a = s / 2
    xs = x + aspect * 0.78 * a * np.sin(t) * np.sin(t / 2) ** 1.25
    ys = y + a * np.cos(t)
    ax.add_patch(Polygon(np.column_stack([xs, ys]), closed=True, fc=color, ec=ec or color, lw=lw, zorder=z,
                         alpha=alpha, joinstyle="round"))


def star(ax, x, y, s, color="#FDB863", ec=INK, lw=0.5, z=6):
    ang = np.deg2rad(90 + np.arange(10) * 36)
    rad = np.where(np.arange(10) % 2 == 0, s / 2, s / 5)
    ax.add_patch(Polygon(np.column_stack([x + rad * np.cos(ang), y + rad * np.sin(ang)]), closed=True, fc=color,
                         ec=ec, lw=lw, zorder=z, joinstyle="miter"))


def bracket(ax, x0, x1, y, h, color=GREY_D, lw=0.6, z=3):
    ax.plot([x0, x0, x1, x1], [y - h, y, y, y - h], color=color, lw=lw, zorder=z, solid_joinstyle="miter")


# ---------------------------------------------------------------- export and rendered QA
def export(fig, name, axes, ids, row_groups=None, column_groups=None, exemptions=(), exclude_axes=(),
           require_labels=False):
    base = FIGDIR / name
    kw = dict(axes=axes, panel_ids=ids, exemptions=list(exemptions), exclude_axes=list(exclude_axes))
    if row_groups is not None:
        kw["row_groups"] = row_groups
    if column_groups is not None:
        kw["column_groups"] = column_groups
    report = require_matplotlib_panel_alignment(
        fig, json_out=f"{base}.alignment.json", tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        require_panel_labels=require_labels, strict=True, **kw)
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.svg")
    fig.savefig(f"{base}.png", dpi=600)
    fig.savefig(f"{base}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    qa = {"alignment": report.get("verdict", report.get("summary"))}
    py = sys.executable
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    run = dict(capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    t = subprocess.run([py, str(SKILL_SCRIPTS / "audit_pdf_text.py"), f"{base}.pdf", "--min-pt", "5", "--json"], **run)
    try:
        rep = json.loads(t.stdout)
        qa["pdf_text"] = {k: rep.get(k) for k in ("minimum_found_pt", "below_minimum_count", "warnings")}
    except (json.JSONDecodeError, TypeError):
        qa["pdf_text"] = (t.stdout + t.stderr)[-400:]
    c = subprocess.run([py, str(SKILL_SCRIPTS / "audit_figure_collisions.py"), f"{base}.pdf",
                        "--json-out", f"{base}.collision-audit.json"], **run)
    qa["collision_exit"] = c.returncode
    qa["collision_tail"] = c.stdout.strip().splitlines()[-12:] if c.stdout else c.stderr[-400:]
    print(name, json.dumps(qa, indent=1, ensure_ascii=False))
    return qa
