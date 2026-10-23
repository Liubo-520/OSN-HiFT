"""Graphical abstract: study effects limit zero-shot prediction; one measurement unlocks calibrated transfer."""
import json

from matplotlib.patches import Circle, Rectangle

import figstyle as fs
from osn_core import RESULTS

plt = fs.plt
K = json.load(open(RESULTS / "key_numbers.json", encoding="utf-8"))


def ga_content(S, x0, y0, w, h, compact=False):
    """Three numbered cards in a strip whose top-left corner is (x0, y0) mm; compact shortens the wording for the
    TOC graphic instead of shrinking the type."""
    var = K["variance"]["Pure-solvent permeance | Raw (solvent fixed effects)"]
    bm = K["benchmark"]
    fsp = K["fewshot_sp"]
    cov = K["fewshot_sp_coverage"]
    ceil = K["noise"]["Pure-solvent permeance"]["r2_ceiling_unseen_paper"]
    head_fs, small = (6.0, 5.0) if compact else (6.6, 5.4)
    gap = 2.0 if compact else 3.0
    cw = (w - 2 * gap) / 3
    top = y0 + (0 if compact else 7.0)
    ch = y0 + h - top
    heads = (["Source paper\ndominates", "Random splits\noverstate R²", "One measurement\n+ HiFT"] if compact else
             ["Literature data are\ndominated by the source paper", "Random splits overstate\naccuracy for new papers",
              "One measurement + HiFT:\ncalibrated transfer"])
    hh = 10.5 if compact else 12.5
    axes = []
    for i in range(3):
        cx = x0 + i * (cw + gap)
        S.card(cx, top, cw, ch, r=1.2)
        ax = S.blank(cx, top, cw, ch, xlim=(0, cw), ylim=(0, ch))
        ax.add_patch(Circle((2.5, ch - 2.8), 1.6, fc=fs.INK, ec="none"))
        ax.text(2.5, ch - 2.85, str(i + 1), fontsize=small + 0.4, color="white", fontweight="bold", ha="center",
                va="center")
        ax.text(5.6, ch - 1.2, heads[i], fontsize=head_fs, color=fs.INK, ha="left", va="top", linespacing=fs.LS)
        axes.append((ax, cw, ch))
    if not compact:
        S.text(x0, y0 + 0.4, "What literature-mined data can and cannot predict in organic solvent nanofiltration",
               fontsize=7.4, fontweight="bold", va="top")

    # card 1: area-proportional split of the random-effect variance
    ax, cw_, ch_ = axes[0]
    side = min(ch_ - hh - 4.0, (cw_ - 6.0) if compact else 0.50 * cw_)
    bx, by = ((cw_ - side) / 2 if compact else 3.0), (ch_ - hh - side) / 2 + 0.6
    p, m, res = var["paper"], var["membrane"], var["residual"]
    ax.add_patch(Rectangle((bx, by), side * p, side, fc=fs.STUDY, ec="white", lw=0.8))
    ax.add_patch(Rectangle((bx + side * p, by + side * res / (m + res)), side * (1 - p), side * m / (m + res),
                           fc=fs.STUDY_MID, ec="white", lw=0.8))
    ax.add_patch(Rectangle((bx + side * p, by), side * (1 - p), side * res / (m + res), fc=fs.RESIDUAL, ec="white",
                           lw=0.8))
    ax.text(bx + side * p / 2, by + side * 0.56, f"{p:.0%}", fontsize=9.5 if compact else 12, color="white",
            fontweight="bold", ha="center", va="center")
    ax.text(bx + side * p / 2, by + side * 0.30, "paper", fontsize=small, color="white", ha="center", va="center")
    tx = bx + side + 1.8
    if not compact:
        ax.text(tx, by + side, "of the variance\nin log permeance\nis the source\npaper", fontsize=small, color=fs.INK,
                ha="left", va="top", linespacing=fs.LS)
    if not compact:
        ax.text(tx, by, f"membrane {m:.0%}\nresidual {res:.0%}", fontsize=small, color=fs.GREY_D, ha="left",
                va="bottom", linespacing=fs.LS)

    # card 2: R2 collapse against the reproducibility ceiling
    ax, cw_, ch_ = axes[1]
    vals = [bm["Random 5-fold | LightGBM"]["R2"], bm["Unseen paper | LightGBM"]["R2"],
            bm["Future papers (≥2021) | LightGBM"]["R2"]]
    cols = [fs.RANDOM, fs.PAPER, fs.FUTURE]
    labs = ["random", "new\npaper", "future"] if compact else ["random\nsplit", "new\npaper", "future\npapers"]
    base = 8.0 if compact else 9.0
    hmax = ch_ - hh - base - 4.5
    bw = cw_ * (0.17 if compact else 0.15)
    xs = [cw_ * 0.14 + j * (bw + cw_ * 0.08) for j in range(3)]
    ax.plot([cw_ * 0.08, cw_ * 0.95], [base, base], color=fs.GREY_D, lw=0.6)
    yc = base + hmax * ceil
    ax.plot([xs[1] - 1.0, cw_ * 0.95], [yc, yc], color=fs.STUDY, lw=0.8, ls=(0, (3, 1.5)))
    ax.text(cw_ * 0.95, yc + 0.7, "ceiling" if compact else "reproducibility\nceiling", fontsize=small - 0.4 * (not compact),
            color=fs.STUDY, ha="right", va="bottom", linespacing=fs.LS)
    for xb, v, c, lab in zip(xs, vals, cols, labs):
        ax.add_patch(Rectangle((xb, base), bw, hmax * v, fc=c, ec="none"))
        ax.text(xb + bw / 2, base + max(hmax * v, 0) + 0.7, fs.num(v), fontsize=small + 0.6, fontweight="bold",
                color=c, ha="center", va="bottom")
        ax.text(xb + bw / 2, base + min(hmax * v, 0) - 0.9, lab, fontsize=small, color=fs.GREY_D, ha="center",
                va="top", linespacing=fs.LS)

    # card 3: error versus measured solvents
    ax, cw_, ch_ = axes[2]
    ks = [0, 1, 2, 3]
    mae = [fsp["Zero-shot GBM | k=0"]["MAE"]] + [fsp[f"HiFT | k={i}"]["MAE"] for i in (1, 2, 3)]
    x0_, x1_ = cw_ * 0.14, cw_ * 0.90
    y0_, y1_ = (8.5 if compact else 9.5), ch_ - hh - 2.5
    mx = lambda kk: x0_ + (x1_ - x0_) * kk / 3
    my = lambda v: y0_ + (y1_ - y0_) * (v - 0.30) / (0.68 - 0.30)
    ax.plot([x0_ - 1.0, x1_ + 1.0], [y0_ - 1.6] * 2, color=fs.GREY_D, lw=0.6)
    ax.plot([mx(kk) for kk in ks], [my(v) for v in mae], color=fs.HERO, lw=1.6, zorder=3)
    for kk, v in zip(ks, mae):
        ax.add_patch(Circle((mx(kk), my(v)), 0.75, fc=fs.HERO if kk else fs.GREY_D, ec="white", lw=0.4, zorder=4))
        ax.text(mx(kk), y0_ - 2.1, str(kk), fontsize=small, color=fs.GREY_D, ha="center", va="top")
    ax.text((x0_ + x1_) / 2, y0_ - 4.6, "measured solvents", fontsize=small, color=fs.GREY_D, ha="center", va="top")
    drop = 1 - mae[1] / mae[0]
    ax.text(mx(1) + 1.4, my(mae[1]) + 1.2, f"−{drop:.0%} error", fontsize=small + 1.0, color=fs.HERO,
            fontweight="bold", ha="left", va="bottom")
    if not compact:
        c1 = cov["k=1 | level=0.9"]
        ax.text(mx(1.25), my(0.62), f"calibrated 90% interval\n({c1['coverage']:.0%} coverage),\n"
                                    f"×{cov['k=0 | level=0.9']['fold']:.0f} → ×{c1['fold']:.1f}",
                fontsize=small, color=fs.INK, ha="left", va="top", linespacing=fs.LS)
    ax.text(mx(0) + 1.1, my(mae[0]), "zero-shot", fontsize=small, color=fs.GREY_D, ha="left", va="center")
    return [a[0] for a in axes]


if __name__ == "__main__":
    S = fs.Sheet(60, width_mm=150)
    cards = ga_content(S, 2.0, 2.0, 146.0, 56.0)
    fs.export(S.fig, "GraphicalAbstract", axes=cards, ids=["a", "b", "c"], row_groups=[["a", "b", "c"]])
