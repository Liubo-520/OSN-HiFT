"""Figure 5 | Honest uncertainty: naive calibration fails on new papers; HiFT intervals are calibrated and narrower."""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import figstyle as fs
import fewshot_summary as fsum
from osn_core import RESULTS

plt = fs.plt
SM = fs.FS["small"]
cf = pd.read_csv(RESULTS / "conformal_coverage.csv")
oof = pd.read_csv(RESULTS / "benchmark_oof.csv", low_memory=False)
cov_sp = fsum.coverage("pure_solvent_permeance", "HiFT")
cov_pm = fsum.coverage("solution_permeance", "HiFT")
cov_sp.to_csv(RESULTS / "hift_coverage_sp.csv", index=False)
cov_pm.to_csv(RESULTS / "hift_coverage_pm.csv", index=False)
SCOL = {"Random rows": fs.RANDOM, "Unseen papers": fs.PAPER, "Future papers (≥2021)": fs.FUTURE}
OOFS = {"Random rows": "Random 5-fold", "Unseen papers": "Unseen paper", "Future papers (≥2021)": "Future papers (≥2021)"}
CAL = [("Random-row calibration", (0, (3, 1.5)), "o", True), ("Paper-held-out calibration", "-", "s", False)]


def fold(q):
    f = 10 ** q
    return f"×{f:.0f}" if f >= 10 else f"×{f:.1f}"


S = fs.Sheet(107)
XL, XR, W, H = 11.0, 96.0, 66.5, 33.0
Y1, Y2 = 11.0, 65.0

# ---------------------------------------------------------------- legend strips
lg1 = S.blank(XL, 5.0, 150.0, 4.0, xlim=(0, 150), ylim=(0, 4))
x = 0.0
for name, c in SCOL.items():
    lg1.plot([x, x + 4.5], [2, 2], color=c, lw=1.4)
    lg1.text(x + 5.6, 2, name, fontsize=SM, va="center", color=fs.INK)
    x += 7.0 + 1.06 * len(name) * 0.95
x += 3.0
for name, ls, mk, hollow in CAL:
    lg1.plot([x, x + 4.5], [2, 2], color=fs.GREY_D, lw=0.9, ls=ls)
    lg1.plot(x + 2.25, 2, marker=mk, ms=2.6, color=fs.GREY_D, mfc="white" if hollow else fs.GREY_D, mew=0.6)
    label = "calibrated on random rows" if hollow else "calibrated on held-out papers"
    lg1.text(x + 5.6, 2, label, fontsize=SM, va="center", color=fs.INK)
    x += 7.0 + 1.06 * len(label) * 0.95

# ---------------------------------------------------------------- a: why random-row calibration fails
ax_a = S.ax("a", XL, Y1, W, H)
S.letter(ax_a, "a", 0.0, 1.2, "Intervals sized on random splits are too narrow for new papers")
q_rows = cf[(cf.scenario == "Unseen papers") & (cf.calibration == "Random-row calibration") & (cf.level == 0.9)].q.mean()
q_pap = cf[(cf.scenario == "Unseen papers") & (cf.calibration == "Paper-held-out calibration") & (cf.level == 0.9)].q.mean()
for name, scen in OOFS.items():
    o = oof[(oof.scenario == scen) & (oof.model == "LightGBM")]
    r = np.sort((o.y - o.pred).abs().values)
    cdf = np.arange(1, len(r) + 1) / len(r)
    keep = r <= 2.5
    xs_, ys_ = np.r_[0, r[keep], 2.5], np.r_[0, cdf[keep], cdf[keep][-1]]
    ax_a.step(xs_, ys_, where="post", color=SCOL[name],
              lw=1.4 if name == "Unseen papers" else 1.0, zorder=3 if name == "Unseen papers" else 2)
ax_a.axhline(0.9, color=fs.GREY_L, lw=0.6, zorder=1)
ax_a.text(2.48, 0.915, "90% target", fontsize=5.0, color=fs.GREY_D, ha="right", va="bottom")
o = oof[(oof.scenario == "Unseen paper") & (oof.model == "LightGBM")]
ru = (o.y - o.pred).abs().values
for q, ls, lab in [(q_rows, (0, (3, 1.5)), "random-row\ncalibration"), (q_pap, "-", "held-out-paper\ncalibration")]:
    cov = (ru <= q).mean()
    ax_a.plot([q, q], [0, cov], color=fs.PAPER, lw=0.8, ls=ls, zorder=2)
    ax_a.plot(q, cov, "o", ms=3.2, color=fs.PAPER, mfc="white" if ls != "-" else fs.PAPER, mew=0.8, zorder=4)
    ax_a.text(q + 0.04, 0.05, f"{lab}\n{fold(q)}", fontsize=5.0, color=fs.PAPER, ha="left", va="bottom",
              linespacing=fs.LS)
cov_rows = (ru <= q_rows).mean()
ax_a.annotate(f"covers {cov_rows:.0%} of\nnew-paper errors", xy=(q_rows, cov_rows), xytext=(0.72, 0.40),
              fontsize=5.0, color=fs.PAPER, ha="left", va="center", linespacing=fs.LS,
              arrowprops=dict(arrowstyle="-", color=fs.PAPER, lw=0.5, shrinkA=1, shrinkB=2, relpos=(0, 0.5)))
ax_a.set_xlim(0, 2.5)
ax_a.set_ylim(0, 1.02)
ax_a.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
ax_a.set_yticks([0, 0.5, 1.0])
ax_a.set_xlabel("|Residual| of zero-shot LightGBM (log₁₀ units)")
ax_a.set_ylabel("Cumulative fraction")

# ---------------------------------------------------------------- b: conformal reliability
ax_b = S.ax("b", XR, Y1, W, H)
S.letter(ax_b, "b", 85.5, 1.2, "Conformal calibration must hold out papers")
ax_b.plot([0.45, 1], [0.45, 1], color=fs.INK, lw=0.5, ls=(0, (1, 1.4)), zorder=1)
for s, c in SCOL.items():
    for cal, ls, mk, hollow in CAL:
        g = cf[(cf.scenario == s) & (cf.calibration == cal)].groupby("level")["coverage"].agg(["mean", "std"])
        ax_b.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), color=c, ls=ls, marker=mk, ms=2.6, lw=0.9,
                      elinewidth=0.5, capsize=0, mfc="white" if hollow else c, mew=0.7, zorder=3)
up = cf[(cf.scenario == "Unseen papers") & (cf.level == 0.9)].groupby("calibration")["coverage"].mean()
ax_b.annotate(f"promised 90%,\ndelivered {up['Random-row calibration']:.0%}",
              xy=(0.9, up["Random-row calibration"] - 0.03), xytext=(0.985, 0.20), fontsize=5.0, color=fs.PAPER,
              ha="right", va="center", linespacing=fs.LS,
              arrowprops=dict(arrowstyle="-|>", color=fs.PAPER, lw=0.5, mutation_scale=5, shrinkA=1, shrinkB=1,
                              relpos=(0.75, 1.0)))
ax_b.set_xlim(0.45, 1.0)
ax_b.set_ylim(0.1, 1.02)
ax_b.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax_b.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax_b.set_xlabel("Nominal coverage")
ax_b.set_ylabel("Empirical coverage")

# ---------------------------------------------------------------- legend strip, row 2
lg2 = S.blank(XL, 59.0, 150.0, 4.0, xlim=(0, 150), ylim=(0, 4))
x = 0.0
for k, c in fs.K_COLORS.items():
    lg2.plot([x, x + 4.5], [2, 2], color=c, lw=1.2)
    lg2.plot(x + 2.25, 2, "o", ms=2.6, color=c)
    lab = f"HiFT, k = {k}" + (" (no anchor)" if k == 0 else "")
    lg2.text(x + 5.6, 2, lab, fontsize=SM, va="center", color=fs.INK)
    x += 7.0 + 1.06 * len(lab) * 0.95
lg2.plot([x, x + 4.5], [2, 2], color=fs.GREY_D, lw=0.9, ls=(0, (3, 1.5)))
lg2.plot(x + 2.25, 2, "^", ms=2.8, color=fs.GREY_D)
lg2.text(x + 5.6, 2, "HiFT, k = 1, solution permeance", fontsize=SM, va="center", color=fs.INK)

# ---------------------------------------------------------------- c: HiFT reliability
ax_c = S.ax("c", XL, Y2, W, H)
S.letter(ax_c, "c", 0.0, 55.0, "HiFT intervals are calibrated at every k")
ax_c.plot([0.45, 1], [0.45, 1], color=fs.INK, lw=0.5, ls=(0, (1, 1.4)), zorder=1)
for k, col in fs.K_COLORS.items():
    g = cov_sp[cov_sp.k == k]
    ax_c.fill_between(g.level, g.lo.clip(lower=0.4), g.hi.clip(upper=1.02), color=col, alpha=0.10, lw=0)
    ax_c.plot(g.level, g.coverage, color=col, marker="o", ms=2.6, lw=1.0)
g = cov_pm[cov_pm.k == 1]
ax_c.plot(g.level, g.coverage, color=fs.GREY_D, marker="^", ms=2.8, lw=0.8, ls=(0, (3, 1.5)))
ax_c.set_xlim(0.45, 1.0)
ax_c.set_ylim(0.4, 1.02)
ax_c.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax_c.set_yticks([0.4, 0.6, 0.8, 1.0])
ax_c.set_xlabel("Nominal coverage")
ax_c.set_ylabel("Empirical coverage")

# ---------------------------------------------------------------- d: sharpness versus calibration at the 90% level
ax_d = S.ax("d", XR, Y2, W, H)
S.letter(ax_d, "d", 85.5, 55.0, "Anchors sharpen calibrated intervals")
ax_d.axhline(0.9, color=fs.GREY_L, lw=0.6, zorder=0)
ax_d.axhspan(0.4, 0.80, color=fs.GREY_XL, alpha=0.45, lw=0, zorder=0)
ax_d.text(2.05, 0.43, "overconfident", fontsize=5.0, color=fs.GREY_D, ha="left", va="bottom", style="italic")
pts = []
for cal, mk, hollow in [("Random-row calibration", "o", True), ("Paper-held-out calibration", "s", False)]:
    g = cf[(cf.scenario == "Unseen papers") & (cf.calibration == cal) & (cf.level == 0.9)]
    m, sd, q = g.coverage.mean(), g.coverage.std(), g.q.mean()
    ax_d.errorbar(10 ** q, m, yerr=sd, fmt=mk, color=fs.PAPER, ms=3.4, mfc="white" if hollow else fs.PAPER, mew=0.8,
                  elinewidth=0.6, capsize=0, zorder=4)
    pts.append((10 ** q, m, ("zero-shot, random-row cal." if hollow else "zero-shot, held-out papers") + f" {fold(q)}",
                fs.PAPER))
hx, hy = [], []
for k, col in fs.K_COLORS.items():
    r = cov_sp[(cov_sp.k == k) & np.isclose(cov_sp.level, 0.9)].iloc[0]
    ax_d.errorbar(10 ** r.half_width_log, r.coverage, yerr=[[r.coverage - r.lo], [r.hi - r.coverage]], fmt="o",
                  color=col, ms=3.6, elinewidth=0.6, capsize=0, zorder=5, mec="white", mew=0.4)
    hx.append(10 ** r.half_width_log)
    hy.append(r.coverage)
    pts.append((10 ** r.half_width_log, r.coverage, f"k = {k} {fold(r.half_width_log)}", col))
fs.arrow(ax_d, (hx[0] * 0.88, 0.995), (hx[3] * 1.12, 0.995), color=fs.HERO, lw=0.7, ms=5)
ax_d.text(np.sqrt(hx[0] * hx[3]), 1.005, "HiFT: more anchors", fontsize=5.0, color=fs.HERO, ha="center", va="bottom")
zx = {lab.split(",")[1].split(" ×")[0].strip(): (xv, yv) for xv, yv, lab, col in pts[:2]}
ax_d.text(zx["random-row cal."][0] * 1.15, zx["random-row cal."][1], f"zero-shot, random rows {fold(np.log10(zx['random-row cal.'][0]))}",
          fontsize=5.0, color=fs.PAPER, ha="left", va="center")
ax_d.text(zx["held-out papers"][0] * 0.98, 1.035,
          f"zero-shot {fold(np.log10(zx['held-out papers'][0]))}", fontsize=5.0, color=fs.PAPER, ha="left", va="bottom")
for k in (0, 1, 3):
    label_x = hx[k] * 1.22 if k == 0 else hx[k]
    ax_d.text(label_x, 0.850, fold(np.log10(hx[k])), fontsize=5.0, color=fs.K_COLORS[k], ha="center", va="top")
ax_d.set_xscale("log")
ax_d.set_xlim(2, 60)
ax_d.set_xticks([2, 5, 10, 20, 50], ["×2", "×5", "×10", "×20", "×50"])
ax_d.minorticks_off()
ax_d.set_ylim(0.4, 1.06)
ax_d.set_yticks([0.4, 0.6, 0.8, 1.0])
ax_d.set_xlabel("90% interval half-width (fold change)")
ax_d.set_ylabel("Empirical coverage")

fs.export(S.fig, "Fig5_uncertainty", axes=[ax_a, ax_b, ax_c, ax_d], ids=list("abcd"),
          row_groups=[["a", "b"], ["c", "d"]], column_groups=[["a", "c"], ["b", "d"]])
