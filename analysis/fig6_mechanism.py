"""Figure 6 | Within-membrane physics, membrane solvent-response fingerprints, and which solvent to measure first."""
import json

import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr

import figstyle as fs
from osn_core import RESULTS, SOLV

plt = fs.plt
SM = fs.FS["small"]
se = pd.read_csv(RESULTS / "solvent_effects.csv", index_col=0)
fp = pd.read_csv(RESULTS / "fingerprints.csv")
pop = json.load(open(RESULTS / "fingerprint_population.json"))
ex = pd.read_csv(RESULTS / "exemplar_series.csv")
des = pd.read_csv(RESULTS / "anchor_design.csv")
emp = pd.read_csv(RESULTS / "anchor_empirical.csv")

S = fs.Sheet(116)
Y1, H1 = 8.0, 36.0
Y2, H2 = 64.0, 40.0

# ---------------------------------------------------------------- a: physics appears only within membranes
ax_a = S.ax("a", 10.5, Y1, 54.0, H1)
S.letter(ax_a, "a", 0.0, 1.2, "Viscosity acts within membranes")
x = se.log_eta.values
for xi, b, w in zip(x, se.between, se.within):
    ax_a.plot([xi, xi], [b, w], color=fs.GREY_XL, lw=1.4, zorder=1, solid_capstyle="round")
ax_a.plot([x, x], [se.within_lo, se.within_hi], color=fs.HERO, lw=0.6, alpha=0.55, zorder=2)
ax_a.scatter(x, se.between, s=14, facecolor="white", edgecolor=fs.GREY_M, lw=0.8, zorder=3)
ax_a.scatter(x, se.within, s=16, color=fs.HERO, edgecolor="white", lw=0.4, zorder=4)
xx = np.linspace(-0.58, 0.36, 10)
for yv, col, ls in [(se.within.values, fs.HERO, "-"), (se.between.values, fs.GREY_M, (0, (3, 1.5)))]:
    b1, b0 = np.polyfit(x, yv, 1)
    ax_a.plot(xx, b0 + b1 * xx, color=col, lw=0.8, ls=ls, zorder=2)
rw, rb = pearsonr(se.within, se.log_eta).statistic, pearsonr(se.between, se.log_eta).statistic
ax_a.text(0.985, 0.985, f"within membranes  r = {fs.num(rw)}", transform=ax_a.transAxes, ha="right", va="top",
          fontsize=SM, color=fs.HERO)
ax_a.text(0.985, 0.875, f"literature-wide  r = {fs.num(rb)}", transform=ax_a.transAxes, ha="right", va="top",
          fontsize=SM, color=fs.GREY_D)
ax_a.text(0.985, 0.765, f"{len(se)} solvents", transform=ax_a.transAxes, ha="right", va="top", fontsize=5.0,
          color=fs.GREY_M)
ax_a.set_xlim(-0.69, 0.40)
ax_a.set_ylim(-0.78, 0.92)
ax_a.axhline(0, color=fs.GREY_L, lw=0.5, zorder=0)
LAB_A = {"MeCN": (-0.411, 0.62, "left", "center"), "Acetone": (-0.537, 0.345, "right", "center"),
         "EtOAc": (-0.352, 0.40, "left", "center"), "MEK": None,
         "Hexane": (-0.523, -0.315, "center", "top"), "Heptane": (-0.412, -0.575, "center", "top"),
         "THF": (-0.330, -0.215, "center", "top"), "MeOH": (-0.243, 0.135, "left", "center"),
         "Toluene": (-0.231, -0.301, "left", "center"), "EtOH": (0.052, -0.197, "left", "center"),
         "IPA": (0.287, -0.595, "right", "center")}
for nm in se.short:
    if LAB_A[nm] is None:
        continue
    lx, ly_, ha, va = LAB_A[nm]
    ax_a.text(lx, ly_, "EtOAc, MEK" if nm == "EtOAc" else nm, fontsize=5.0, color=fs.GREY_D, ha=ha, va=va, zorder=6)
ax_a.set_xticks([-0.5, -0.25, 0, 0.25])
ax_a.set_xlabel("log₁₀ solvent viscosity (mPa s)")
ax_a.set_ylabel("Solvent effect (log₁₀ P)")
ax_a.legend(handles=[Line2D([], [], marker="o", ls="none", color=fs.HERO, ms=3.2, label="Within membranes"),
                     Line2D([], [], marker="o", ls="none", mfc="white", mec=fs.GREY_M, mew=0.8, ms=3.0,
                            label="Literature-wide")],
            loc="lower left", bbox_to_anchor=(0.40, 0.0), fontsize=5.0, handletextpad=0.1, borderaxespad=0.1,
            labelspacing=0.2)

# ---------------------------------------------------------------- b: solvent-response fingerprints
ax_b = S.ax("b", 77.5, Y1, 40.5, H1)
S.letter(ax_b, "b", 68.5, 1.2, f"Fingerprints of {len(fp)} membranes")
bx, by = (-1.32, 0.02), (-1.2, 1.55)
ax_b.axvline(-1, color=fs.GREY_M, lw=0.6, ls=(0, (1, 1.4)))
ax_b.text(-1.015, 1.50, "viscous\nflow, η⁻¹", fontsize=5.0, color=fs.GREY_D, ha="right", va="top", linespacing=fs.LS)
pm = pop["beta"]["log_eta"]
ax_b.axvline(pm, color=fs.INK, lw=0.6, ls=(0, (3, 1.5)))
ax_b.text(pm + 0.02, 1.50, f"population\n{fs.num(pm)}", fontsize=5.0, color=fs.INK, ha="left", va="top",
          linespacing=fs.LS)
ax_b.axhline(0, color=fs.GREY_L, lw=0.5, zorder=0)
for mat in fs.MATERIAL_ORDER:
    s = fp[fp.material == mat]
    if not len(s):
        continue
    c = fs.MATERIAL_COLORS[mat]
    xl_, xh_ = (s.b_log_eta - s.sd_log_eta).clip(*bx), (s.b_log_eta + s.sd_log_eta).clip(*bx)
    yl_, yh_ = (10 * (s.b_dP - s.sd_dP)).clip(*by), (10 * (s.b_dP + s.sd_dP)).clip(*by)
    ax_b.plot([xl_, xh_], [10 * s.b_dP, 10 * s.b_dP], color=c, lw=0.5, alpha=0.30, zorder=1)
    ax_b.plot([s.b_log_eta, s.b_log_eta], [yl_, yh_], color=c, lw=0.5, alpha=0.30, zorder=1)
    ax_b.scatter(s.b_log_eta, 10 * s.b_dP, s=8, color=c, lw=0.25, edgecolor="white", zorder=3)
means = {}
for mat in fs.MATERIAL_ORDER:
    s = fp[fp.material == mat]
    if len(s) >= 3:
        means[mat] = (s.b_log_eta.mean(), 10 * s.b_dP.mean(), len(s))
        ax_b.scatter(*means[mat][:2], s=36, marker="D", color=fs.MATERIAL_COLORS[mat], edgecolor=fs.INK, lw=0.6,
                     zorder=5)
LAB_B = {"TFC/TFN": (-0.39, 1.06, "TFC/TFN (19)"), "Commercial polymer": (-0.43, 0.47, "Commercial (4)"),
         "Ceramic": (-0.185, -0.93, "Ceramic\n(9)"), "Glassy ISA": (-1.29, 0.78, "Glassy\nISA (24)"),
         "PDMS": (-1.29, -0.42, "PDMS (3)")}
for mat, (mx, my, n) in means.items():
    lx, ly_, txt = LAB_B[mat]
    txt = txt.rsplit("(", 1)[0] + f"({n})"
    right = lx > mx
    ax_b.annotate(txt, xy=(mx, my), xytext=(lx, ly_), fontsize=5.0, ha="left", va="center", color=fs.INK,
                  zorder=6, linespacing=fs.LS,
                  arrowprops=dict(arrowstyle="-", color=fs.GREY_D, lw=0.4, shrinkA=0.5, shrinkB=2.5,
                                  relpos=(0, 0.5) if right else (1, 0.5)))
ax_b.set_xlim(bx)
ax_b.set_ylim(by)
ax_b.set_xticks([-1.0, -0.5, 0])
ax_b.set_xticklabels(["−1", "−0.5", "0"])
ax_b.set_xlabel("Viscosity exponent  ∂log P / ∂log η")
ax_b.set_ylabel("Polar affinity (per 10 MPa½ δP)")

# ---------------------------------------------------------------- c: surface chemistry inverts the solvent ranking
ax_c = S.ax("c", 129.0, Y1, 22.0, H1)
S.letter(ax_c, "c", 120.5, 1.2, "Grafting inverts ranking")
t = ex[ex.series == "tio2_alkyl_series"]
order = ["Hexane", "Toluene", "Tetrahydrofuran (THF)"]
names = ["TiO2-unmodified-1nm", "TiO2-1nm-C1", "TiO2-1nm-C5", "TiO2-1nm-C8", "TiO2-1nm-C12"]
labels = ["native", "C1", "C5", "C8", "C12"]
cols = [fs.GREY_D] + [plt.get_cmap("YlOrBr")(v) for v in (0.35, 0.55, 0.75, 0.92)]
xs = np.arange(len(order))
ends = []
for nm, lab, c in zip(names, labels, cols):
    g = t[t.name == nm].set_index("testConditions.solvent1").reindex(order)
    ax_c.plot(xs, g.y, marker="o", color=c, lw=1.2 if nm == names[0] else 1.0, ms=3.0, mec="white", mew=0.3)
    ends.append((lab, g.y.iloc[-1], c if nm != names[0] else fs.INK))
ly = fs.repel_1d([e[1] for e in ends], 0.13)
for (lab, yv, c), yl in zip(ends, ly):
    ax_c.text(2.12, yl, lab, va="center", ha="left", fontsize=5.0, color=c)
y0 = t[(t.name == names[0]) & (t["testConditions.solvent1"] == "Hexane")].y.iloc[0]
y8 = t[(t.name == "TiO2-1nm-C8") & (t["testConditions.solvent1"] == "Hexane")].y.iloc[0]
fs.arrow(ax_c, (-0.14, y0 + 0.03), (-0.14, y8 - 0.03), color=fs.HERO, lw=0.7, ms=4.5, style="<|-|>")
ax_c.text(-0.2, (y0 + y8) / 2, f"×{10 ** (y8 - y0):.0f}", fontsize=5.4, color=fs.HERO, ha="right", va="center",
          fontweight="bold")
ax_c.set_xticks(xs, [f"{SOLV.loc[s, 'short']}\n{SOLV.loc[s, 'dP']:.1f}" for s in order], fontsize=5.6,
                linespacing=1.15)
ax_c.tick_params(axis="x", length=0)
ax_c.text(-0.06, -0.135, "δP", transform=ax_c.transAxes, ha="right", va="center", fontsize=5.0, color=fs.GREY_D)
ax_c.set_xlim(-0.6, 2.1)
ax_c.set_ylim(0.0, 2.35)
ax_c.set_yticks([0, 1, 2], ["1", "10", "100"])
ax_c.set_ylabel("Permeance (L m⁻² h⁻¹ bar⁻¹)")

# ---------------------------------------------------------------- d: Bayesian anchor design
ax_d = S.ax("d", 16.0, Y2, 52.0, H2)
S.letter(ax_d, "d", 0.0, 56.5, "Which solvent to measure first")
s1 = des[des.k == 1].set_index("anchor1")["reduction"].sort_values(ascending=False)
cand = list(s1.index)
M = np.full((len(cand), len(cand)), np.nan)
for _, r in des.iterrows():
    i = cand.index(r.anchor1)
    j = i if r.k == 1 else cand.index(r.anchor2)
    a, b = max(i, j), min(i, j)
    M[a, b] = r.reduction
cm = mcolors.LinearSegmentedColormap.from_list("info", ["#F3F8F5", "#A9D8BC", "#3E9E6B", "#145A38"])
cm.set_bad("white")
im = ax_d.imshow(np.ma.masked_invalid(M), cmap=cm, vmin=0.25, vmax=0.68, aspect="auto")
for (i, j), v in np.ndenumerate(M):
    if not np.isnan(v):
        ax_d.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=5.0,
                  color="white" if v > 0.55 else fs.INK, fontweight="bold" if i == j else "normal")
short = [SOLV.loc[s, "short"] for s in cand]
ax_d.set_xticks(range(len(cand)), short, rotation=90, fontsize=5.6)
ax_d.set_yticks(range(len(cand)), short, fontsize=5.6)
ax_d.tick_params(length=0, pad=1.5)
for sp_ in ax_d.spines.values():
    sp_.set_visible(False)
cax = ax_d.inset_axes([0.60, 0.88, 0.36, 0.045])
cb = S.fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_ticks([0.3, 0.45, 0.6])
cb.set_ticklabels(["30%", "45%", "60%"])
cb.ax.tick_params(labelsize=5.0, width=0.4, length=1.4, pad=0.8)
cb.outline.set_linewidth(0.4)
cax.set_title("expected variance reduction", fontsize=5.0, pad=2.4, color=fs.GREY_D)
best2 = des[des.k == 2].sort_values("reduction", ascending=False).iloc[0]
ax_d.text(0.97, 0.70, f"diagonal: one anchor\nbelow: anchor pairs\nbest pair: {SOLV.loc[best2.anchor1, 'short']} + "
                      f"{SOLV.loc[best2.anchor2, 'short']} ({100 * best2.reduction:.0f}%)",
          transform=ax_d.transAxes, ha="right", va="top", fontsize=5.0, color=fs.GREY_D, linespacing=fs.LS)

# ---------------------------------------------------------------- e: validation on held-out membranes
ax_e = S.ax("e", 84.0, Y2, 79.0, H2)
S.letter(ax_e, "e", 73.5, 56.5, "The design rule holds on held-out membranes")
e = emp[emp.n_membranes >= 5].copy()
e["short"] = SOLV.loc[e.anchor, "short"].values
sig_good, sig_bad = e.hi < 0, e.lo > 0
cols = np.where(sig_good, fs.GOOD, np.where(sig_bad, fs.BAD, fs.GREY_M))
ex_lim, ey_lim = (23, 58), (-0.14, 0.165)
ax_e.axhline(0, color=fs.GREY_D, lw=0.5, zorder=1)
ax_e.axhspan(0, ey_lim[1], color=fs.GREY_XL, alpha=0.35, lw=0, zorder=0)
ax_e.text(28.4, 0.158, "worse first anchor", fontsize=5.0, color=fs.GREY_D, ha="left", va="top", style="italic")
ax_e.text(23.6, -0.104, "better first anchor", fontsize=5.0, color=fs.GREY_D, ha="left", va="bottom", style="italic")
for xi, lo_, hi_, c in zip(100 * e.reduction, e.lo, e.hi, cols):
    ax_e.plot([xi, xi], [lo_, hi_], color=c if c != fs.GREY_M else fs.GREY_L, lw=0.9, zorder=2,
              solid_capstyle="butt")
sz = 3 + 0.55 * e.n_membranes.values
ax_e.scatter(100 * e.reduction, e.centered_mae, c=cols, s=sz, zorder=3, edgecolor="white", lw=0.4)
rho = spearmanr(e.reduction, -e.centered_mae).statistic
ax_e.text(0.985, 0.97, f"Spearman ρ = {rho:.2f} ({len(e)} anchor solvents)", transform=ax_e.transAxes, ha="right",
          va="top", fontsize=SM)
ax_e.set_xlabel("Predicted variance reduction from one anchor (%)")
ax_e.set_ylabel("Relative MAE at k = 1 (log₁₀)")
ax_e.set_xlim(ex_lim)
ax_e.set_ylim(ey_lim)
ax_e.set_yticks([-0.1, -0.05, 0, 0.05, 0.1, 0.15])
ex_ = e.assign(px=100 * e.reduction, col=[c if c != fs.GREY_M else fs.GREY_D for c in cols]).set_index("short")
for nm, dx, dy in [("MeCN", 0.9, 0.0), ("MeOH", 0.9, 0.013)]:
    r = ex_.loc[nm]
    ax_e.text(r.px + dx, r.centered_mae + dy, nm, fontsize=5.0, color=r.col, ha="left", va="center")
for row_y, names, xs_lab, top in [(0.124, ["DMF", "Hexane", "MEK", "EtOAc", "THF"], [41.6, 45.4, 49.6, 53.4, 57.0], True),
                                  (-0.115, ["EtOH", "IPA", "Toluene", "Acetone"], [43.6, 47.4, 51.6, 55.8], False)]:
    for nm, xl in zip(names, xs_lab):
        r = ex_.loc[nm]
        ax_e.text(xl, row_y, nm, fontsize=5.0, color=r.col, ha="center", va="bottom" if top else "top", zorder=6)
        y_end = (r.hi + 0.004) if top else (r.lo - 0.004)
        ax_e.plot([xl, r.px], [row_y - 0.003 if top else row_y + 0.003, y_end], color=fs.GREY_L, lw=0.4, zorder=1)
for nn, xx_ in [(5, 26.2), (20, 29.2), (40, 32.6)]:
    ax_e.scatter([xx_], [-0.035], s=3 + 0.55 * nn, color=fs.GREY_L, edgecolor="white", lw=0.4)
    ax_e.text(xx_, -0.047, f"{nn}", fontsize=5.0, color=fs.GREY_D, ha="center", va="top")
ax_e.text(29.4, -0.018, "membranes", fontsize=5.0, color=fs.GREY_D, ha="center", va="bottom")

fs.export(S.fig, "Fig6_mechanism", axes=[ax_a, ax_b, ax_c, ax_d, ax_e], ids=list("abcde"),
          row_groups=[["a", "b", "c"], ["d", "e"]],
          exemptions=[{"panels": ["a", "b", "c"], "checks": ["panel-width", "horizontal-gutter"],
                       "reason": "three-solvent series needs less width than the two continuous panels"},
                      {"panels": ["d", "e"], "checks": ["panel-width"], "reason": "matrix and scatter sized to content"}])
