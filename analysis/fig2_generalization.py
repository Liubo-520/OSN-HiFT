"""Figure 2 | Reported accuracy collapses under deployment-relevant validation."""
import json

import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch

import figstyle as fs
from osn_core import RESULTS, SOLV

plt = fs.plt
SM = fs.FS["small"]
met = pd.read_csv(RESULTS / "benchmark_metrics.csv")
oof = pd.read_csv(RESULTS / "benchmark_oof.csv", low_memory=False)
tc = pd.read_csv(RESULTS / "temporal_control.csv")
rj = pd.read_csv(RESULTS / "rejection_benchmark.csv")
mlp = pd.read_csv(RESULTS / "si_mlp.csv").groupby("protocol")["R2"].mean()
nt = pd.read_csv(RESULTS / "si_nested_tuning.csv").set_index(["protocol", "tuning"])["R2"]
cr = pd.read_csv(RESULTS / "si_cleaning_rules.csv").groupby(["dataset", "protocol"])["R2"].mean()
noise = json.load(open(RESULTS / "noise_summary.json"))["Pure-solvent permeance"]


def bm(s, m):
    return met[(met.scenario == s) & (met.model == m)].iloc[0]


S = fs.Sheet(113)
YA, HA = 21.0, 33.0
YC, HC = 74.0, 27.0

# ---------------------------------------------------------------- a: protocol pictograms + R2 ladder
scen = ["Random 5-fold", "Unseen solvent", "Unseen paper", "Unseen material class", "Future papers (≥2021)"]
scen_lbl = ["Random\n5-fold", "Unseen\nsolvent", "Unseen\npaper", "Unseen\nmaterial class", "Future papers\n(≥2021)"]
AX0, AW = 13.0, 88.0
ax_a = S.ax("a", AX0, YA, AW, HA)
S.letter(ax_a, "a", 0.0, 1.2, "Accuracy collapses outside the training papers")
measured = {0: [0, 1], 1: [0, 2, 3], 2: [0, 1], 3: [0, 3], 4: [1, 3]}       # paper -> solvents measured
paper_class = {0: 0, 1: 1, 2: 0, 3: 0, 4: 1}
test_cells = {
    "Random 5-fold": {(0, 1), (1, 3), (3, 0), (4, 1)},
    "Unseen solvent": {(p, 1) for p, ss in measured.items() if 1 in ss},
    "Unseen paper": {(2, s) for s in measured[2]},
    "Unseen material class": {(p, s) for p, ss in measured.items() for s in ss if paper_class[p] == 1},
    "Future papers (≥2021)": {(p, s) for p, ss in measured.items() for s in ss if p >= 3},
}
colw = AW / len(scen)
for i, s in enumerate(scen):
    cx = AX0 + (i + 0.5) * colw
    px = S.blank(cx - 7.0, 6.6, 14.0, 12.4, xlim=(0, 14.0), ylim=(0, 12.4))
    col = fs.PROTOCOL_COLORS[s]
    for p in range(5):
        x = 1.6 + p * 2.7
        px.add_patch(FancyBboxPatch((x - 1.05, 2.3), 2.1, 9.4, boxstyle="round,pad=0,rounding_size=0.5",
                                    fc="white", ec=fs.GREY_L, lw=0.4, zorder=1))
        px.add_patch(FancyBboxPatch((x - 0.9, 1.15), 1.8, 0.7, boxstyle="round,pad=0,rounding_size=0.25",
                                    fc=fs.GREY_D if paper_class[p] else fs.GREY_L, ec="none", zorder=1))
        for sv in range(4):
            y = 10.4 - sv * 2.25
            if sv in measured[p]:
                test = (p, sv) in test_cells[s]
                px.add_patch(Circle((x, y), 0.68, fc=col if test else "white", ec=col if test else fs.GREY_M,
                                    lw=0.45, zorder=3))
            else:
                px.add_patch(Circle((x, y), 0.16, fc=fs.GREY_XL, ec="none", zorder=2))
    if s.startswith("Future"):
        fs.arrow(px, (0.6, 0.35), (13.4, 0.35), color=fs.GREY_D, lw=0.5, ms=4)
kx = S.blank(0, 6.6, 12.5, 12.4, xlim=(0, 12.5), ylim=(0, 12.4))
kx.add_patch(Circle((1.4, 10.4), 0.68, fc=fs.GREY_D, ec=fs.GREY_D, lw=0.45))
kx.text(2.6, 10.4, "test", fontsize=SM, va="center", color=fs.GREY_D)
kx.add_patch(Circle((1.4, 8.15), 0.68, fc="white", ec=fs.GREY_M, lw=0.45))
kx.text(2.6, 8.15, "train", fontsize=SM, va="center", color=fs.GREY_D)
kx.add_patch(FancyBboxPatch((0.35, 2.3), 2.1, 4.2, boxstyle="round,pad=0,rounding_size=0.5", fc="white",
                            ec=fs.GREY_L, lw=0.4))
kx.text(2.9, 4.4, "paper", fontsize=SM, va="center", color=fs.GREY_D)
kx.text(2.9, 1.5, "class", fontsize=SM, va="center", color=fs.GREY_D)
kx.add_patch(FancyBboxPatch((0.5, 1.15), 1.8, 0.7, boxstyle="round,pad=0,rounding_size=0.25", fc=fs.GREY_D,
                            ec="none"))

lo, hi = noise["r2_ceiling_unseen_paper_lo"], noise["r2_ceiling_unseen_paper_hi"]
ax_a.fill_between([1.5, 4.5], lo, min(hi, 0.95), color=fs.STUDY_PALE, lw=0, zorder=0)
ax_a.plot([1.5, 4.5], [noise["r2_ceiling_unseen_paper"]] * 2, color=fs.STUDY, lw=0.8, ls=(0, (3, 1.5)), zorder=1)
ax_a.text(4.45, noise["r2_ceiling_unseen_paper"] + 0.035, "reproducibility ceiling", ha="right", va="bottom",
          fontsize=SM, color=fs.STUDY)
ax_a.axhline(0, color=fs.GREY_D, lw=0.5, zorder=1)
models = [("LightGBM", fs.INK, "o", "full"), ("Random forest", "#707070", "s", "full"),
          ("Ridge", "#A8A8A8", "D", "full"), ("Solvent mean", "#A8A8A8", "^", "none")]
floor_y = -0.68
for j, (m, c, mk, fill) in enumerate(models):
    off = (j - 1.5) * 0.17
    for i, s in enumerate(scen):
        r = bm(s, m)
        x = i + off
        if r.R2 < floor_y:
            ax_a.plot(x, floor_y, marker="v", color=c, ms=3.0, zorder=4, mec="none")
            ax_a.text(x - 0.07, floor_y, fs.num(r.R2, 1), ha="right", va="center", fontsize=5.0, color=fs.GREY_D)
            continue
        ax_a.plot([x, x], [max(r.R2_lo, floor_y), r.R2_hi], color=c, lw=0.8, zorder=3, solid_capstyle="butt")
        ax_a.plot(x, r.R2, marker=mk, color=c, ms=3.3 if m == "LightGBM" else 3.0, zorder=4,
                  mfc=c if fill == "full" else "white", mec=c if fill == "none" else "white", mew=0.6)
r0, r2 = bm("Random 5-fold", "LightGBM").R2, bm("Unseen paper", "LightGBM").R2
fs.arrow(ax_a, (-0.255 + 0.06, r0 + 0.035), (2 - 0.255 - 0.04, r2 + 0.06), color=fs.INK, lw=0.7, ms=5, rad=-0.2)
ax_a.text(0.80, 1.005, f"ΔR² = {fs.num(r2 - r0)}", ha="center", va="bottom", fontsize=fs.FS["note"], color=fs.INK,
          fontweight="bold")
ax_a.set_xticks(range(len(scen)), scen_lbl, fontsize=5.8, linespacing=1.15)
ax_a.tick_params(axis="x", length=0)
ax_a.set_xlim(-0.5, 4.5)
ax_a.set_ylim(-0.80, 1.12)
ax_a.set_yticks([-0.5, 0, 0.5, 1.0])
ax_a.set_ylabel("Out-of-sample R²")
handles = [Line2D([], [], marker=mk, color=c, ls="-", lw=0.8, ms=3.0, mfc=c if fill == "full" else "white",
                  mec=c if fill == "none" else "white", label=m) for m, c, mk, fill in models]
ax_a.legend(handles=handles, loc="lower left", fontsize=SM, handlelength=1.2, labelspacing=0.25,
            borderaxespad=0.15, bbox_to_anchor=(0.0, 0.02))

# ---------------------------------------------------------------- b: robustness of the gap
ax_b = S.ax("b", 128.5, YA, 32.0, HA)
S.letter(ax_b, "b", 104.0, 1.2, "The gap is not an artefact")
rows = [("Model class", None, None),
        ("LightGBM", bm("Random 5-fold", "LightGBM").R2, bm("Unseen paper", "LightGBM").R2),
        ("Random forest", bm("Random 5-fold", "Random forest").R2, bm("Unseen paper", "Random forest").R2),
        ("MLP", mlp["Random 5-fold"], mlp["Unseen paper"]),
        ("Protocol", None, None),
        ("Grouped tuning", nt[("Random 5-fold", "Grouped inner CV")], nt[("Unseen paper", "Grouped inner CV")]),
        ("Original cleaning", cr[("Original cleaning rules", "Random 5-fold")],
         cr[("Original cleaning rules", "Unseen paper")]),
        ("Target", None, None),
        ("Rejection", rj[(rj.scenario == "Random 5-fold") & (rj.model == "LightGBM")].R2.iloc[0],
         rj[(rj.scenario == "Unseen paper") & (rj.model == "LightGBM")].R2.iloc[0])]
ylab, ypos = [], []
y = 0.0
for name, a, b in rows:
    if a is None:
        ax_b.text(-0.135, y + 0.15, name, transform=ax_b.get_yaxis_transform(), ha="right", va="center",
                  fontsize=SM, color=fs.GREY_M, style="italic")
        y += 0.85
        continue
    ax_b.plot([b, a], [y, y], color=fs.GREY_XL, lw=2.2, solid_capstyle="round", zorder=1)
    fs.arrow(ax_b, (a - 0.02, y), (b + 0.035, y), color=fs.GREY_M, lw=0.6, ms=4, z=2)
    ax_b.plot(a, y, "o", color=fs.RANDOM, ms=3.3, zorder=3, mec="white", mew=0.4)
    ax_b.plot(b, y, "o", color=fs.PAPER, ms=3.3, zorder=3, mec="white", mew=0.4)
    ylab.append(name)
    ypos.append(y)
    y += 1.0
ax_b.set_yticks(ypos, ylab, fontsize=5.8)
ax_b.tick_params(axis="y", length=0)
ax_b.spines["left"].set_visible(False)
ax_b.set_ylim(y - 0.3, -0.6)
ax_b.set_xlim(-0.1, 1.0)
ax_b.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
ax_b.axvline(0, color=fs.GREY_L, lw=0.5, zorder=0)
ax_b.set_xlabel("R²")
ax_b.legend(handles=[Line2D([], [], marker="o", ls="none", color=fs.RANDOM, ms=3.3, label="Random"),
                     Line2D([], [], marker="o", ls="none", color=fs.PAPER, ms=3.3, label="Unseen paper")],
            loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=SM, handletextpad=0.1,
            columnspacing=0.8, borderaxespad=0.1)

# ---------------------------------------------------------------- c, d: parity
ext = (-2.6, 3.7)
norm = mcolors.LogNorm(vmin=1, vmax=60)
cmap = mcolors.LinearSegmentedColormap.from_list("ink", ["#E3ECF5", "#8FB4D6", "#2E6DA4", "#0B2A4A"])
axes_cd = []
for k, (pid, x0, s, title, col) in enumerate([("c", 12.0, "Random 5-fold", "Random 5-fold", fs.RANDOM),
                                            ("d", 45.0, "Unseen paper", "Unseen paper", fs.PAPER)]):
    ax = S.ax(pid, x0, YC, 27.0, HC)
    axes_cd.append(ax)
    o = oof[(oof.scenario == s) & (oof.model == "LightGBM") & (oof.seed == 0)]
    hb = ax.hexbin(o["y"], o["pred"], gridsize=30, extent=ext + ext, cmap=cmap, norm=norm, mincnt=1, lw=0.1,
                   edgecolors="face")
    ax.plot(ext, ext, color=fs.INK, lw=0.5, ls=(0, (3, 1.5)))
    r = bm(s, "LightGBM")
    ax.text(0.04, 0.97, title, transform=ax.transAxes, ha="left", va="top", fontsize=SM, color=col,
            fontweight="bold")
    ax.text(0.04, 0.86, f"R² = {r.R2:.2f}\nMAE = {r.MAE:.2f}", transform=ax.transAxes, ha="left", va="top",
            fontsize=SM, linespacing=fs.LS)
    ax.set_xlim(ext)
    ax.set_ylim(ext)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.set_xlabel("Measured log₁₀ P")
axes_cd[0].set_ylabel("Predicted log₁₀ P")
axes_cd[1].set_yticklabels([])
S.letter(axes_cd[0], "c", 0.0, 67.0, "Parity, LightGBM")
S.letter(axes_cd[1], "d", 41.0, 67.0)
cax = axes_cd[1].inset_axes([0.58, 0.10, 0.36, 0.045])
cb = S.fig.colorbar(hb, cax=cax, orientation="horizontal")
cb.set_ticks([1, 10, 50])
cb.set_ticklabels(["1", "10", "50"])
cb.ax.tick_params(labelsize=5.0, width=0.4, length=1.4, pad=0.8)
cb.outline.set_linewidth(0.4)
cax.set_title("records", fontsize=5.0, pad=2.4, color=fs.GREY_D)

# ---------------------------------------------------------------- e: temporal drift versus data size
ax_e = S.ax("e", 86.0, YC, 25.0, HC)
S.letter(ax_e, "e", 76.0, 67.0, "Drift, not data size")
labels = ["Past only (≤2020)", "Same size, mixed years"]
piv = tc.pivot(index="rep", columns="training", values="R2")[labels]
for _, row in piv.iterrows():
    ax_e.plot([0, 1], row.values, color=fs.GREY_L, lw=0.5, zorder=1)
for i, (lab, c) in enumerate(zip(labels, [fs.FUTURE, fs.GREY_D])):
    v = piv[lab].values
    ax_e.scatter(np.full(len(v), i), v, s=6, color=c, zorder=2, lw=0)
    ax_e.errorbar(i + (0.22 if i else -0.22), v.mean(), yerr=v.std(ddof=1), fmt="o", color=c, ms=3.2, lw=0.9,
                  capsize=1.4, zorder=3)
diff = piv[labels[1]] - piv[labels[0]]
ax_e.text(0.5, 0.985, f"ΔR² = +{diff.mean():.2f} ± {diff.std(ddof=1):.2f}\n{(diff > 0).sum()}/{len(diff)} splits",
          transform=ax_e.transAxes, ha="center", va="top", fontsize=SM, linespacing=fs.LS)
ax_e.axhline(0, color=fs.GREY_D, lw=0.5, zorder=0)
ax_e.set_xticks([0, 1], ["Past\nonly", "Mixed\nyears"], fontsize=5.8, linespacing=1.15)
ax_e.tick_params(axis="x", length=0)
ax_e.set_xlim(-0.5, 1.5)
ax_e.set_ylim(-0.75, 0.95)
ax_e.set_yticks([-0.5, 0, 0.5])
ax_e.set_ylabel("R², papers ≥2021")

# ---------------------------------------------------------------- f: per-solvent decomposition
ax_f = S.ax("f", 123.0, YC, 38.0, HC)
S.letter(ax_f, "f", 114.0, 67.0, "New paper harder than new solvent")
lgo = oof[oof.model == "LightGBM"]
per = {}
for s in ["Random 5-fold", "Unseen solvent", "Unseen paper"]:
    xx = lgo[lgo.scenario == s]
    per[s] = xx.assign(ae=(xx.y - xx.pred).abs()).groupby("solvent")["ae"].mean()
solv = per["Unseen solvent"].sort_values().index
xs = np.arange(len(solv))
for i, sname in enumerate(solv):
    vals = [per[s][sname] for s in per]
    ax_f.plot([i, i], [min(vals), max(vals)], color=fs.GREY_XL, lw=1.6, zorder=1, solid_capstyle="round")
for s, c, mk in [("Random 5-fold", fs.RANDOM, "o"), ("Unseen solvent", fs.SOLVENT, "s"), ("Unseen paper", fs.PAPER, "D")]:
    ax_f.scatter(xs, per[s].reindex(solv).values, color=c, marker=mk, s=8, zorder=3, lw=0.3, edgecolor="white",
                 label=s)
ax_f.axhline(noise["mae_floor_interlab"], color=fs.STUDY, lw=0.7, ls=(0, (3, 1.5)), zorder=2, label="Inter-lab floor")
ax_f.set_xticks(xs, [SOLV.loc[s, "short"] for s in solv], rotation=90, fontsize=5.6)
ax_f.tick_params(axis="x", length=0, pad=1.2)
ax_f.set_xlim(-0.6, len(solv) - 0.4)
ax_f.set_ylim(0, 1.08)
ax_f.set_yticks([0, 0.5, 1.0])
ax_f.set_ylabel("MAE (log₁₀)")
ax_f.legend(loc="upper left", ncol=2, fontsize=5.0, handletextpad=0.2, columnspacing=0.6, borderaxespad=0.05,
            handlelength=1.3, bbox_to_anchor=(-0.02, 1.04))

fs.export(S.fig, "Fig2_generalization", axes=[ax_a, ax_b, axes_cd[0], axes_cd[1], ax_e, ax_f], ids=list("abcdef"),
          row_groups=[["a", "b"], ["c", "d", "e", "f"]],
          exemptions=[{"panels": ["a", "b"], "checks": ["panel-width"], "reason": "hero ladder is wider than the robustness panel"},
                      {"panels": ["c", "d", "e", "f"], "checks": ["panel-width", "horizontal-gutter"],
                       "reason": "square parity pair shares one y axis; paired and per-solvent panels sized to content"}])
