"""Figure 1 | The OSN literature dataset is nested, sparse, drifting and heterogeneous across material classes."""
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde

import figstyle as fs
from osn_core import DATA, REF, S1, SOLV

plt = fs.plt
sp = pd.read_csv(DATA / "pure_solvent_permeance_domain.csv", low_memory=False)
n_pap, n_mem, n_rows, n_solv = sp[REF].nunique(), sp["entity"].nunique(), len(sp), sp[S1].nunique()
nsol = sp.groupby("entity")[S1].nunique()
ns = nsol.value_counts().sort_index()
n4 = int((nsol >= 4).sum())
n4p = sp[sp["entity"].isin(nsol[nsol >= 4].index)][REF].nunique()
frac1 = (nsol == 1).mean()
empty = 1 - sp.groupby(["entity", S1]).ngroups / (n_mem * n_solv)

S = fs.Sheet(112)
Y1, H1 = 7.4, 34.0           # row 1 plot band
Y2S, H2S = 59.5, 6.5         # row 2 marginal strips
Y2, H2 = 67.5, 30.0          # row 2 main plots
SM = fs.FS["small"]

# ---------------------------------------------------------------- a: nested data (concept card)
S.card(0, 0, 55, 51)
ax_a = S.blank(0, 0, 55, 51, xlim=(0, 55), ylim=(0, 51))
S.letter(ax_a, "a", 1.6, 1.2, "Nested literature data")
tiers = [(37.5, f"{n_pap}", "papers"), (24.2, f"{n_mem:,}", "membranes"),
         (13.0, f"{n_rows:,}", f"permeances\nin {n_solv} solvents")]
for y, big, small in tiers:
    ax_a.text(2.2, y + 0.9, big, fontsize=7.6, fontweight="bold", color=fs.STUDY if small == "papers" else fs.INK,
              ha="left", va="bottom")
    ax_a.text(2.2, y + 0.6, small, fontsize=SM, color=fs.GREY_D, ha="left", va="top", linespacing=fs.LS)
gx = [26.0, 38.0, 49.6]
mem_x = [[gx[0] - 2.7, gx[0] + 2.7], [gx[1] - 4.1, gx[1], gx[1] + 4.1], [gx[2]]]
for g, mxs in zip(gx, mem_x):
    fs.paper_icon(ax_a, g - 2.4, 35.0, 4.8, 6.0, color=fs.STUDY, lw=0.6)
    for mx in mxs:
        ax_a.plot([g, mx], [34.8, 25.0], color=fs.GREY_L, lw=0.55, zorder=1, solid_capstyle="round")
        fs.membrane_icon(ax_a, mx - 1.7, 22.6, 3.4, 2.2, color=fs.STUDY_MID, top=fs.STUDY_MID)
ax_a.text(54.4, 38.0, "…", fontsize=7, color=fs.GREY_D, ha="right", va="center")
drop_c = "#6BAED6"
for mxs in mem_x[:2]:
    for mx in mxs:
        ax_a.plot([mx, mx], [22.4, 16.6], color=fs.GREY_L, lw=0.55, zorder=1)
        fs.droplet(ax_a, mx, 14.6, 3.0, color=drop_c, lw=0.3)
for dx in (-3.6, -1.2, 1.2, 3.6):
    ax_a.plot([gx[2], gx[2] + dx], [22.4, 16.6], color=fs.HERO, lw=0.55, zorder=1, alpha=0.8)
    fs.droplet(ax_a, gx[2] + dx, 14.6, 3.0, color=drop_c, ec=fs.HERO, lw=0.5)
fs.bracket(ax_a, mem_x[1][0] - 1.9, mem_x[1][-1] + 1.9, 11.4, 0.8, color=fs.STUDY, lw=0.55)
ax_a.text(gx[1], 10.6, "siblings", fontsize=SM, color=fs.STUDY, ha="center", va="top")
ax_a.text(2.2, 1.6, "Siblings share laboratory, protocol and equipment.\n"
                    f"{frac1:.0%} of membranes were tested in one solvent;\n"
                    f"red, ≥4 solvents ({n4} membranes)",
          fontsize=SM, color=fs.GREY_D, ha="left", va="bottom", linespacing=fs.LS)

# ---------------------------------------------------------------- b: sparsity
ax_b = S.ax("b", 69.0, Y1, 32.0, H1)
S.letter(ax_b, "b", 58.5, 1.2, "Mostly one solvent per membrane")
cols = [fs.HERO if k >= 4 else (fs.GREY_M if k >= 2 else fs.GREY_L) for k in ns.index]
ax_b.bar(ns.index, ns.values, color=cols, width=0.74, lw=0)
ax_b.set_yscale("log")
ax_b.set_ylim(0.7, 20000)
ax_b.set_xlim(0.4, ns.index.max() + 0.6)
ax_b.set_xticks([1, 3, 5, 7, 9, 11])
ax_b.set_yticks([1, 10, 100, 1000])
ax_b.set_yticklabels(["1", "10", "10²", "10³"])
ax_b.minorticks_off()
ax_b.set_xlabel("Solvents per membrane")
ax_b.set_ylabel("Membranes")
ax_b.text(1.55, ns.loc[1], f"{ns.loc[1]:,}", ha="left", va="center", fontsize=SM, color=fs.GREY_D)
top4 = ns[ns.index >= 4].max()
fs.bracket(ax_b, 3.6, ns.index.max() + 0.4, top4 * 2.4, top4 * 0.5, color=fs.HERO, lw=0.6)
ax_b.text((3.6 + ns.index.max() + 0.4) / 2, top4 * 3.0, f"{n4} membranes,\n{n4p} papers", ha="center",
          va="bottom", fontsize=SM, color=fs.HERO, linespacing=fs.LS)
ax_b.text(0.99, 0.995, f"{empty:.0%} of membrane ×\nsolvent cells empty", transform=ax_b.transAxes,
          ha="right", va="top", fontsize=SM, color=fs.INK, linespacing=fs.LS)

# ---------------------------------------------------------------- c: drift of the material mix
ax_c = S.ax("c", 114.5, Y1, 50.0, H1)
S.letter(ax_c, "c", 105.0, 1.2, "Drifting material mix")
yr = sp.dropna(subset=["year"]).copy()
plabels = ["≤10", "11–15", "16–18", "19–20", "21", "22", "23", "24"]
yr["period"] = pd.cut(yr["year"], [1999, 2010, 2015, 2018, 2020, 2021, 2022, 2023, 2024], labels=plabels)
comp = (yr.groupby(["period", "material"], observed=False).size().unstack(fill_value=0)
        .reindex(columns=fs.MATERIAL_ORDER, fill_value=0))
share = comp.div(comp.sum(1), axis=0)
x = np.arange(len(share))
bottom = np.zeros(len(share))
seg = {}
for m in fs.MATERIAL_ORDER:
    ax_c.bar(x, share[m].values, bottom=bottom, color=fs.MATERIAL_COLORS[m], width=0.8, lw=0.25, edgecolor="white")
    seg[m] = (bottom.copy(), share[m].values.copy())
    bottom += share[m].values
dominant = {"Glassy ISA": 1, "Commercial polymer": 2, "TFC/TFN": 3, "2D/framework": 7, "Ceramic": 1}
short = {"Commercial polymer": "Commercial"}
for m, i in dominant.items():
    b0, h = seg[m][0][i], seg[m][1][i]
    if h < 0.16:
        continue
    ax_c.text(i, b0 + h / 2, short.get(m, m), rotation=90, ha="center", va="center", fontsize=SM,
              color=fs.INK if m in ("Ceramic", "Commercial polymer") else "white")
for xi, n in zip(x, comp.sum(1).values):
    ax_c.text(xi, 1.025, f"{n}", ha="center", va="bottom", fontsize=5.0, color=fs.GREY_D)
ax_c.axvline(3.5, color=fs.INK, lw=0.6, ls=(0, (2, 1.5)))
ax_c.set_xticks(x, plabels, fontsize=5.6)
ax_c.set_xlim(-0.55, len(x) - 0.45)
ax_c.set_ylim(0, 1.0)
ax_c.set_yticks([0, 0.5, 1.0], ["0", "50", "100"])
ax_c.set_ylabel("Records (%)")
ax_c.set_xlabel("Audited publication year (20xx)")

# ---------------------------------------------------------------- d, e: material block sharing class rows
order = fs.MATERIAL_ORDER
ax_dm = S.ax(None, 25.0, Y2S, 50.0, H2S)
ax_d = S.ax("d", 25.0, Y2, 50.0, H2)
S.letter(ax_d, "d", 1.6, 54.2, "Five orders of magnitude in permeance")
xlim_d = (-2.7, 4.7)
ax_dm.hist(sp["logP"], bins=np.arange(-2.7, 4.71, 0.2), color=fs.GREY_L, lw=0)
ax_dm.set_xlim(xlim_d)
ax_dm.set_xticks([])
ax_dm.set_yticks([])
fs.frame_only(ax_dm, "bottom")
ax_dm.text(0.0, 0.98, f"All {n_rows:,} records", transform=ax_dm.transAxes, fontsize=SM, color=fs.GREY_D,
           ha="left", va="top")
rng = np.random.default_rng(1)
for i, m in enumerate(order):
    v = sp.loc[sp.material == m, "logP"].values
    c = fs.MATERIAL_COLORS[m]
    grid = np.linspace(max(v.min() - 0.3, xlim_d[0]), min(v.max() + 0.3, xlim_d[1]), 200)
    dens = gaussian_kde(v)(grid)
    dens = 0.40 * dens / dens.max()
    ax_d.fill_between(grid, i - 0.04, i - 0.04 - dens, color=c, alpha=0.38, lw=0)
    ax_d.plot(grid, i - 0.04 - dens, color=c, lw=0.5)
    ax_d.scatter(v, i + 0.10 + rng.uniform(0, 0.24, len(v)), s=0.9, color=c, lw=0, alpha=0.75, rasterized=True)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    ax_d.plot([q1, q3], [i, i], color=fs.INK, lw=1.3, solid_capstyle="butt", zorder=4)
    ax_d.plot(med, i, marker="o", ms=2.6, mfc="white", mec=fs.INK, mew=0.6, zorder=5)
    ax_d.text(1.035, i, f"{len(v)}", transform=ax_d.get_yaxis_transform(), ha="left", va="center", fontsize=SM,
              color=fs.GREY_D, clip_on=False)
ax_d.text(1.035, -0.62, "n", transform=ax_d.get_yaxis_transform(), ha="left", va="bottom", fontsize=SM,
          color=fs.GREY_D, style="italic")
ax_d.set_ylim(len(order) - 0.45, -0.6)
ax_d.set_yticks(range(len(order)), order)
ax_d.tick_params(axis="y", length=0, pad=6.5)
for i, m in enumerate(order):
    ax_d.add_patch(Rectangle((-0.030, i - 0.16), 0.016, 0.32, transform=ax_d.get_yaxis_transform(), clip_on=False,
                             fc=fs.MATERIAL_COLORS[m], ec="none"))
ax_d.spines["left"].set_visible(False)
ax_d.set_xlim(xlim_d)
ax_d.set_xticks([-2, -1, 0, 1, 2, 3, 4], ["10⁻²", "10⁻¹", "1", "10", "10²", "10³", "10⁴"])
ax_d.set_xlabel("Pure-solvent permeance (L m⁻² h⁻¹ bar⁻¹)")

top = sp[S1].value_counts().head(15).index
mat = (sp[sp[S1].isin(top)].groupby(["material", S1]).size().unstack(fill_value=0)
       .reindex(index=order, columns=top, fill_value=0))
ax_em = S.ax(None, 88.0, Y2S, 63.0, H2S)
ax_e = S.ax("e", 88.0, Y2, 63.0, H2)
S.letter(ax_e, "e", 82.0, 54.2, "Uneven solvent coverage by material class")
xe = np.arange(len(top))
tot = mat.sum(0).values
ax_em.bar(xe, tot, color=fs.GREY_L, width=0.7, lw=0)
ax_em.set_xlim(-0.6, len(top) - 0.4)
ax_em.set_ylim(0, tot.max() * 1.5)
ax_em.set_xticks([])
ax_em.set_yticks([])
fs.frame_only(ax_em, "bottom")
ax_em.text(0, tot[0] * 1.04, f"{tot[0]}", fontsize=5.0, color=fs.GREY_D, ha="center", va="bottom")
ax_em.text(1.0, 0.98, "records per solvent", transform=ax_em.transAxes, fontsize=SM, color=fs.GREY_D,
           ha="right", va="top")
for i in range(len(order)):
    ax_e.axhline(i, color=fs.GREY_XL, lw=0.4, zorder=0)
for j in xe:
    ax_e.axvline(j, color=fs.GREY_XL, lw=0.4, zorder=0)
smax = 150.0
for i, m in enumerate(order):
    v = mat.loc[m].values
    ok = v > 0
    ax_e.scatter(xe[ok], np.full(ok.sum(), i), s=smax * v[ok] / mat.values.max(), color=fs.MATERIAL_COLORS[m],
                 alpha=0.88, lw=0.3, edgecolor="white", zorder=3)
ax_e.set_ylim(len(order) - 0.45, -0.6)
ax_e.set_xlim(-0.6, len(top) - 0.4)
ax_e.set_yticks(range(len(order)), [])
ax_e.tick_params(axis="y", length=0)
ax_e.set_xticks(xe, [SOLV.loc[s, "short"] for s in top], rotation=90, fontsize=5.8)
ax_e.tick_params(axis="x", length=0, pad=1.5)
for s_ in ("left", "bottom"):
    ax_e.spines[s_].set_visible(False)
ax_l = S.blank(152.5, Y2, 12.5, H2, xlim=(0, 12.5), ylim=(0, H2))
for nn, yy in [(10, 24.0), (50, 17.5), (200, 9.5)]:
    ax_l.scatter([3.0], [yy], s=smax * nn / mat.values.max(), color=fs.GREY_L, lw=0.3, edgecolor="white")
    ax_l.text(6.2, yy, f"{nn}", fontsize=SM, va="center", ha="left", color=fs.GREY_D)
ax_l.text(0.4, 29.6, "Records", fontsize=SM, va="top", ha="left", color=fs.GREY_D)

fs.export(S.fig, "Fig1_landscape", axes=[ax_b, ax_c, ax_d, ax_e], ids=list("bcde"),
          row_groups=[["b", "c"], ["d", "e"]],
          exemptions=[{"panels": ["b", "c"], "checks": ["panel-width"], "reason": "count and share panels sized to content"},
                      {"panels": ["d", "e"], "checks": ["panel-width"],
                       "reason": "distribution and 15-solvent matrix differ in content width"}])
