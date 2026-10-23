"""Figure 4 | One to three measurements of a new membrane enable accurate cross-solvent prediction (HiFT)."""
import json

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch
from scipy.stats import norm

import figstyle as fs
import fewshot_summary as fsum
from osn_core import RESULTS, SOLV

plt = fs.plt
SM = fs.FS["small"]
noise = json.load(open(RESULTS / "noise_summary.json"))["Pure-solvent permeance"]
ind = json.load(open(RESULTS / "indep_summary.json", encoding="utf-8"))
sp_sum = fsum.summary("pure_solvent_permeance")
pm_sum = fsum.summary("solution_permeance")
sp_raw = fsum.load("pure_solvent_permeance")
sp_sum.to_csv(RESULTS / "fewshot_summary_sp.csv", index=False)
pm_sum.to_csv(RESULTS / "fewshot_summary_pm.csv", index=False)
BASELINES = ["Viscosity scaling", "Vm/η scaling", "Additive solvent offset", "GBM + mean shift", "Bayesian level shift",
             "Linear random-coefficient", "Pairwise transfer GBM"]

S = fs.Sheet(146)

# ---------------------------------------------------------------- a: concept card
S.card(0, 0, 104, 50)
ax_a = S.blank(0, 0, 104, 50, xlim=(0, 104), ylim=(0, 50))
S.letter(ax_a, "a", 1.6, 1.2, "HiFT: hierarchical few-shot transfer")


def stage(x, n, text):
    ax_a.add_patch(Circle((x + 1.5, 41.6), 1.45, fc=fs.INK, ec="none", zorder=3))
    ax_a.text(x + 1.5, 41.55, str(n), fontsize=5.6, color="white", fontweight="bold", ha="center", va="center", zorder=4)
    ax_a.text(x + 3.7, 41.6, text, fontsize=6.0, color=fs.INK, ha="left", va="center")


stage(2.5, 1, "Learn the solvent response")
stage(40.0, 2, "Measure one solvent")
stage(70.0, 3, "Predict every solvent")
shape = np.array([0.0, 0.55, 0.25, 0.95, 0.62])
xs1 = np.linspace(6.5, 31.5, 5)
jit = np.array([[0.0, 0.25, -0.2, 0.1, 0.2], [0.2, -0.15, 0.25, -0.2, 0.0], [-0.2, 0.1, 0.0, 0.25, -0.25],
                [0.1, -0.25, 0.15, -0.1, 0.25]])
greys = ["#C9C3E3", "#ADA4D6", "#9086C6", "#7468B4"]
for i, off in enumerate([0.0, 2.5, 5.0, 7.5]):
    y = 27.4 + off + 2.4 * shape + jit[i]
    ax_a.plot(xs1, y, color=greys[i], lw=0.8, zorder=2)
    ax_a.scatter(xs1, y, s=4, color=greys[i], zorder=3, lw=0)
ax_a.text(4.0, 38.6, "multi-solvent membranes, training papers", fontsize=5.0, color=fs.GREY_D, ha="left", va="center")
fs.arrow(ax_a, (19.0, 25.6), (19.0, 21.4), color=fs.GREY_D, lw=0.7, ms=5)
ax_a.text(20.4, 23.5, "remove each\nmembrane level", fontsize=5.0, color=fs.GREY_D, ha="left", va="center",
          linespacing=fs.LS)
for i in range(4):
    ax_a.plot(xs1, 13.6 + 2.4 * shape + 0.45 * jit[i], color=fs.HERO_LIGHT, lw=0.7, zorder=2)
ax_a.plot(xs1, 13.6 + 2.4 * shape, color=fs.HERO, lw=1.5, zorder=3)
ax_a.text(4.0, 10.6, "shared within-membrane response ĉ\n= ½ backfitted GBM + ½ pairwise GBM", fontsize=5.0,
          color=fs.INK, ha="left", va="top", linespacing=fs.LS)
for sx in xs1:
    ax_a.plot([sx, sx], [12.4, 12.9], color=fs.GREY_L, lw=0.5)

fs.membrane_icon(ax_a, 45.0, 20.5, 14.0, 4.6, color=fs.STUDY_MID, top=fs.STUDY_MID, lw=0.6)
fs.droplet(ax_a, 52.0, 31.6, 6.0, color="#6BAED6", lw=0.4)
fs.star(ax_a, 55.4, 34.4, 3.6, color="#FDB863", lw=0.5)
fs.arrow(ax_a, (52.0, 28.2), (52.0, 25.6), color=fs.GREY_D, lw=0.7, ms=5)
ax_a.text(52.0, 19.4, "new membrane, new lab", fontsize=5.0, color=fs.GREY_D, ha="center", va="top")
ax_a.text(57.8, 31.0, "anchor\ny(a)", fontsize=5.0, color=fs.INK, ha="left", va="center", linespacing=fs.LS)

xs3 = np.linspace(73.5, 98.5, 6)
shape3 = np.array([0.30, 0.85, 0.15, 1.0, 0.55, 0.0])
lvl, prior = 15.0, 19.5
for j, (x, sh) in enumerate(zip(xs3, shape3)):
    y0 = prior + 7.0 * sh
    ax_a.plot([x - 0.9, x - 0.9], [y0 - 6.2, y0 + 6.2], color=fs.HERO_LIGHT, lw=0.9, alpha=0.75, solid_capstyle="butt")
    ax_a.add_patch(Circle((x - 0.9, y0), 0.55, fc=fs.HERO_LIGHT, ec="none", zorder=3))
    y1 = lvl + 7.0 * sh
    if j == 2:
        fs.star(ax_a, x + 0.9, y1, 3.2, color="#FDB863", lw=0.5)
        continue
    ax_a.plot([x + 0.9, x + 0.9], [y1 - 2.6, y1 + 2.6], color=fs.HERO, lw=0.9, solid_capstyle="butt")
    ax_a.add_patch(Circle((x + 0.9, y1), 0.6, fc=fs.HERO, ec="none", zorder=3))
ax_a.text(72.0, 38.0, "no anchor: zero-shot prior", fontsize=5.0, color=fs.HERO_MID, ha="left", va="center")
ax_a.text(72.0, 8.7, "one anchor: level fixed,\n90% interval narrows", fontsize=5.0, color=fs.HERO, ha="left", va="top",
          linespacing=fs.LS)
ax_a.plot([72.0, 101.5], [11.2, 11.2], color=fs.GREY_L, lw=0.5)
fs.arrow(ax_a, (35.5, 14.6), (70.6, 14.6), color=fs.HERO, lw=0.8, ms=6, rad=0.0)
fs.arrow(ax_a, (61.0, 31.0), (70.6, 24.0), color=fs.GREY_D, lw=0.8, ms=6, rad=-0.15)
ax_a.text(4.0, 2.6, "ŷ(t) = mean over anchors a of [y(a) + ĉ(t ← a)],   σ̂(t) = σ̂ √(1 + 1/k);   "
                    "without anchors, ŷ falls back on the zero-shot prior", fontsize=5.0, color=fs.GREY_D,
          ha="left", va="bottom")

# ---------------------------------------------------------------- b: pre-specified representative membrane
ax_b = S.ax("b", 116.5, 8.0, 46.5, 28.0)
S.letter(ax_b, "b", 107.0, 1.2, "One anchor fixes the level")
s_all = sp_raw[(sp_raw.seed == 0) & (sp_raw.method == "HiFT")]
gain = pd.Series({e: g[g.k == 0]["abs_err"].mean() - g[g.k == 1]["abs_err"].mean() for e, g in s_all.groupby("entity")})
nsol = s_all.groupby("entity")["n_solvents"].first()
cands = gain[nsol.reindex(gain.index) >= 6]
ent = (cands - gain.median()).abs().idxmin()
ex = s_all[s_all.entity == ent]
per_anchor = ex[ex.k == 1].groupby("anchors")["abs_err"].mean()
anchor = (per_anchor - per_anchor.median()).abs().idxmin()
k0 = ex[ex.k == 0].set_index("target")
k1 = ex[(ex.k == 1) & (ex.anchors == anchor)].set_index("target")
order = k0.sort_values("y").index.tolist()
xs = np.arange(len(order))
z = norm.ppf(0.95)
tg = [s for s in order if s != anchor]
ax_b.errorbar(xs - 0.2, k0.loc[order, "pred"], yerr=z * k0.loc[order, "sd"], fmt="D", color=fs.HERO_LIGHT, ms=2.3,
              lw=0.8, capsize=0)
ax_b.errorbar(np.array([order.index(s) for s in tg]) + 0.2, k1.loc[tg, "pred"], yerr=z * k1.loc[tg, "sd"], fmt="o",
              color=fs.HERO, ms=2.5, lw=0.9, capsize=0)
ax_b.scatter(xs, k0.loc[order, "y"], s=10, color=fs.INK, zorder=5, lw=0)
ai = order.index(anchor)
fs.star(ax_b, ai, k0.loc[anchor, "y"], 0.5, color="#FDB863", lw=0.5)
mae0 = np.mean(np.abs(k0.loc[tg, "y"] - k0.loc[tg, "pred"]))
mae1 = np.mean(np.abs(k1.loc[tg, "y"] - k1.loc[tg, "pred"]))
ax_b.set_xticks(xs, [SOLV.loc[s, "short"] for s in order], rotation=90, fontsize=5.6)
ax_b.tick_params(axis="x", length=0, pad=1.2)
ax_b.set_ylabel("log₁₀ P")
lo = min((k0.pred - z * k0.sd).min(), k0.y.min()) - 0.15
hi = max((k0.pred + z * k0.sd).max(), k0.y.max())
ax_b.set_ylim(lo, hi + 0.40 * (hi - lo))
ax_b.set_xlim(-0.6, len(order) - 0.4)
ax_b.legend(handles=[Line2D([], [], marker="o", ls="none", color=fs.INK, ms=2.8, label="Measured"),
                     Line2D([], [], marker="*", ls="none", mfc="#FDB863", mec=fs.INK, mew=0.4, ms=5.5,
                            label=f"Anchor ({SOLV.loc[anchor, 'short']})"),
                     Line2D([], [], marker="D", ls="-", color=fs.HERO_LIGHT, ms=2.3, lw=0.8, label="k = 0"),
                     Line2D([], [], marker="o", ls="-", color=fs.HERO, ms=2.5, lw=0.9, label="k = 1")],
            loc="upper left", ncol=2, fontsize=5.0, handlelength=1.1, columnspacing=0.8, borderaxespad=0.1,
            handletextpad=0.3)
ax_b.text(0.99, 0.985, f"MAE {mae0:.2f} → {mae1:.2f}", transform=ax_b.transAxes, ha="right", va="top", fontsize=SM,
          color=fs.HERO, fontweight="bold")
example_info = dict(entity=ent, anchor=anchor, mae_k0=mae0, mae_k1=mae1, n_solvents=len(order),
                    material=k0["material"].iloc[0], median_gain_all=float(gain.median()),
                    gain_example=float(gain[ent]), n_candidates=int(len(cands)), n_membranes=int(len(gain)))
json.dump(example_info, open(RESULTS / "fig4_example.json", "w"), indent=1)

# ---------------------------------------------------------------- c: error versus number of anchors
YR, HR = 61.0, 31.0
ax_c = S.ax("c", 9.5, YR, 29.5, HR)
S.letter(ax_c, "c", 0.0, 53.5, "Error versus anchors")
f_floor = noise["mae_floor_interlab"]
ax_c.axhline(f_floor, color=fs.STUDY, lw=0.7, ls=(0, (3, 1.5)), zorder=1)
zs = sp_sum[(sp_sum.method == "Zero-shot GBM") & (sp_sum.k == 0)].iloc[0]
ax_c.axhline(zs.mae, color=fs.GREY_D, lw=0.8, ls=(0, (4, 2)), zorder=1)
show = ["Vm/η scaling", "Viscosity scaling", "Bayesian level shift", "Pairwise transfer GBM", "HiFT"]
ends = []
for m in show:
    st = fs.METHOD_STYLE[m]
    g = sp_sum[(sp_sum.method == m) & (sp_sum.k <= 3)].sort_values("k")
    if m == "HiFT":
        ax_c.fill_between(g.k, g.lo.clip(lower=0.24), g.hi.clip(upper=0.70), color=fs.HERO, alpha=0.10, lw=0,
                          zorder=1)
    ax_c.plot(g.k, g.mae, color=st["color"], ls=st["ls"], marker=st["marker"], ms=2.8 if m != "HiFT" else 3.4,
              lw=st["lw"], zorder=st["z"], mec="white", mew=0.3)
    ends.append((fs.METHOD_LABEL.get(m, m).replace("V$_m$", "Vm"), g.mae.iloc[-1], st["color"]))
ends += [("Zero-shot LightGBM", zs.mae, fs.GREY_D), ("Inter-lab floor", f_floor, fs.STUDY)]
ax_c.set_xticks([0, 1, 2, 3])
ax_c.set_xlim(-0.25, 3.15)
ax_c.set_ylim(0.24, 0.70)
ax_c.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
ax_c.set_xlabel("Anchors measured (k)")
ax_c.set_ylabel("MAE (log₁₀)")
gap = 0.031
ly = fs.repel_1d([e[1] for e in ends], gap, lo=0.262, hi=0.69)
for (lab, yv, col), yl in zip(ends, ly):
    ax_c.annotate(lab, xy=(3.15, yv), xytext=(3.55, yl), textcoords="data", fontsize=5.0, color=col, ha="left",
                  va="center", annotation_clip=False,
                  arrowprops=dict(arrowstyle="-", color=fs.GREY_L, lw=0.4, shrinkA=0.8, shrinkB=1.0, relpos=(0, 0.5)))

# ---------------------------------------------------------------- d: paired differences and ablations
ax_d = S.ax("d", 94.5, YR, 26.5, HR)
S.letter(ax_d, "d", 62.0, 53.5, "Paired comparison")
rows = [("Pairwise transfer GBM", "Pairwise transfer GBM"), ("GBM + mean shift", "LightGBM + mean shift"),
        ("Bayesian level shift", "Bayesian level shift"), ("Viscosity scaling", "Viscosity scaling"),
        ("Linear random-coefficient", "Linear random coefficients"), ("Additive solvent offset", "Additive offset"),
        ("Vm/η scaling", "Vm/η scaling"), (None, None),
        ("HiFT − pairwise learner", "− pairwise learner"), ("HiFT − auxiliary data", "− auxiliary data"),
        ("HiFT + zero-shot prior shrinkage", "+ prior shrinkage")]
paired, yt, yl = [], [], []
y = 0.0
for m, lab in rows:
    if m is None:
        y += 0.55
        continue
    for k, mk, dy in [(1, "o", -0.2), (3, "s", 0.2)]:
        mean, lo_, hi_, frac = fsum.paired_gain("pure_solvent_permeance", m, "HiFT", k)
        paired.append(dict(method=m, k=k, d_mae=mean, lo=lo_, hi=hi_, frac_hift_better=frac))
        col = fs.HERO if lo_ > 0 else fs.GREY_M
        ax_d.plot([lo_, hi_], [y + dy, y + dy], color=col, lw=0.7, solid_capstyle="butt")
        ax_d.plot(mean, y + dy, marker=mk, color=col, ms=2.6, mew=0.7, ls="none",
                  mfc=col if k == 1 else "white")
    yt.append(y)
    yl.append(lab)
    y += 1.0
pd.DataFrame(paired).to_csv(RESULTS / "fewshot_paired_sp.csv", index=False)
ax_d.axvline(0, color=fs.GREY_D, lw=0.5)
ax_d.set_yticks(yt, yl, fontsize=5.4)
ax_d.tick_params(axis="y", length=0, pad=1.5)
ax_d.spines["left"].set_visible(False)
ax_d.set_ylim(y - 0.35, -0.75)
ax_d.set_xlim(-0.06, 0.29)
ax_d.set_xticks([0, 0.1, 0.2])
ax_d.set_xlabel("ΔMAE vs HiFT")
ax_d.text(0.285, -0.55, "baselines", fontsize=5.0, color=fs.GREY_M, style="italic", ha="right", va="center")
ax_d.text(0.285, yt[-3] - 0.75, "ablations", fontsize=5.0, color=fs.GREY_M, style="italic", ha="right",
          va="center")
ax_d.legend(handles=[Line2D([], [], marker="o", ls="none", color=fs.HERO, ms=2.6, label="k = 1"),
                     Line2D([], [], marker="s", ls="none", color=fs.HERO, mfc="white", mew=0.7, ms=2.6, label="k = 3"),
                     Line2D([], [], marker="o", ls="none", color=fs.GREY_M, ms=2.6, label="CI includes 0")],
            loc="upper right", bbox_to_anchor=(1.03, 0.93), fontsize=5.0, handletextpad=0.1, borderaxespad=0.1,
            labelspacing=0.2)

# ---------------------------------------------------------------- f: per-membrane gains
ax_f = S.ax("f", 132.0, YR, 31.0, HR)
S.letter(ax_f, "f", 124.0, 53.5, "Largest gains where zero-shot fails")
ce = fsum.cell_errors(sp_raw)
pm = ce[((ce.method == "Zero-shot GBM") & (ce.k == 0)) | ((ce.method == "HiFT") & (ce.k == 1))]
pm = pm.groupby(["entity", "material", "method"])["abs_err"].mean().unstack("method").reset_index()
lim = (0, 1.7)
ax_f.fill_between(lim, [0, 0], lim, color=fs.HERO_PALE, lw=0, zorder=0, alpha=0.6)
ax_f.plot(lim, lim, color=fs.GREY_D, lw=0.5, ls=(0, (3, 1.5)))
for mat in fs.MATERIAL_ORDER:
    s = pm[pm.material == mat]
    if len(s):
        ax_f.scatter(s["Zero-shot GBM"], s["HiFT"], s=9, color=fs.MATERIAL_COLORS[mat], lw=0.3, edgecolor="white",
                     label=f"{mat.replace(' polymer', '')} ({len(s)})", zorder=3)
imp = (pm["HiFT"] < pm["Zero-shot GBM"]).mean()
ax_f.set_xlim(lim)
ax_f.set_ylim(lim)
ax_f.set_xticks([0, 0.5, 1.0, 1.5])
ax_f.set_yticks([0, 0.5, 1.0, 1.5])
ax_f.set_xlabel("Zero-shot MAE")
ax_f.set_ylabel("HiFT (k = 1) MAE")
ax_f.text(0.97, 0.70, f"{imp:.0%} of {len(pm)}\nmembranes\nimprove", transform=ax_f.transAxes, ha="right",
          va="top", fontsize=5.0, color=fs.HERO, linespacing=fs.LS)
ax_f.legend(loc="upper left", fontsize=5.0, handletextpad=0.0, borderaxespad=0.1, labelspacing=0.15, markerscale=0.9)

# ---------------------------------------------------------------- e: independent verification (forest table)
YE, HE = 112.0, 26.5
ax_e = S.ax("e", 40.0, YE, 70.0, HE)
S.letter(ax_e, "e", 0.0, 102.5, "The one-anchor gain is recovered on tests that did not select the model")


def best_baseline(get):
    vals = {b: get(b) for b in BASELINES}
    vals = {b: v for b, v in vals.items() if v is not None and v[0] is not None}
    b = min(vals, key=lambda kk: vals[kk][0])
    return b, vals[b][0]


sps = sp_sum.set_index(["method", "k"])
pms = pm_sum.set_index(["method", "k"])


def raw_mae(path, method, k, nmin):
    r = pd.read_csv(RESULTS / path, low_memory=False)
    r = r[(r.n_solvents >= nmin) & (r.method == method) & (r.k == k)]
    return fsum.boot_mean(fsum.cell_errors(r), "abs_err")


unt = ind["untouched"]["pure_solvent_permeance"]
UNT, EXT, NST = "fewshot_untouched_pure_solvent_permeance.csv", "fewshot_external_pure_solvent_permeance.csv", \
    "indep_nested_mapped_pure_solvent_permeance.csv"
n_ext = pd.read_csv(RESULTS / EXT, low_memory=False).query("n_solvents >= 2 and method == 'HiFT' and k == 1")
tsets = [
    ("Primary benchmark", f"{int(zs.n_membranes)} membranes, {int(zs.n_papers)} papers", (zs.mae, zs.lo, zs.hi), tuple(sps.loc[("HiFT", 1), ["mae", "lo", "hi"]]),
     best_baseline(lambda b: tuple(sps.loc[(b, 1), ["mae", "lo", "hi"]]))),
    ("Nested configuration selection", "same membranes, inner-fold choice", (zs.mae, zs.lo, zs.hi),
     raw_mae(NST, "Nested-selected HiFT", 1, 4), best_baseline(lambda b: tuple(sps.loc[(b, 1), ["mae", "lo", "hi"]]))),
    ("Membranes unused for selection", f"{unt['n_membranes']} membranes, {unt['n_papers']} papers",
     raw_mae(UNT, "Zero-shot GBM", 0, 2), raw_mae(UNT, "HiFT", 1, 2), best_baseline(lambda b: raw_mae(UNT, b, 1, 2))),
    ("Papers added after the snapshot", f"{n_ext.entity.nunique()} membranes, {n_ext.paper.nunique()} papers",
     raw_mae(EXT, "Zero-shot GBM", 0, 2), raw_mae(EXT, "HiFT", 1, 2), best_baseline(lambda b: raw_mae(EXT, b, 1, 2))),
    ("Solution permeance", f"{int(pms.loc[('HiFT', 1), 'n_membranes'])} membranes, "
                           f"{int(pms.loc[('HiFT', 1), 'n_papers'])} papers",
     tuple(pms.loc[("Zero-shot GBM", 0), ["mae", "lo", "hi"]]), tuple(pms.loc[("HiFT", 1), ["mae", "lo", "hi"]]),
     best_baseline(lambda b: tuple(pms.loc[(b, 1), ["mae", "lo", "hi"]]))),
]
json.dump([dict(test=t[0], zero_shot=t[2], hift=t[3], best_baseline=t[4][0], best_baseline_mae=t[4][1],
                reduction=1 - t[3][0] / t[2][0]) for t in tsets],
          open(RESULTS / "fig4_verification.json", "w"), indent=1)
tx = S.blank(0, YE, 40.0, HE, xlim=(0, 40.0), ylim=(len(tsets) - 0.5, -0.5))
rx = S.blank(111.0, YE, 54.0, HE, xlim=(0, 54.0), ylim=(len(tsets) - 0.5, -0.5))
rx.text(1.0, -0.95, "HiFT vs zero-shot", fontsize=5.0, color=fs.GREY_M, style="italic", ha="left", va="center")
rx.text(22.0, -0.95, "best anchored baseline", fontsize=5.0, color=fs.GREY_M, style="italic", ha="left", va="center")
for i, (name, desc, z0, h1, (bname, bval)) in enumerate(tsets):
    tx.text(0.0, i - 0.17, name, fontsize=5.6, color=fs.INK, ha="left", va="center")
    tx.text(0.0, i + 0.24, desc, fontsize=5.0, color=fs.GREY_M, ha="left", va="center")
    ax_e.plot([z0[1], z0[2]], [i - 0.12, i - 0.12], color=fs.GREY_L, lw=0.8, solid_capstyle="butt", zorder=1)
    ax_e.plot(z0[0], i - 0.12, "o", ms=3.0, mfc="white", mec=fs.GREY_D, mew=0.7, zorder=3)
    ax_e.plot([h1[1], h1[2]], [i + 0.12, i + 0.12], color=fs.HERO, lw=0.8, solid_capstyle="butt", zorder=2, alpha=0.6)
    ax_e.plot(h1[0], i + 0.12, "o", ms=3.3, color=fs.HERO, mec="white", mew=0.3, zorder=4)
    fs.arrow(ax_e, (z0[0] - 0.012, i), (h1[0] + 0.012, i), color=fs.GREY_M, lw=0.5, ms=3.5, z=2)
    bst = fs.METHOD_STYLE[bname]
    ax_e.plot(bval, i + 0.12, marker=bst["marker"], ms=3.0, color=bst["color"], mec=fs.INK, mew=0.3, ls="none", zorder=5)
    red = 1 - h1[0] / z0[0]
    rx.text(1.0, i, f"{h1[0]:.3f} vs {z0[0]:.3f} (−{100 * red:.0f}%)", fontsize=5.4, color=fs.HERO, ha="left",
            va="center")
    blab = fs.METHOD_LABEL.get(bname, bname).replace("V$_m$", "Vm")
    rx.text(22.0, i, f"{blab} {bval:.3f}", fontsize=5.4, color=fs.INK if bval < h1[0] else fs.GREY_D,
            fontweight="bold" if bval < h1[0] else "normal", ha="left", va="center")
ax_e.set_ylim(len(tsets) - 0.5, -0.5)
ax_e.set_yticks([])
ax_e.spines["left"].set_visible(False)
ax_e.axvline(f_floor, color=fs.STUDY, lw=0.6, ls=(0, (3, 1.5)), zorder=0)
ax_e.set_xlim(0.15, 1.05)
ax_e.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax_e.set_xlabel("MAE at k = 1 (log₁₀; 95% paper-bootstrap CI)")
ax_e.legend(handles=[Line2D([], [], marker="o", ls="none", mfc="white", mec=fs.GREY_D, mew=0.7, ms=3.0, label="Zero-shot"),
                     Line2D([], [], marker="o", ls="none", color=fs.HERO, ms=3.3, label="HiFT"),
                     Line2D([], [], marker="v", ls="none", color=fs.METHOD_STYLE["Viscosity scaling"]["color"], ms=3.0,
                            label="Viscosity scaling"),
                     Line2D([], [], marker="P", ls="none", color=fs.METHOD_STYLE["Pairwise transfer GBM"]["color"],
                            ms=3.0, label="Pairwise transfer GBM"),
                     Line2D([], [], marker="h", ls="none", color=fs.METHOD_STYLE["Bayesian level shift"]["color"],
                            ms=3.0, label="Bayesian level shift")],
            loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=3, fontsize=5.0, handletextpad=0.05,
            columnspacing=0.9, borderaxespad=0.15, labelspacing=0.2)

fs.export(S.fig, "Fig4_hift", axes=[ax_b, ax_c, ax_d, ax_f, ax_e], ids=list("bcdfe"),
          row_groups=[["c", "d", "f"]],
          exemptions=[{"panels": ["c", "d", "f"], "checks": ["panel-width", "horizontal-gutter"],
                       "reason": "direct-labelled curves, labelled forest and square scatter need different widths"}])
