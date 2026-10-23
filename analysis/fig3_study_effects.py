"""Figure 3 | Study effects dominate, and inter-laboratory reproducibility caps zero-shot accuracy."""
import json

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import figstyle as fs
import fewshot_summary as fsum
from osn_core import RESULTS, S1, SOLV

plt = fs.plt
SM = fs.FS["small"]
vc = pd.read_csv(RESULTS / "variance_components.csv")
noise = json.load(open(RESULTS / "noise_summary.json"))
nsp = noise["Pure-solvent permeance"]
K = json.load(open(RESULTS / "key_numbers.json", encoding="utf-8"))
sens = K["interlab_sensitivity"]
cells = pd.read_csv(RESULTS / "interlab_cells.csv")
met = pd.read_csv(RESULTS / "benchmark_metrics.csv")
fsm = fsum.summary("pure_solvent_permeance").set_index(["method", "k"])

S = fs.Sheet(104)
XL, WL = 29.0, 57.0          # left column plot band
YA, HA = 11.0, 30.5
YC, HC = 58.5, 36.0

# ---------------------------------------------------------------- a: variance partition
ax_a = S.ax("a", XL, YA, WL, HA)
S.letter(ax_a, "a", 0.0, 1.2, "The source paper carries most of the variance")
comp = [("share_paper", "Paper", fs.STUDY), ("share_membrane", "Membrane within paper", fs.STUDY_MID),
        ("share_residual", "Residual", fs.RESIDUAL)]
ax_a.legend(handles=[Patch(color=c, label=name) for _, name, c in comp], loc="lower left", ncol=3,
            bbox_to_anchor=(-0.01, 1.0), fontsize=SM, handlelength=1.0, handleheight=0.8, columnspacing=1.4,
            borderaxespad=0.15)
order = [("Pure-solvent permeance", "Raw (solvent fixed effects)"), ("Solution permeance", "Raw (solvent fixed effects)"),
         ("Rejection", "Raw (solvent fixed effects)"), ("Pure-solvent permeance", "Unseen-paper LightGBM residual"),
         ("Rejection", "Unseen-paper LightGBM residual")]
ys = [0, 1, 2, 3.55, 4.55]
for y, (ds, basis) in zip(ys, order):
    r = vc[(vc.dataset == ds) & (vc.basis == basis)].iloc[0]
    left = 0
    for key, name, c in comp:
        ax_a.barh(y, r[key], left=left, color=c, height=0.62, lw=0)
        if r[key] > 0.09:
            ax_a.text(left + r[key] / 2, y, f"{r[key]:.0%}", ha="center", va="center", fontsize=SM,
                      color="white" if key == "share_paper" else fs.INK)
        left += r[key]
    ax_a.plot([r.share_paper_lo, r.share_paper_hi], [y - 0.43] * 2, color=fs.INK, lw=0.6, solid_capstyle="butt")
    for xx in (r.share_paper_lo, r.share_paper_hi):
        ax_a.plot([xx, xx], [y - 0.49, y - 0.37], color=fs.INK, lw=0.6)
ax_a.text(0.0, -0.80, "Raw data, solvent as fixed effect", ha="left", va="center", fontsize=SM, color=fs.GREY_M,
          style="italic")
ax_a.text(0.0, 2.75, "Residuals of the zero-shot model on unseen papers", ha="left", va="center", fontsize=SM,
          color=fs.GREY_M, style="italic")
ax_a.set_yticks(ys, [o[0] for o in order], fontsize=5.9)
ax_a.tick_params(axis="y", length=0)
ax_a.spines["left"].set_visible(False)
ax_a.set_ylim(5.0, -1.25)
ax_a.set_xlim(0, 1)
ax_a.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100"])
ax_a.set_xlabel("Share of random-effect variance (%)")

# ---------------------------------------------------------------- c: accuracy ladder against the reproducibility floor
ax_c = S.ax("c", XL, YC, WL, HC)
S.letter(ax_c, "c", 0.0, 50.0, "Random-split error is below the reproducibility floor")
lg = met[met.model == "LightGBM"].set_index("scenario")
rows = [("Random 5-fold", lg.loc["Random 5-fold"], fs.RANDOM), ("Unseen paper", lg.loc["Unseen paper"], fs.PAPER),
        ("Future papers (≥2021)", lg.loc["Future papers (≥2021)"], fs.FUTURE)]
rows = [(lab, r["MAE"], r["MAE_lo"], r["MAE_hi"], col) for lab, r, col in rows]
for k in (0, 1, 3):
    r = fsm.loc[("HiFT", k)]
    rows.append(("HiFT, no anchor" if k == 0 else f"HiFT, {k} anchor" + ("s" if k > 1 else ""), r.mae, r.lo, r.hi,
                 fs.K_COLORS[k]))
floor = nsp["mae_floor_interlab"]
f_lo, f_hi = nsp["interlab_sd_lo"] * np.sqrt(2 / np.pi), nsp["interlab_sd_hi"] * np.sqrt(2 / np.pi)
ax_c.axvspan(f_lo, f_hi, color=fs.STUDY_PALE, lw=0, zorder=0)
ax_c.axvline(floor, color=fs.STUDY, lw=0.8, ls=(0, (3, 1.5)), zorder=1)
ax_c.axvline(nsp["mae_floor_within"], color=fs.GREY_D, lw=0.7, ls=(0, (1, 1.2)), zorder=1)
ypos = [0, 1, 2, 3.55, 4.55, 5.55]
for y, (lab, m, lo, hi, col) in zip(ypos, rows):
    ax_c.barh(y, m, color=col, height=0.6, lw=0, zorder=2)
    ax_c.plot([lo, hi], [y, y], color=fs.INK, lw=0.6, zorder=3, solid_capstyle="butt")
    ax_c.text(1.015, y, f"{m:.2f}", va="center", ha="left", fontsize=SM, color=fs.INK,
              transform=ax_c.get_yaxis_transform(), clip_on=False)
ax_c.text(-0.02, -0.80, "Zero-shot, all papers", ha="right", va="center", fontsize=SM, color=fs.GREY_M,
          style="italic", transform=ax_c.get_yaxis_transform())
ax_c.text(-0.02, 2.75, f"Few-shot set, {K['fewshot_sp']['HiFT | k=1']['n_membranes']} membranes", ha="right", va="center", fontsize=SM, color=fs.GREY_M,
          style="italic", transform=ax_c.get_yaxis_transform())
ax_c.text(f_lo - 0.025, -0.85, "same-lab\ninformation", ha="right", va="center", fontsize=5.0,
          color=fs.STUDY, linespacing=fs.LS)
ax_c.text(f_hi + 0.025, -0.85, "new-lab\nfloor", ha="left", va="center", fontsize=5.0, color=fs.STUDY,
          linespacing=fs.LS)
ax_c.set_yticks(ypos, [r[0] for r in rows], fontsize=5.9)
ax_c.tick_params(axis="y", length=0)
ax_c.spines["left"].set_visible(False)
ax_c.set_ylim(6.1, -1.55)
ax_c.set_xlim(0, 0.92)
ax_c.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
ax_c.set_xlabel("MAE (log₁₀ permeance)")

# ---------------------------------------------------------------- b: same membrane and solvent, different laboratories
ax_b = S.ax("b", 124.5, YA, 35.5, YC + HC - YA)
S.letter(ax_b, "b", 92.0, 1.2, "Same membrane, solvent; different labs")
c = cells[cells.dataset == "Pure-solvent permeance"].copy()
c["ref_norm"] = c["report.reference"].astype(str).str.lower().str.replace(r"^(https?://)?(dx\.)?(doi\.org/)", "",
                                                                         regex=True)
c = c.drop_duplicates(["commercialName", S1, "ref_norm"])
c["label"] = c["commercialName"].str.replace("Inopor nano 0,9 nm", "Inopor 0.9 nm") + " · " + c[S1].map(SOLV["short"])
st = c.groupby("label").agg(mean=("y", "mean"), lo=("y", "min"), hi=("y", "max"), n=("y", "size"),
                            nuniq=("y", "nunique"), name=("commercialName", "first")).sort_values("mean")
for i, (lab, s) in enumerate(st.iterrows()):
    col = fs.MATERIAL_COLORS["Ceramic"] if "Inopor" in s["name"] else "#7A7A7A"
    same = s.nuniq == 1
    ax_b.plot([s.lo, s.hi], [i, i], color=col, lw=2.0, alpha=0.30, solid_capstyle="round", zorder=1)
    v = c.loc[c.label == lab, "y"].values
    ax_b.scatter(v, np.full(len(v), i), s=11, color="white" if same else col, edgecolor=col, lw=0.7, zorder=3)
    ax_b.text(1.03, i, f"{int(s.n)}" + (" =" if same else ""), transform=ax_b.get_yaxis_transform(), ha="left",
              va="center", fontsize=5.0, color=fs.GREY_D, clip_on=False)
ax_b.text(1.03, len(st) - 0.15, "labs", transform=ax_b.get_yaxis_transform(), ha="left", va="bottom", fontsize=5.0,
          color=fs.GREY_D)
ax_b.set_yticks(range(len(st)), st.index, fontsize=5.5)
ax_b.tick_params(axis="y", length=0)
ax_b.spines["left"].set_visible(False)
ax_b.set_ylim(-0.7, len(st) + 3.6)
ax_b.set_xlim(-1.15, 2.15)
ax_b.set_xticks([-1, 0, 1, 2], ["0.1", "1", "10", "100"])
ax_b.set_xlabel("Reported permeance (L m⁻² h⁻¹ bar⁻¹)")
sd0 = nsp["interlab_sd"]
yb = len(st) + 3.0
ax_b.plot([0.15, 0.15 + sd0], [yb, yb], color=fs.STUDY, lw=1.6, solid_capstyle="butt")
for xx in (0.15, 0.15 + sd0):
    ax_b.plot([xx, xx], [yb - 0.35, yb + 0.35], color=fs.STUDY, lw=0.7)
ax_b.text(0.15 + sd0 + 0.08, yb, f"1 SD = {sd0:.2f} log (×{10 ** sd0:.1f})", ha="left", va="center", fontsize=SM,
          color=fs.STUDY)
ax_b.text(0.15, yb - 0.75, f"pooled over {sens['primary']['cells']} pairs\n"
           f"{sens['merged_refs_no_identical']['sd']:.2f} without identical-value\npairs (=)", ha="left", va="top",
           fontsize=5.0, color=fs.GREY_D, linespacing=fs.LS)
ax_b.legend(handles=[Line2D([], [], marker="o", ls="none", color="#7A7A7A", ms=3.0, label="Polymeric"),
                     Line2D([], [], marker="o", ls="none", color=fs.MATERIAL_COLORS["Ceramic"], ms=3.0,
                            label="Ceramic")],
            loc="lower right", bbox_to_anchor=(1.0, 0.005), fontsize=5.0, handletextpad=0.1, borderaxespad=0.1,
            labelspacing=0.25)

fs.export(S.fig, "Fig3_study_effects", axes=[ax_a, ax_b, ax_c], ids=list("abc"), column_groups=[["a", "c"]],
          exemptions=[{"panels": ["b"], "checks": ["row"], "reason": "b spans both rows of the left column"}])
