"""Supporting Information figures S1-S11 on the same modular layout and semantic palette as the main figures."""
import json
import sys

import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import figstyle as fs
import fewshot_summary as fsum
from osn_core import DATA, RESULTS, S1, SOLV

plt = fs.plt
SM = fs.FS["small"]
K = json.load(open(RESULTS / "key_numbers.json", encoding="utf-8"))
NSP = K["noise"]["Pure-solvent permeance"]
HEX = mcolors.LinearSegmentedColormap.from_list("ink", ["#E3ECF5", "#8FB4D6", "#2E6DA4", "#0B2A4A"])


def row(S, titles, y=7.5, h=36.0, left=11.0, right=1.5, gutter=14.0, widths=None, lab_dx=10.5, ids="abcdef",
        letter_y=1.2):
    """Equal-height panels in one row; returns the axes. Letters sit lab_dx mm left of each plot area."""
    n = len(titles)
    if widths is None:
        widths = [(S.w - left - right - (n - 1) * gutter) / n] * n
    axes, x = [], left
    for i, (t, w) in enumerate(zip(titles, widths)):
        ax = S.ax(ids[i], x, y, w, h)
        S.letter(ax, ids[i], max(x - lab_dx, 0.0), letter_y, t)
        axes.append(ax)
        x += w + gutter
    return axes


def top_legend(ax, handles, ncol=None, dy=1.0):
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, dy), ncol=ncol or len(handles), fontsize=SM,
              handlelength=1.6, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.2)


def proto_handles(protos=("Random 5-fold", "Unseen paper"), marker="o"):
    return [Line2D([], [], color=fs.PROTOCOL_COLORS[p], marker=marker, ms=3, lw=1.0, label=p) for p in protos]


def s1_years():
    ya = pd.read_csv(DATA / "year_audit.csv")
    sp = pd.read_csv(DATA / "pure_solvent_permeance_domain.csv", low_memory=False)
    S = fs.Sheet(58)
    a, b = row(S, [f"Per-paper year audit ({len(ya)} papers)", "Effect on the temporal split"], y=10.0, h=36.0, gutter=17.0)
    cmap = {"recorded": fs.GREY_M, "decoded epoch-ms": fs.PAPER, "DOI (curation timestamp)": "#E08214",
            "DOI (recorded year inconsistent)": fs.STUDY, "DOI": fs.FUTURE}
    rng = np.random.default_rng(0)
    for src, c in cmap.items():
        s = ya[(ya.year_source == src) & ya.year.notna() & ya.recorded_year.notna()]
        if len(s):
            a.scatter(s.recorded_year + rng.uniform(-0.25, 0.25, len(s)), s.year + rng.uniform(-0.25, 0.25, len(s)),
                      s=7, color=c, lw=0, alpha=0.85, label=f"{src} ({len(s)})")
    a.plot([1998, 2026], [1998, 2026], color=fs.GREY_D, lw=0.5, ls=(0, (3, 1.5)))
    a.set_xlim(1998, 2026)
    a.set_ylim(1998, 2026)
    a.set_xticks([2000, 2010, 2020])
    a.set_yticks([2000, 2010, 2020])
    a.set_xlabel("Recorded publication year (decoded)")
    a.set_ylabel("Audited publication year")
    a.legend(loc="upper left", fontsize=5.0, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.2, markerscale=1.2)
    rec = pd.to_datetime(sp["report.publicationDate"], errors="coerce").dt.year
    rec = rec.where(rec != 1970)
    bins = np.arange(1999.5, 2026.5, 1)
    b.hist(rec.dropna(), bins=bins, color=fs.GREY_L, label="recorded", lw=0)
    b.hist(sp["year"].dropna(), bins=bins, histtype="step", color=fs.INK, lw=1.0, label="audited")
    b.axvline(2020.5, color=fs.FUTURE, lw=0.7, ls=(0, (3, 1.5)))
    b.set_xlabel("Publication year")
    b.set_ylabel("Pure-solvent permeance records")
    b.set_xlim(1999.5, 2025.5)
    b.text(0.02, 0.97, f"records dated ≥2021: {int((rec >= 2021).sum())} recorded, {int((sp['year'] >= 2021).sum())} audited\n"
                       f"records without year: {int(rec.isna().sum())} recorded, {int(sp['year'].isna().sum())} audited",
           transform=b.transAxes, ha="left", va="top", fontsize=SM, linespacing=fs.LS)
    b.legend(loc="center left", bbox_to_anchor=(0.0, 0.55), fontsize=SM)
    fs.export(S.fig, "FigS1_year_audit", axes=[a, b], ids=["a", "b"], row_groups=[["a", "b"]])


def s2_learning():
    lc = pd.read_csv(RESULTS / "si_learning_curve.csv")
    S = fs.Sheet(56)
    a, b = row(S, ["More training data helps slowly", "Error versus training-set size"], y=11.0, h=34.0, gutter=17.0)
    for prot in ("Random 5-fold", "Unseen paper"):
        c = fs.PROTOCOL_COLORS[prot]
        g = lc[lc.protocol == prot].groupby("frac").agg(R2=("R2", "mean"), R2sd=("R2", "std"), MAE=("MAE", "mean"),
                                                        MAEsd=("MAE", "std"))
        a.errorbar(100 * g.index, g.R2, yerr=g.R2sd, color=c, marker="o", ms=2.8, lw=1.0, elinewidth=0.6, capsize=0)
        b.errorbar(100 * g.index, g.MAE, yerr=g.MAEsd, color=c, marker="o", ms=2.8, lw=1.0, elinewidth=0.6, capsize=0)
    a.axhline(NSP["r2_ceiling_unseen_paper"], color=fs.STUDY, ls=(0, (3, 1.5)), lw=0.8)
    b.axhline(NSP["mae_floor_interlab"], color=fs.STUDY, ls=(0, (3, 1.5)), lw=0.8)
    for ax, yl in [(a, "R² (log₁₀ permeance)"), (b, "MAE (log₁₀ units)")]:
        ax.set_xlabel("Training data used (% of training fold)")
        ax.set_ylabel(yl)
        ax.set_xticks([20, 40, 60, 80, 100])
        ax.set_xlim(12, 108)
    a.set_ylim(-0.75, 1.05)
    b.set_ylim(0.15, 0.85)
    top_legend(a, proto_handles() + [Line2D([], [], color=fs.STUDY, ls=(0, (3, 1.5)), lw=0.8,
                                            label="Reproducibility ceiling / inter-lab floor")])
    fs.export(S.fig, "FigS2_learning_curve", axes=[a, b], ids=["a", "b"], row_groups=[["a", "b"]])


def seed_r2(scen, model):
    oof = pd.read_csv(RESULTS / "benchmark_oof.csv", low_memory=False)
    o = oof[(oof.scenario == scen) & (oof.model == model)]
    v = [1 - np.sum((g.y - g.pred) ** 2) / np.sum((g.y - g.y.mean()) ** 2) for _, g in o.groupby("seed")]
    return np.mean(v), np.std(v, ddof=1)


def s3_robustness():
    cr = pd.read_csv(RESULTS / "si_cleaning_rules.csv")
    mlp = pd.read_csv(RESULTS / "si_mlp.csv")
    nt = pd.read_csv(RESULTS / "si_nested_tuning.csv")
    S = fs.Sheet(56)
    a, b, c = row(S, ["Original cleaning rules", "Model class", "Nested hyperparameter tuning"], y=11.0, h=32.0,
                  gutter=15.0)
    protos = ("Random 5-fold", "Unseen paper")

    def bars(ax, groups, vals, errs, labels):
        for i in range(len(groups)):
            for j, prot in enumerate(protos):
                v, e = vals[i][j], errs[i][j]
                ax.bar(i + (j - 0.5) * 0.36, v, width=0.34, color=fs.PROTOCOL_COLORS[prot], lw=0)
                if e is not None:
                    ax.plot([i + (j - 0.5) * 0.36] * 2, [v - e, v + e], color=fs.INK, lw=0.6)
                ax.text(i + (j - 0.5) * 0.36, max(v, 0) + (e or 0) + 0.03, fs.num(v), ha="center", va="bottom",
                        fontsize=5.0)
        ax.axhline(0, color=fs.GREY_D, lw=0.5)
        ax.set_xticks(range(len(groups)), labels, fontsize=5.8, linespacing=1.15)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(-0.1, 1.0)
        ax.set_ylabel("R² (log₁₀ permeance)")

    g = cr.groupby(["dataset", "protocol"])["R2"].agg(["mean", "std"])
    ds = ["All records", "Original cleaning rules"]
    bars(a, ds, [[g.loc[(d, p), "mean"] for p in protos] for d in ds], [[g.loc[(d, p), "std"] for p in protos] for d in ds],
         ["All\nrecords", "Original\ncleaning rules"])
    mv = mlp.groupby("protocol")["R2"].agg(["mean", "std"])
    vals = [[seed_r2(p, "LightGBM")[0] for p in protos], [seed_r2(p, "Random forest")[0] for p in protos],
            [mv.loc[p, "mean"] for p in protos]]
    errs = [[seed_r2(p, "LightGBM")[1] for p in protos], [seed_r2(p, "Random forest")[1] for p in protos],
            [mv.loc[p, "std"] for p in protos]]
    bars(b, ["LightGBM", "RF", "MLP"], vals, errs, ["LightGBM", "Random\nforest", "MLP"])
    tv = nt.set_index(["protocol", "tuning"])["R2"]
    tunings = ["Grouped inner CV", "Row-level inner CV"]
    for i, t in enumerate(tunings):
        for j, prot in enumerate(protos):
            v = tv[(prot, t)]
            c.bar(i + (j - 0.5) * 0.36, v, width=0.34, color=fs.PROTOCOL_COLORS[prot], lw=0)
            c.text(i + (j - 0.5) * 0.36, v + 0.03, fs.num(v), ha="center", va="bottom", fontsize=5.0)
    c.set_xticks([0, 1], ["Grouped\ninner folds", "Record-level\ninner folds"], fontsize=5.8, linespacing=1.15)
    c.tick_params(axis="x", length=0)
    c.set_ylim(-0.1, 1.0)
    c.axhline(0, color=fs.GREY_D, lw=0.5)
    c.set_ylabel("Outer-fold R²")
    top_legend(a, [Patch(color=fs.PROTOCOL_COLORS[p], label=p) for p in protos])
    fs.export(S.fig, "FigS3_robustness", axes=[a, b, c], ids=list("abc"), row_groups=[["a", "b", "c"]])


def s4_per_class():
    pc = pd.read_csv(RESULTS / "si_per_class_heldout.csv")
    S = fs.Sheet(58)
    a, b = row(S, ["Error when the class is held out", "Systematic bias for unseen classes"], y=11.0, h=31.0,
               gutter=17.0)
    order = [m for m in fs.MATERIAL_ORDER if m in set(pc.material)]
    x = np.arange(len(order))
    for j, (m, col) in enumerate([("LightGBM", fs.INK), ("Random forest", "#8C8C8C")]):
        g = pc[pc.model == m].set_index("material").reindex(order)
        a.bar(x + (j - 0.5) * 0.36, g.MAE, width=0.34, color=col, lw=0, label=m)
        b.bar(x + (j - 0.5) * 0.36, g.bias, width=0.34, color=col, lw=0, label=m)
    n = pc[pc.model == "LightGBM"].set_index("material").reindex(order)["n"]
    short = {"Commercial polymer": "Commercial", "2D/framework": "2D/frame-\nwork", "Glassy ISA": "Glassy\nISA"}
    for ax in (a, b):
        ax.set_xticks(x, [f"{short.get(o, o)}\n({int(k)})" for o, k in zip(order, n)], fontsize=5.4, linespacing=fs.LS)
        ax.tick_params(axis="x", length=0)
    b.axhline(0, color=fs.GREY_D, lw=0.5)
    a.axhline(NSP["mae_floor_interlab"], color=fs.STUDY, ls=(0, (3, 1.5)), lw=0.8)
    a.set_ylabel("MAE (log₁₀)")
    b.set_ylabel("Mean bias, predicted − measured (log₁₀)")
    top_legend(a, [Patch(color=fs.INK, label="LightGBM"), Patch(color="#8C8C8C", label="Random forest"),
                   Line2D([], [], color=fs.STUDY, ls=(0, (3, 1.5)), lw=0.8, label="Inter-lab floor")])
    fs.export(S.fig, "FigS4_per_class", axes=[a, b], ids=["a", "b"], row_groups=[["a", "b"]])


def s5_rejection():
    S = fs.Sheet(56, width_mm=120)
    rr, ru = K["rejection"]["Random 5-fold | LightGBM"], K["rejection"]["Unseen paper | LightGBM"]
    a, b = row(S, [f"Random 5-fold: R² {rr['R2']:.2f}, MAE {rr['MAE_pp']:.1f} pp",
                   f"Unseen paper: R² {ru['R2']:.2f}, MAE {ru['MAE_pp']:.1f} pp"], y=7.5, h=38.0, left=11.0,
               right=14.0, gutter=9.0,
               widths=[38.0, 38.0], lab_dx=10.5)
    norm_ = mcolors.LogNorm(vmin=1, vmax=150)
    for ax, f, t in [(a, "rejection_oof_random.csv", "Random 5-fold"), (b, "rejection_oof_unseen_paper.csv", "Unseen paper")]:
        o = pd.read_csv(RESULTS / f)
        o = o[(o.model == "LightGBM") & (o.seed == 0)]
        hb = ax.hexbin(100 * o.y, 100 * o.pred, gridsize=26, extent=(0, 100, 0, 100), cmap=HEX, norm=norm_,
                       mincnt=1, lw=0.1, edgecolors="face")
        ax.plot([0, 100], [0, 100], color=fs.INK, lw=0.5, ls=(0, (3, 1.5)))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0, 50, 100])
        ax.set_xlabel("Measured rejection (%)")
    a.set_ylabel("Predicted rejection (%)")
    b.set_yticklabels([])
    cax = b.inset_axes([1.05, 0.15, 0.05, 0.7])
    cb = S.fig.colorbar(hb, cax=cax)
    cb.set_ticks([1, 10, 100])
    cb.set_ticklabels(["1", "10", "100"])
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=5.0, width=0.4, length=1.4)
    cb.outline.set_linewidth(0.4)
    cax.set_title("count", fontsize=5.0, pad=2.4, color=fs.GREY_D)
    fs.export(S.fig, "FigS5_rejection", axes=[a, b], ids=["a", "b"], row_groups=[["a", "b"]])


def s6_interlab_pm():
    c = pd.read_csv(RESULTS / "interlab_cells.csv")
    c = c[c.dataset == "Solution permeance"].copy()
    c["rn"] = c["report.reference"].astype(str).str.lower().str.replace(r"^(https?://)?(dx\.)?(doi\.org/)", "", regex=True)
    c = c.drop_duplicates(["commercialName", S1, "rn"])
    c["label"] = c["commercialName"].str.replace("Inopor nano 0,9 nm", "Inopor 0.9 nm") + " · " + c[S1].map(SOLV["short"])
    st = c.groupby("label").agg(mean=("y", "mean"), lo=("y", "min"), hi=("y", "max"), n=("y", "size"),
                                name=("commercialName", "first")).sort_values("mean")
    n = K["noise"]["Solution permeance"]
    S = fs.Sheet(92, width_mm=100)
    ax = S.ax("a", 37.0, 9.0, 55.0, 72.0)
    S.letter(ax, "a", 0.0, 1.2, f"Inter-lab SD {n['interlab_sd']:.2f} log ({n['interlab_cells']} pairs)")
    for i, (lab, s) in enumerate(st.iterrows()):
        col = fs.MATERIAL_COLORS["Ceramic"] if "Inopor" in s["name"] else "#7A7A7A"
        ax.plot([s.lo, s.hi], [i, i], color=col, lw=2.0, alpha=0.30, solid_capstyle="round")
        v = c.loc[c.label == lab, "y"].values
        ax.scatter(v, np.full(len(v), i), s=10, color=col, zorder=3, lw=0.3, edgecolor="white")
        ax.text(1.03, i, f"{int(s.n)}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=5.0,
                color=fs.GREY_D, clip_on=False)
    ax.text(1.03, len(st) - 0.3, "labs", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=5.0,
            color=fs.GREY_D)
    ax.set_yticks(range(len(st)), st.index, fontsize=5.4)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.7, len(st) - 0.3)
    ax.set_xlabel("Reported solution permeance (log₁₀ L m⁻² h⁻¹ bar⁻¹)")
    fs.export(S.fig, "FigS6_interlab_solution", axes=[ax], ids=["a"])


def s7_fewshot():
    s = pd.read_csv(RESULTS / "fewshot_summary_sp.csv")
    pcm = pd.read_csv(RESULTS / "si_fewshot_per_class.csv")
    ew = pd.read_csv(RESULTS / "si_ensemble_weight.csv")
    S = fs.Sheet(60)
    a = S.ax("a", 10.0, 7.5, 32.0, 36.0)
    S.letter(a, "a", 0.0, 1.2, "All methods and ablations")
    b = S.ax("b", 87.0, 7.5, 42.0, 36.0)
    S.letter(b, "b", 77.0, 1.2, "By material class")
    c = S.ax("c", 140.0, 7.5, 23.5, 36.0)
    S.letter(c, "c", 130.5, 1.2, "Ensemble weight")
    meths = ["Zero-shot GBM", "Vm/η scaling", "Viscosity scaling", "Per-membrane η fit", "Additive solvent offset",
             "GBM + mean shift", "Bayesian level shift", "Linear random-coefficient", "Pairwise transfer GBM",
             "HiFT − pairwise learner", "HiFT − auxiliary data", "HiFT + zero-shot prior shrinkage", "HiFT"]
    handles = []
    for m in meths:
        st = fs.METHOD_STYLE[m]
        g = s[s.method == m].sort_values("k")
        if m == "Zero-shot GBM":
            g = pd.concat([g, g.assign(k=3)])
        a.plot(g.k, g.mae, color=st["color"], ls=st["ls"], marker=st["marker"], ms=2.4, lw=st["lw"], zorder=st["z"])
        handles.append(Line2D([], [], color=st["color"], ls=st["ls"], marker=st["marker"], ms=2.4, lw=st["lw"],
                              label=fs.METHOD_LABEL.get(m, m).replace("V$_m$", "Vm")))
    a.set_xticks([0, 1, 2, 3])
    a.set_ylim(0.28, 0.75)
    a.set_xlabel("Anchors (k)")
    a.set_ylabel("MAE (log₁₀)")
    lg = S.blank(44.0, 7.5, 31.0, 36.0)
    lg.legend(handles=handles, loc="center left", fontsize=5.0, handlelength=2.0, labelspacing=0.35,
              borderaxespad=0.0)
    order = [m for m in fs.MATERIAL_ORDER if m in set(pcm.material)]
    x = np.arange(len(order))
    for j, (meth, k, col, lab) in enumerate([("Zero-shot GBM", 0, fs.GREY_D, "Zero-shot"),
                                             ("HiFT", 1, fs.K_COLORS[1], "HiFT, k = 1"), ("HiFT", 3, fs.K_COLORS[3], "HiFT, k = 3")]):
        g = pcm[(pcm.method == meth) & (pcm.k == k)].set_index("material").reindex(order)
        b.bar(x + (j - 1) * 0.26, g.MAE, width=0.25, color=col, lw=0, label=lab)
    nn = pcm[(pcm.method == "HiFT") & (pcm.k == 1)].set_index("material").reindex(order)["n_membranes"]
    short = {"Commercial polymer": "Comm.", "Glassy ISA": "Glassy", "TFC/TFN": "TFC"}
    b.set_xticks(x, [f"{short.get(o, o)}\n({int(k)})" for o, k in zip(order, nn)], fontsize=5.4, linespacing=fs.LS)
    b.tick_params(axis="x", length=0)
    b.set_ylabel("MAE (log₁₀)")
    b.legend(loc="upper right", fontsize=5.0, borderaxespad=0.1, handlelength=1.0)
    e = ew[ew.dataset == "pure_solvent_permeance"]
    for k in (1, 2, 3):
        g = e[e.k == k].sort_values("w_backfit")
        c.plot(g.w_backfit, g.MAE, color=fs.K_COLORS[k], marker="o", ms=2.4, lw=1.0, label=f"k = {k}")
    c.axvline(0.5, color=fs.GREY_D, lw=0.6, ls=(0, (1, 1.4)))
    c.set_xlabel("Backfitted-learner weight")
    c.set_ylabel("MAE (log₁₀)")
    c.legend(loc="upper left", fontsize=5.0, borderaxespad=0.1, handlelength=1.2)
    fs.export(S.fig, "FigS7_fewshot_extra", axes=[a, b, c], ids=list("abc"), row_groups=[["a", "b", "c"]],
              exemptions=[{"panels": ["a", "b", "c"], "checks": ["panel-width", "horizontal-gutter"],
                           "reason": "panel a carries a dedicated legend module"}])


def s8_reliability():
    rel = pd.read_csv(RESULTS / "si_reliability.csv")
    cov = pd.read_csv(RESULTS / "hift_coverage_pm.csv")
    S = fs.Sheet(56)
    a, b = row(S, ["Error versus predicted uncertainty", "Calibration on solution permeance"], y=7.5, h=38.0,
               gutter=17.0)
    lim = (0, max(rel.sd_mid.max(), rel.rmse.max()) * 1.1)
    a.plot(lim, lim, color=fs.GREY_D, lw=0.5, ls=(0, (3, 1.5)))
    a.scatter(rel.sd_mid, rel.rmse, s=rel.n / rel.n.max() * 34 + 6, color=fs.HERO, lw=0.3, edgecolor="white")
    a.set_xlim(lim)
    a.set_ylim(lim)
    a.set_xlabel("Predicted SD (HiFT, binned)")
    a.set_ylabel("Observed RMSE")
    b.plot([0.45, 1], [0.45, 1], color=fs.GREY_D, lw=0.5, ls=(0, (1, 1.4)))
    for k, col in fs.K_COLORS.items():
        g = cov[cov.k == k]
        b.plot(g.level, g.coverage, color=col, marker="o", ms=2.6, lw=1.0, label=f"k = {k}")
    b.set_xlim(0.45, 1.0)
    b.set_ylim(0.4, 1.02)
    b.set_xlabel("Nominal coverage")
    b.set_ylabel("Empirical coverage")
    b.legend(loc="lower right", fontsize=5.0, borderaxespad=0.1)
    fs.export(S.fig, "FigS8_reliability", axes=[a, b], ids=["a", "b"], row_groups=[["a", "b"]])


def s9_fingerprints():
    fp = pd.read_csv(RESULTS / "fingerprints.csv")
    S = fs.Sheet(64)
    a, b = row(S, ["Second fingerprint plane", "Partial pooling stabilizes slopes"], y=13.0, h=36.0, gutter=17.0)
    for mat in fs.MATERIAL_ORDER:
        s = fp[fp.material == mat]
        if len(s):
            a.scatter(s.b_log_Vm, 10 * s.b_dH, s=9, color=fs.MATERIAL_COLORS[mat], lw=0.3, edgecolor="white",
                      label=f"{mat.replace(' polymer', '')} ({len(s)})")
    a.axhline(0, color=fs.GREY_L, lw=0.5)
    a.set_xlabel("Molar-volume exponent  ∂log P / ∂log Vm")
    a.set_ylabel("H-bond affinity (per 10 MPa½ δH)")
    a.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=5.0, handletextpad=0.0, borderaxespad=0.15,
             labelspacing=0.2, ncol=3, columnspacing=0.8)
    s = fp[fp.ols_log_eta.notna()]
    lim = (min(s.ols_log_eta.min(), s.b_log_eta.min()) - 0.2, max(s.ols_log_eta.max(), s.b_log_eta.max()) + 0.2)
    b.plot(lim, lim, color=fs.GREY_D, lw=0.5, ls=(0, (3, 1.5)))
    b.scatter(s.ols_log_eta, s.b_log_eta, s=10, color=[fs.MATERIAL_COLORS.get(m, fs.GREY_M) for m in s.material],
              lw=0.3, edgecolor="white")
    b.set_xlim(lim)
    b.set_ylim(lim)
    b.set_xlabel("Unpooled least-squares viscosity exponent")
    b.set_ylabel("Hierarchical posterior viscosity exponent")
    b.text(0.03, 0.97, f"{len(s)} membranes, ≥6 solvents", transform=b.transAxes, ha="left", va="top", fontsize=SM)
    fs.export(S.fig, "FigS9_fingerprints", axes=[a, b], ids=["a", "b"], row_groups=[["a", "b"]])


def s10_exemplars():
    ex = pd.read_csv(RESULTS / "exemplar_series.csv")
    S = fs.Sheet(72)
    axs = row(S, ["Cross-linked PBI series", "PDMS-coated vs dense PI", "Ceramic: native vs benzyl-grafted"], y=7.5,
              h=30.0, gutter=13.0)
    specs = ["pbi_series", "pdms_vs_pi", "ceramic_grafting"]
    for ax, ser in zip(axs, specs):
        t = ex[ex.series == ser]
        order = t.groupby("short")["dP"].first().sort_values().index.tolist()
        names = t.name.unique()
        cm = mcolors.LinearSegmentedColormap.from_list("seq", ["#BDBDBD", "#2E6DA4", "#0B2A4A"])
        for i, nm in enumerate(names):
            g = t[t.name == nm].set_index("short").reindex(order)
            ax.plot(range(len(order)), g.y, marker="o", ms=2.4, lw=0.9, color=cm(i / max(1, len(names) - 1)), label=nm)
        ax.set_xticks(range(len(order)), order, rotation=90, fontsize=5.4)
        ax.tick_params(axis="x", length=0, pad=1.2)
        ax.set_ylabel("log₁₀ P")
        ax.legend(fontsize=5.0, loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=2, handlelength=1.4,
                  columnspacing=0.6, labelspacing=0.2)
    fs.export(S.fig, "FigS10_exemplars", axes=axs, ids=["a", "b", "c"], row_groups=[["a", "b", "c"]])


def s11_independent():
    """Frozen HiFT on independent test sets (cell-level aggregation, as in the text) and nested selection."""
    S = fs.Sheet(110)
    meths = [("Zero-shot GBM", None), ("Viscosity scaling", None), ("Bayesian level shift", None),
             ("Pairwise transfer GBM", None), ("HiFT", None)]

    def summ(path, nmin, kmax):
        r = pd.read_csv(RESULTS / path, low_memory=False)
        r = r[(r.n_solvents >= nmin) & (r.k <= kmax)]
        rows = []
        for (m, k), g in fsum.cell_errors(r).groupby(["method", "k"]):
            mean, lo, hi = fsum.boot_mean(g, "abs_err")
            rows.append(dict(method=m, k=k, mae=mean, lo=lo, hi=hi, n_membranes=g.entity.nunique(),
                             n_papers=g.paper.nunique()))
        return pd.DataFrame(rows)

    sets = [("Primary benchmark", pd.read_csv(RESULTS / "fewshot_summary_sp.csv"), 3),
            ("Membranes unused for selection", summ("fewshot_untouched_pure_solvent_permeance.csv", 2, 2), 2),
            ("Papers added after the snapshot", summ("fewshot_external_pure_solvent_permeance.csv", 2, 2), 2)]
    axs = row(S, [t[0] for t in sets], y=13.0, h=33.0, gutter=12.0, left=11.0)
    for ax, (name, df, kmax) in zip(axs, sets):
        for m, _ in meths:
            st = fs.METHOD_STYLE[m]
            g = df[(df.method == m) & (df.k <= kmax)].sort_values("k")
            if m == "Zero-shot GBM":
                g = pd.concat([g[g.k == 0], g[g.k == 0].assign(k=kmax)])
            if m == "HiFT":
                ax.fill_between(g.k, g.lo, g.hi, color=fs.HERO, alpha=0.10, lw=0)
            ax.plot(g.k, g.mae, color=st["color"], ls=st["ls"], marker=st["marker"], ms=2.6, lw=st["lw"], zorder=st["z"])
        h1 = df[(df.method == "HiFT") & (df.k == 1)].iloc[0]
        ax.text(0.03, 0.03, f"{int(h1.n_membranes)} membranes, {int(h1.n_papers)} papers", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=5.0, color=fs.GREY_D)
        ax.set_xticks(range(kmax + 1))
        ax.set_xlim(-0.2, kmax + 0.2)
        ax.set_ylim(0.15, 1.05)
        ax.set_xlabel("Anchors (k)")
    axs[0].set_ylabel("MAE (log₁₀; cell level)")
    for ax in axs[1:]:
        ax.set_yticklabels([])
    top_legend(axs[0], [Line2D([], [], color=fs.METHOD_STYLE[m]["color"], ls=fs.METHOD_STYLE[m]["ls"],
                               marker=fs.METHOD_STYLE[m]["marker"], ms=2.6, lw=fs.METHOD_STYLE[m]["lw"],
                               label=fs.METHOD_LABEL.get(m, m)) for m, _ in meths], ncol=5, dy=1.0)

    mapped = pd.read_csv(RESULTS / "indep_nested_mapped_pure_solvent_permeance.csv", low_memory=False)
    d, e = row(S, ["Nested selection of the specified grid", "Outer folds: frozen versus nested choice"], y=68.0,
               h=32.0, gutter=17.0, ids="de", letter_y=61.5)
    for meth, col, ls, mk, lab in [("Frozen HiFT", fs.HERO, "-", "o", "Frozen"),
                                   ("Nested-selected HiFT", fs.PAPER, (0, (3, 1.5)), "s", "Nested"),
                                   ("Oracle (test-tuned)", fs.GREY_M, (0, (1, 1.2)), "D", "Oracle")]:
        g = mapped[(mapped.method == meth) & (mapped.n_solvents >= 4)]
        ce = g.groupby(["seed", "k", "entity", "target"], as_index=False)["abs_err"].mean()
        m = ce.groupby("k")["abs_err"].mean()
        d.plot(m.index, m.values, color=col, ls=ls, marker=mk, ms=2.8, lw=1.1, label=lab)
    zs = pd.read_csv(RESULTS / "fewshot_summary_sp.csv").query("method == 'Zero-shot GBM' and k == 0").mae.iloc[0]
    d.axhline(zs, color=fs.GREY_D, lw=0.7, ls=(0, (4, 2)))
    d.text(3.05, zs - 0.01, "zero-shot", fontsize=5.0, color=fs.GREY_D, ha="right", va="top")
    d.set_xticks([1, 2, 3])
    d.set_xlim(0.8, 3.2)
    d.set_ylim(0.28, 0.68)
    d.set_xlabel("Anchors (k)")
    d.set_ylabel("MAE (log₁₀; cell level)")
    d.legend(loc="center right", fontsize=5.0, borderaxespad=0.1)
    ch = pd.read_csv(RESULTS / "indep_nested_choices_pure_solvent_permeance.csv")
    same = ch.selected == K["independent"]["frozen"]
    for _, r in ch.iterrows():
        e.plot([0, 1], [r.frozen_outer_mae, r.selected_outer_mae], color=fs.GREY_L, lw=0.6, zorder=1)
    e.scatter(np.zeros(len(ch)), ch.frozen_outer_mae, s=10, color=fs.HERO, zorder=3, lw=0)
    e.scatter(np.ones(len(ch)), ch.selected_outer_mae, s=10, color=np.where(same, fs.HERO, fs.PAPER), zorder=3, lw=0)
    better = int((ch.selected_outer_mae < ch.frozen_outer_mae).sum())
    e.text(0.5, 0.98, f"nested choice better in {better} of {len(ch)} folds;\nfrozen setting chosen in {int(same.sum())}",
           transform=e.transAxes, ha="center", va="top", fontsize=5.0, color=fs.GREY_D, linespacing=fs.LS)
    e.set_xticks([0, 1], ["Frozen", "Nested choice"], fontsize=5.8)
    e.tick_params(axis="x", length=0)
    e.set_xlim(-0.4, 1.4)
    e.set_ylim(0.2, 0.62)
    e.set_ylabel("Outer-fold MAE at k = 1 (fold mean)")
    fs.export(S.fig, "FigS11_independent", axes=axs + [d, e], ids=list("abcde"),
              row_groups=[["a", "b", "c"], ["d", "e"]])


ALL = dict(s1=s1_years, s2=s2_learning, s3=s3_robustness, s4=s4_per_class, s5=s5_rejection, s6=s6_interlab_pm,
           s7=s7_fewshot, s8=s8_reliability, s9=s9_fingerprints, s10=s10_exemplars, s11=s11_independent)

if __name__ == "__main__":
    for key in (sys.argv[1:] or ALL):
        ALL[key]()
