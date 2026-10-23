"""Collect every number quoted in the manuscript into results/key_numbers.json (single source of truth)."""
import json

import numpy as np
import pandas as pd

import fewshot_summary as fsum
from osn_core import DATA, RESULTS


def r(x, n=6):
    # stored at 6 dp so that the manuscript rounds once; rounding to 3 dp here and to 2 dp in the build gave
    # double-rounding errors (e.g. R2 0.2454 -> 0.245 -> 0.24)
    return None if pd.isna(x) else round(float(x), n)


K = {}
K["curation"] = json.load(open(DATA / "curation_audit.json", encoding="utf-8"))

met = pd.read_csv(RESULTS / "benchmark_metrics.csv")
K["benchmark"] = {f"{s} | {m}": dict(R2=r(g.R2.iloc[0]), R2_CI=[r(g.R2_lo.iloc[0]), r(g.R2_hi.iloc[0])],
                                       MAE=r(g.MAE.iloc[0]), MAE_CI=[r(g.MAE_lo.iloc[0]), r(g.MAE_hi.iloc[0])],
                                       rho=r(g.rho.iloc[0]), n_rows=int(g.n_rows.iloc[0]), n_papers=int(g.n_papers.iloc[0]))
                  for (s, m), g in met.groupby(["scenario", "model"])}
tc = pd.read_csv(RESULTS / "temporal_control.csv")
piv = tc.pivot(index="rep", columns="training", values="R2")
K["temporal_control"] = {c: dict(R2_mean=r(piv[c].mean()), R2_sd=r(piv[c].std())) for c in piv}
d = piv["Same size, mixed years"] - piv["Past only (≤2020)"]
K["temporal_control"]["paired_dR2"] = dict(mean=r(d.mean()), sd=r(d.std()), n_better=int((d > 0).sum()), n=len(d))
rj = pd.read_csv(RESULTS / "rejection_benchmark.csv")
K["rejection"] = {f"{s} | {m}": dict(R2=r(g.R2.iloc[0]), R2_CI=[r(g.R2_lo.iloc[0]), r(g.R2_hi.iloc[0])],
                                       MAE_pp=r(100 * g.MAE.iloc[0], 1), n_rows=int(g.n_rows.iloc[0]))
                  for (s, m), g in rj.groupby(["scenario", "model"])}

vc = pd.read_csv(RESULTS / "variance_components.csv")
K["variance"] = {f"{row.dataset} | {row.basis}": dict(paper=r(row.share_paper), paper_CI=[r(row.share_paper_lo), r(row.share_paper_hi)],
                                                     membrane=r(row.share_membrane), residual=r(row.share_residual),
                                                     n_cells=int(row.n_cells), n_papers=int(row.n_papers))
                 for row in vc.itertuples()}
K["noise"] = json.load(open(RESULTS / "noise_summary.json"))


def pooled_sd(cells):
    g = cells.groupby("cell")["y"]
    n, s2 = g.size(), g.var(ddof=1)
    return float(np.sqrt(((n - 1) * s2).sum() / (n - 1).sum())), int(len(n)), int((n - 1).sum())


ilc = pd.read_csv(RESULTS / "interlab_cells.csv")
ilc = ilc[ilc.dataset == "Pure-solvent permeance"].copy()
ilc["cell"] = ilc["commercialName"] + " | " + ilc["testConditions.solvent1"]
ilc["ref_norm"] = ilc["report.reference"].astype(str).str.lower().str.replace(r"^(https?://)?(dx\.)?(doi\.org/)", "",
                                                                             regex=True)
ded = ilc.drop_duplicates(["cell", "ref_norm"])
identical = ded.groupby("cell")["y"].agg(lambda v: v.nunique() == 1)
sens = ded[ded.cell.isin(identical[~identical].index)]
var_y = K["noise"]["Pure-solvent permeance"]["var_logP"]
K["interlab_sensitivity"] = {}
for key, frame in [("primary", ilc), ("merged_refs_no_identical", sens),
                   ("polymeric_only", ded[~ded.commercialName.str.contains("Inopor")])]:
    sd, cells, df = pooled_sd(frame)
    K["interlab_sensitivity"][key] = dict(sd=r(sd), cells=cells, df=df, mae_floor=r(sd * np.sqrt(2 / np.pi)),
                                          r2_ceiling=r(1 - sd ** 2 / var_y))
K["interlab_sensitivity"]["n_identical_pairs"] = int(identical.sum())
K["interlab_sensitivity"]["n_duplicate_reference_records"] = int(len(ilc) - len(ded))

for tag, key in [("pure_solvent_permeance", "fewshot_sp"), ("solution_permeance", "fewshot_pm")]:
    s = fsum.summary(tag)
    K[key] = {f"{row.method} | k={row.k}": dict(MAE=r(row.mae), CI=[r(row.lo), r(row.hi)], n_cells=int(row.n_cells),
                                                n_membranes=int(row.n_membranes), n_papers=int(row.n_papers))
              for row in s.itertuples()}
    pg = {}
    for b in ["Pairwise transfer GBM", "GBM + mean shift", "Bayesian level shift", "Viscosity scaling",
              "Vm/η scaling", "Additive solvent offset", "Zero-shot GBM", "HiFT − pairwise learner",
              "HiFT − auxiliary data", "HiFT + zero-shot prior shrinkage", "Linear random-coefficient"]:
        for k in (1, 2, 3):
            m, lo, hi, fr = fsum.paired_gain(tag, b, "HiFT", k)
            pg[f"{b} | k={k}"] = dict(dMAE=r(m), CI=[r(lo), r(hi)], frac_HiFT_better=r(fr, 2))
    K[key + "_paired_vs_HiFT"] = pg
    cov = fsum.coverage(tag, "HiFT")
    K[key + "_coverage"] = {f"k={row.k} | level={row.level}": dict(coverage=r(row.coverage), CI=[r(row.lo), r(row.hi)],
                                                                  half_width_log=r(row.half_width_log),
                                                                  fold=r(10 ** row.half_width_log, 1))
                            for row in cov.itertuples()}
hp = pd.read_csv(RESULTS / "hift_params_pure_solvent_permeance.csv")
K["hift_params_sp"] = dict(sigma=r(np.sqrt(hp.sigma2).mean()), tau=r(np.sqrt(hp.tau2).mean()),
                           tau_all=r(np.sqrt(hp.tau2_all).mean()), n_train_multi=r(hp.n_train_multi.mean(), 1),
                           n_aux_multi=r(hp.n_aux_multi.mean(), 1))

cf = pd.read_csv(RESULTS / "conformal_coverage.csv")
c = cf.groupby(["scenario", "calibration", "level"])[["coverage", "q"]].mean().reset_index()
K["conformal"] = {f"{x.scenario} | {x.calibration} | {x.level}": dict(coverage=r(x.coverage), half_width_log=r(x.q),
                                                                     fold=r(10 ** x.q, 1)) for x in c.itertuples()}

fp = pd.read_csv(RESULTS / "fingerprints.csv")
K["fingerprints"] = dict(population=json.load(open(RESULTS / "fingerprint_population.json")), n=len(fp),
                         class_means={m: dict(n=len(g), visc=r(g.b_log_eta.mean()), dP_per10=r(10 * g.b_dP.mean()),
                                              dH_per10=r(10 * g.b_dH.mean()), logVm=r(g.b_log_Vm.mean()))
                                      for m, g in fp.groupby("material")})
se = pd.read_csv(RESULTS / "solvent_effects.csv", index_col=0)
from scipy.stats import pearsonr, spearmanr
K["solvent_effects"] = dict(r_between_within=r(pearsonr(se.between, se.within).statistic),
                            r_within_logeta=r(pearsonr(se.within, se.log_eta).statistic),
                            r_between_logeta=r(pearsonr(se.between, se.log_eta).statistic),
                            table=se[["short", "between", "within", "n_multi_membranes"]].round(3).to_dict(orient="index"))
des = pd.read_csv(RESULTS / "anchor_design.csv")
emp = pd.read_csv(RESULTS / "anchor_empirical.csv")
K["anchor_design_single"] = des[des.k == 1].sort_values("reduction", ascending=False)[["anchor1", "reduction"]] \
    .round(3).to_dict(orient="records")
K["anchor_design_pairs_top5"] = des[des.k == 2].sort_values("reduction", ascending=False).head(5)[
    ["anchor1", "anchor2", "reduction"]].round(3).to_dict(orient="records")
K["anchor_empirical"] = emp[["anchor", "n_membranes", "centered_mae", "lo", "hi", "reduction"]].round(3).to_dict(orient="records")
e5 = emp[emp.n_membranes >= 5]
K["anchor_spearman"] = r(spearmanr(e5.reduction, -e5.centered_mae).statistic)
if (RESULTS / "indep_summary.json").exists():
    K["independent"] = json.load(open(RESULTS / "indep_summary.json", encoding="utf-8"))
if (RESULTS / "reference_duplicates.json").exists():
    K["reference_duplicates"] = json.load(open(RESULTS / "reference_duplicates.json"))

K["fig4_example"] = json.load(open(RESULTS / "fig4_example.json"))
ex = pd.read_csv(RESULTS / "exemplar_series.csv")
K["tio2_series"] = ex[ex.series == "tio2_alkyl_series"].pivot_table(index="name", columns="short", values="y").round(2) \
    .to_dict(orient="index")

with open(RESULTS / "key_numbers.json", "w", encoding="utf-8") as f:
    json.dump(K, f, indent=1, ensure_ascii=False)
print(json.dumps({k: K[k] for k in ["temporal_control", "hift_params_sp", "anchor_spearman"]}, indent=1, ensure_ascii=False))
for key in ["fewshot_sp", "fewshot_pm"]:
    print(key, {k: v["MAE"] for k, v in K[key].items() if k.startswith(("HiFT |", "Zero-shot GBM | k=0", "Pairwise", "Viscosity"))})
print({k: v for k, v in K["fewshot_sp_coverage"].items() if "level=0.9" in k})
