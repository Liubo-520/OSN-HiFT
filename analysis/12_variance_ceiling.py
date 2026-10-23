"""Study effects and reproducibility ceiling.

1. Variance components (paper / membrane-within-paper / residual) for pure-solvent permeance, solution permeance
   and rejection, on raw values (solvent fixed effects) and on unseen-paper model residuals. Paper bootstrap CIs.
2. Inter-laboratory spread of the same commercial membrane in the same solvent, and within-paper replicate noise.
3. Implied ceilings for zero-shot prediction on unseen papers.
"""
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from joblib import Parallel, delayed

from osn_core import DATA, REF, RESULTS, S1

warnings.filterwarnings("ignore")
B = 100


def fit_vc(cells, formula):
    md = smf.mixedlm(formula, cells, groups="paper", re_formula="1", vc_formula={"memb": "0 + C(memb)"})
    r = md.fit(method="lbfgs", reml=True)
    v_p, v_m, v_e = float(r.cov_re.iloc[0, 0]), float(r.vcomp[0]), float(r.scale)
    s = v_p + v_m + v_e
    return dict(var_paper=v_p, var_membrane=v_m, var_residual=v_e, share_paper=v_p / s,
                share_membrane=v_m / s, share_residual=v_e / s)


def boot_once(cells, formula, seed):
    rng = np.random.default_rng(seed)
    papers = cells["paper"].unique()
    pick = rng.choice(papers, len(papers), replace=True)
    parts = []
    for i, p in enumerate(pick):
        c = cells[cells["paper"] == p].copy()
        c["paper"] = f"{p}#{i}"
        c["memb"] = c["memb"].astype(str) + f"#{i}"
        parts.append(c)
    try:
        return fit_vc(pd.concat(parts, ignore_index=True), formula)
    except Exception:
        return None


def components(cells, formula, label):
    point = fit_vc(cells, formula)
    boots = Parallel(n_jobs=16)(delayed(boot_once)(cells, formula, s) for s in range(B))
    boots = pd.DataFrame([b for b in boots if b is not None])
    row = dict(dataset=label, n_cells=len(cells), n_papers=cells["paper"].nunique(),
               n_membranes=cells["memb"].nunique(), **point)
    for k in ["share_paper", "share_membrane", "share_residual"]:
        row[f"{k}_lo"], row[f"{k}_hi"] = np.percentile(boots[k], [2.5, 97.5])
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return row


def main():
    sp = pd.read_csv(DATA / "pure_solvent_permeance_domain.csv", low_memory=False)
    pm = pd.read_csv(DATA / "solution_permeance_domain.csv", low_memory=False)
    rj = pd.read_csv(DATA / "rejection_long.csv", low_memory=False)
    rj["y"] = rj["R"] / 100
    rj["lmw"] = np.log10(rj["solute_MW"])

    rows = []
    for d, label in [(sp, "Pure-solvent permeance"), (pm, "Solution permeance")]:
        c = d.groupby(["entity", S1], as_index=False).agg(y=("logP", "median"), paper=(REF, "first"))
        c = c.rename(columns={S1: "solvent"})
        c["memb"] = c["entity"]
        rows.append({**components(c, "y ~ C(solvent)", label), "basis": "Raw (solvent fixed effects)"})
    c = rj.groupby(["entity", S1, "solute_name"], as_index=False).agg(
        y=("y", "median"), paper=(REF, "first"), lmw=("lmw", "median"))
    c = c.rename(columns={S1: "solvent"})
    c["memb"] = c["entity"]
    rows.append({**components(c, "y ~ C(solvent) + lmw", "Rejection"), "basis": "Raw (solvent fixed effects)"})

    oof = pd.read_csv(RESULTS / "benchmark_oof.csv", low_memory=False)
    o = oof[(oof.scenario == "Unseen paper") & (oof.model == "LightGBM") & (oof.seed == 0)].copy()
    o["res"] = o["y"] - o["pred"]
    o = o.merge(sp[["entity"]].reset_index().rename(columns={"index": "row"}), on="row")
    c = o.groupby(["entity", "solvent"], as_index=False).agg(y=("res", "median"), paper=("paper", "first"))
    c["memb"] = c["entity"]
    rows.append({**components(c, "y ~ 1", "Pure-solvent permeance"), "basis": "Unseen-paper LightGBM residual"})

    ro = pd.read_csv(RESULTS / "rejection_oof_unseen_paper.csv", low_memory=False)
    ro = ro[(ro.model == "LightGBM") & (ro.seed == 0)].copy()
    ro["res"] = ro["y"] - ro["pred"]
    ro = ro.merge(rj[["entity", "solute_name"]].reset_index().rename(columns={"index": "row"}), on="row")
    c = ro.groupby(["entity", "solvent", "solute_name"], as_index=False).agg(y=("res", "median"), paper=("paper", "first"))
    c["memb"] = c["entity"]
    rows.append({**components(c, "y ~ 1", "Rejection"), "basis": "Unseen-paper LightGBM residual"})
    pd.DataFrame(rows).to_csv(RESULTS / "variance_components.csv", index=False)

    print("\n== inter-laboratory spread (commercial membranes) ==", flush=True)
    summary = {}
    allcells = []
    for d, label in [(sp, "Pure-solvent permeance"), (pm, "Solution permeance")]:
        c = d[(d["family"] == "Commercial") & d["commercialName"].notna() & (d["commercialName"] != "Other")]
        per = c.groupby(["commercialName", S1, REF], as_index=False).agg(
            y=("logP", "median"), P=("testConditions.hydraulicP", "median"),
            mode=("testConditions.filtrationMode", "first"))
        g = per.groupby(["commercialName", S1])
        cells = g.agg(n_papers=("y", "size"), sd=("y", "std"), lo=("y", "min"), hi=("y", "max"),
                      mean=("y", "mean")).reset_index()
        cells = cells[cells.n_papers >= 2]
        cells["dataset"] = label
        allcells.append(cells.merge(per, on=["commercialName", S1]))
        w = cells.n_papers - 1
        pooled = np.sqrt(np.sum(w * cells.sd ** 2) / w.sum())
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(cells), len(cells))
            bs.append(np.sqrt(np.sum(w.values[i] * cells.sd.values[i] ** 2) / w.values[i].sum()))
        matched = per.groupby(["commercialName", S1]).filter(
            lambda x: len(x) >= 2 and x["mode"].nunique() == 1 and (x["P"].max() / x["P"].min() <= 2))
        mg = matched.groupby(["commercialName", S1])["y"].agg(["size", "std"])
        mpooled = np.sqrt(np.sum((mg["size"] - 1) * mg["std"] ** 2) / (mg["size"] - 1).sum()) if len(mg) else np.nan

        rep = d.groupby(["entity", S1, "testConditions.hydraulicP", "testConditions.temperature"])["logP"]
        rep = rep.agg(["size", "std"]).dropna()
        rep = rep[rep["size"] >= 2]
        within = np.sqrt(np.sum((rep["size"] - 1) * rep["std"] ** 2) / (rep["size"] - 1).sum())
        var_y = d["logP"].var()
        summary[label] = dict(
            interlab_cells=int(len(cells)), interlab_papers=int(per[REF].nunique()),
            interlab_sd=float(pooled), interlab_sd_lo=float(np.percentile(bs, 2.5)),
            interlab_sd_hi=float(np.percentile(bs, 97.5)),
            matched_cells=int(len(mg)), matched_interlab_sd=float(mpooled),
            within_paper_replicate_groups=int(len(rep)), within_paper_sd=float(within),
            var_logP=float(var_y), sd_logP=float(np.sqrt(var_y)),
            mae_floor_interlab=float(pooled * np.sqrt(2 / np.pi)),
            mae_floor_within=float(within * np.sqrt(2 / np.pi)),
            r2_ceiling_unseen_paper=float(1 - pooled ** 2 / var_y),
            r2_ceiling_unseen_paper_lo=float(1 - np.percentile(bs, 97.5) ** 2 / var_y),
            r2_ceiling_unseen_paper_hi=float(1 - np.percentile(bs, 2.5) ** 2 / var_y),
        )
        print(label, json.dumps(summary[label], indent=1), flush=True)
    pd.concat(allcells).to_csv(RESULTS / "interlab_cells.csv", index=False)
    with open(RESULTS / "noise_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
