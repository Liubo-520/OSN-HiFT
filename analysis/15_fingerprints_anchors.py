"""Solvent-response fingerprints and Bayesian anchor-solvent design.

Fingerprints: random-coefficient model on raw log10 permeance with physical (centered, unscaled) descriptors,
so each membrane receives posterior slopes d logP/d log(eta), d logP/d dP, d logP/d dH, d logP/d log(Vm).
Anchor design: expected posterior predictive variance of HiFT after measuring one or two anchor solvents.
"""
import json

import numpy as np
import pandas as pd

from hift import HierarchicalTransfer
from osn_core import DATA, REF, RESULTS, S1, SOLV, TRANSFER_DESC, transfer_design

COEF = ["intercept"] + TRANSFER_DESC


def fingerprints(d):
    cells = d.groupby(["entity", S1], as_index=False).agg(
        y=("logP", "median"), paper=(REF, "first"), material=("material", "first"),
        name=("name", "first"))
    solvents = sorted(d[S1].unique())
    center = SOLV.loc[solvents, TRANSFER_DESC].to_numpy(float).mean(0)
    groups, ents = [], []
    for ent, g in cells.groupby("entity"):
        z = SOLV.loc[g[S1], TRANSFER_DESC].to_numpy(float) - center
        groups.append((np.column_stack([np.ones(len(g)), z]), g["y"].to_numpy()))
        ents.append(ent)
    model = HierarchicalTransfer(prior_scale=1e-4).fit(groups)
    rows = []
    meta = cells.groupby("entity").agg(paper=("paper", "first"), material=("material", "first"),
                                       name=("name", "first"), n_solvents=(S1, "nunique"))
    for ent, (Z, y) in zip(ents, groups):
        if len(y) < 3:
            continue
        u, V = model.posterior(Z, y)
        coef = model.beta_ + u
        sd = np.sqrt(np.diag(V))
        ols = np.linalg.lstsq(Z, y, rcond=None)[0] if len(y) >= 6 else np.full(len(coef), np.nan)
        rows.append(dict(entity=ent, **meta.loc[ent].to_dict(),
                         **{f"b_{c}": v for c, v in zip(COEF, coef)},
                         **{f"sd_{c}": v for c, v in zip(COEF, sd)},
                         **{f"ols_{c}": v for c, v in zip(COEF, ols)}))
    fp = pd.DataFrame(rows)
    fp.to_csv(RESULTS / "fingerprints.csv", index=False)
    pop = dict(beta=dict(zip(COEF, model.beta_.round(4).tolist())),
               sd_between=dict(zip(COEF, np.sqrt(np.diag(model.Sigma_)).round(4).tolist())),
               s_resid=float(np.sqrt(model.s2_)), center=dict(zip(TRANSFER_DESC, center.round(4).tolist())))
    print(json.dumps(pop, indent=1))
    print(fp.groupby("material")[["b_log_eta", "b_dP", "b_dH", "b_log_Vm"]].agg(["mean", "count"]).round(3))
    return fp, pop


def exemplars(d):
    picks = {
        "10.1016/j.memsci.2017.10.016": "ceramic_grafting",
        "10.1016/j.memsci.2013.12.032": "tio2_alkyl_series",
        "10.1016/j.memsci.2022.121039": "pbi_series",
        "10.1016/j.memsci.2011.03.027": "pdms_vs_pi",
    }
    e = d[d[REF].isin(picks)].groupby([REF, "name", S1], as_index=False).agg(y=("logP", "median"))
    e["series"] = e[REF].map(picks)
    e = e.merge(SOLV[["log_eta", "dP", "dH", "log_Vm", "short"]], left_on=S1, right_index=True)
    e.to_csv(RESULTS / "exemplar_series.csv", index=False)
    print(e.pivot_table(index=["series", "name"], columns="short", values="y").round(2).to_string())


def anchor_design(d):
    oof = pd.read_csv(RESULTS / "benchmark_oof.csv", low_memory=False)
    o = oof[(oof.scenario == "Unseen paper") & (oof.model == "LightGBM") & (oof.seed == 0)].copy()
    o["entity"] = d.loc[o["row"], "entity"].values
    cells = o.assign(r=o["y"] - o["pred"]).groupby(["entity", "solvent"], as_index=False)["r"].median()
    solvents = sorted(d[S1].unique())
    _, center, scale = transfer_design(solvents)
    groups = []
    for _, g in cells.groupby("entity"):
        Z, _, _ = transfer_design(g["solvent"], center, scale)
        groups.append((Z, g["r"].to_numpy()))
    model = HierarchicalTransfer().fit(groups)
    counts = d[S1].value_counts()
    cand = list(counts[counts >= 40].index)
    Zc, _, _ = transfer_design(cand, center, scale)
    base_var = np.einsum("ij,jk,ik->i", Zc, model.Sigma_, Zc) + model.s2_
    rows = []
    for i, a in enumerate(cand):
        for j, b in enumerate(cand):
            if j < i:
                continue
            A = [i] if i == j else [i, j]
            tgt = [t for t in range(len(cand)) if t not in A]
            v = model.expected_variance(Zc[A], Zc[tgt])
            rows.append(dict(anchor1=a, anchor2=b if i != j else "", k=len(A),
                             mean_var=v.mean(), mean_var_zero_shot=base_var[tgt].mean(),
                             reduction=1 - v.mean() / base_var[tgt].mean()))
    des = pd.DataFrame(rows)
    des.to_csv(RESULTS / "anchor_design.csv", index=False)
    print(des[des.k == 1].sort_values("reduction", ascending=False).round(3).to_string(index=False))
    print(des[des.k == 2].sort_values("reduction", ascending=False).head(8).round(3).to_string(index=False))

    fs = pd.read_csv(RESULTS / "fewshot_pure_solvent_permeance.csv", low_memory=False)
    k1 = fs[(fs.k == 1) & (fs.method == "HiFT") & fs.anchors.isin(cand)]
    per = k1.groupby(["seed", "entity", "anchors"], as_index=False)["abs_err"].mean()
    per = per[per.groupby(["seed", "entity"])["anchors"].transform("nunique") >= 2]
    per["centered"] = per["abs_err"] - per.groupby(["seed", "entity"])["abs_err"].transform("mean")
    ent_level = per.groupby(["entity", "anchors"], as_index=False)["centered"].mean()
    rng = np.random.default_rng(0)
    emp = []
    for a, g in ent_level.groupby("anchors"):
        bs = [g["centered"].values[rng.integers(0, len(g), len(g))].mean() for _ in range(2000)]
        emp.append(dict(anchor=a, n_membranes=len(g), centered_mae=g["centered"].mean(),
                        lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    emp = pd.DataFrame(emp).merge(des[des.k == 1][["anchor1", "reduction"]], left_on="anchor", right_on="anchor1")
    emp.to_csv(RESULTS / "anchor_empirical.csv", index=False)
    from scipy.stats import spearmanr
    ok = emp[emp.n_membranes >= 5]
    print(emp.round(3).to_string(index=False))
    print("Spearman(predicted reduction, -empirical centered MAE) over anchors with >=5 membranes:",
          round(spearmanr(ok["reduction"], -ok["centered_mae"]).statistic, 3), "n=", len(ok))


def solvent_effects(d):
    """Between-membrane solvent means vs within-membrane solvent offsets (two-way additive fit), centred on the
    same solvent set. Divergence reflects which membranes researchers choose to test in which solvent."""
    cells = d.groupby(["entity", S1], as_index=False).agg(y=("logP", "median"), paper=(REF, "first"),
                                                          material=("material", "first"))
    ns = cells.groupby("entity")[S1].transform("nunique")
    multi = cells[ns >= 2]

    def within(m):
        a = m.groupby("entity")["y"].mean()
        b = pd.Series(0.0, index=m[S1].unique())
        for _ in range(200):
            b = (m["y"] - m["entity"].map(a)).groupby(m[S1]).mean()
            a = (m["y"] - m[S1].map(b)).groupby(m["entity"]).mean()
        return b

    keep = multi[S1].value_counts()
    keep = keep[keep >= 8].index
    b_w = within(multi)[keep]
    b_b = cells.groupby(S1)["y"].mean()[keep]
    b_w, b_b = b_w - b_w.mean(), b_b - b_b.mean()
    rng = np.random.default_rng(0)
    ents = multi["entity"].unique()
    boots = []
    for _ in range(500):
        pick = rng.choice(ents, len(ents), replace=True)
        mm = pd.concat([multi[multi.entity == e].assign(entity=f"{e}#{i}") for i, e in enumerate(pick)])
        bw = within(mm).reindex(keep)
        boots.append(bw - bw.mean())
    boots = pd.concat(boots, axis=1)
    out = pd.DataFrame({"between": b_b, "within": b_w, "within_lo": boots.quantile(0.025, axis=1),
                        "within_hi": boots.quantile(0.975, axis=1),
                        "n_cells_all": cells.groupby(S1).size()[keep],
                        "n_multi_membranes": multi.groupby(S1)["entity"].nunique()[keep]})
    out["short"] = SOLV.loc[out.index, "short"]
    out["log_eta"] = SOLV.loc[out.index, "log_eta"]
    share = cells.groupby([S1, "material"]).size().unstack(fill_value=0)
    share = share.div(share.sum(1), axis=0)
    out = out.join(share.add_prefix("share_"), how="left")
    out.to_csv(RESULTS / "solvent_effects.csv", index_label="solvent")
    from scipy.stats import pearsonr
    print(out[["short", "between", "within", "n_multi_membranes"]].round(2).to_string())
    print("Pearson between vs within:", round(pearsonr(out.between, out.within).statistic, 3),
          "| corr within vs log_eta:", round(pearsonr(out.within, out.log_eta).statistic, 3),
          "| corr between vs log_eta:", round(pearsonr(out.between, out.log_eta).statistic, 3))


def main():
    d = pd.read_csv(DATA / "pure_solvent_permeance_domain.csv", low_memory=False)
    solvent_effects(d)
    fp, pop = fingerprints(d)
    with open(RESULTS / "fingerprint_population.json", "w") as f:
        json.dump(pop, f, indent=2)
    exemplars(d)
    anchor_design(d)


if __name__ == "__main__":
    main()
