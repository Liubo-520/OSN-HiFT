"""Supporting analyses that pre-empt reviewer questions.

part A (independent of other results): learning curve over training papers, sensitivity to the original cleaning
rules, MLP control, nested grouped hyperparameter tuning.
part B (uses saved results): per-class held-out metrics, ensemble-weight sensitivity, interval reliability,
per-class few-shot errors, fingerprint shrinkage.
"""
import itertools
import json
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import fewshot_summary as fsum
from osn_core import DATA, REF, RESULTS, S1, SP, featurize

warnings.filterwarnings("ignore")
LGB = dict(n_estimators=700, learning_rate=0.03, num_leaves=31, min_child_samples=10, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, n_jobs=8)


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2)


def lgb_fp(dtr, dte, seed, params=None):
    Xtr, voc = featurize(dtr)
    Xte, _ = featurize(dte, vocab=voc)
    p = dict(LGB, **(params or {}))
    return lgb.LGBMRegressor(random_state=seed, **p).fit(Xtr, dtr["y"]).predict(Xte)


def load_sp():
    d = pd.read_csv(DATA / "pure_solvent_permeance_domain.csv", low_memory=False)
    d["y"] = d["logP"]
    return d.reset_index(drop=True)


def learning_curve(d):
    rows = []
    papers = d[REF].unique()
    for seed in (0, 1, 2):
        for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(d, groups=d[REF]):
            dtr_all, dte = d.iloc[tr], d.iloc[te]
            ptr = dtr_all[REF].unique()
            rng = np.random.default_rng(seed)
            for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
                keep = rng.choice(ptr, max(3, int(round(frac * len(ptr)))), replace=False)
                dtr = dtr_all[dtr_all[REF].isin(keep)]
                p = lgb_fp(dtr, dte, seed)
                rows.append(dict(protocol="Unseen paper", seed=seed, frac=frac, n_train_papers=len(keep),
                                 n_train_rows=len(dtr), R2=r2(dte.y.values, p),
                                 MAE=np.mean(np.abs(dte.y.values - p))))
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(d):
            dtr_all, dte = d.iloc[tr], d.iloc[te]
            rng = np.random.default_rng(seed)
            for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
                idx = rng.choice(len(dtr_all), int(round(frac * len(dtr_all))), replace=False)
                dtr = dtr_all.iloc[idx]
                p = lgb_fp(dtr, dte, seed)
                rows.append(dict(protocol="Random 5-fold", seed=seed, frac=frac, n_train_papers=dtr[REF].nunique(),
                                 n_train_rows=len(dtr), R2=r2(dte.y.values, p),
                                 MAE=np.mean(np.abs(dte.y.values - p))))
        print("learning curve seed", seed, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "si_learning_curve.csv", index=False)
    print(out.groupby(["protocol", "frac"])[["R2", "MAE", "n_train_rows"]].mean().round(3))


def cleaning_rules(d):
    """Original pipeline: drop SP > 500 LMH/bar, then drop duplicates on (name, solvent1, solvent2, mode)."""
    keep = d[d[SP] <= 500].drop_duplicates(subset=["name", S1, "testConditions.solvent2",
                                                   "testConditions.filtrationMode"]).index
    rows = []
    for label, dd in [("All records", d), ("Original cleaning rules", d.loc[keep].reset_index(drop=True))]:
        for seed in (0, 1, 2):
            for prot, sp in [("Random 5-fold", KFold(5, shuffle=True, random_state=seed).split(dd)),
                             ("Unseen paper", GroupKFold(5, shuffle=True, random_state=seed).split(dd, groups=dd[REF]))]:
                pred = np.zeros(len(dd))
                for tr, te in sp:
                    pred[te] = lgb_fp(dd.iloc[tr], dd.iloc[te], seed)
                rows.append(dict(dataset=label, protocol=prot, seed=seed, n=len(dd), R2=r2(dd.y.values, pred),
                                 R2_linear=r2(10 ** dd.y.values, 10 ** pred), MAE=np.mean(np.abs(dd.y.values - pred))))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "si_cleaning_rules.csv", index=False)
    print(out.groupby(["dataset", "protocol"])[["n", "R2", "R2_linear", "MAE"]].mean().round(3))


def mlp_control(d):
    rows = []
    for seed in (0, 1, 2):
        for prot, sp in [("Random 5-fold", KFold(5, shuffle=True, random_state=seed).split(d)),
                         ("Unseen paper", GroupKFold(5, shuffle=True, random_state=seed).split(d, groups=d[REF]))]:
            pred = np.zeros(len(d))
            for tr, te in sp:
                Xtr, voc = featurize(d.iloc[tr])
                Xte, _ = featurize(d.iloc[te], vocab=voc)
                m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                  MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True,
                                               max_iter=600, random_state=seed))
                pred[te] = m.fit(Xtr, d.y.values[tr]).predict(Xte)
            rows.append(dict(model="MLP (128-64)", protocol=prot, seed=seed, R2=r2(d.y.values, pred),
                             MAE=np.mean(np.abs(d.y.values - pred))))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "si_mlp.csv", index=False)
    print(out.groupby("protocol")[["R2", "MAE"]].agg(["mean", "std"]).round(3))


def nested_tuning(d):
    grid = [dict(num_leaves=nl, min_child_samples=mc, n_estimators=ne)
            for nl, mc, ne in itertools.product((7, 31), (5, 20), (300, 900))]
    rows = []
    for prot in ("Unseen paper", "Random 5-fold"):
        outer = (GroupKFold(5, shuffle=True, random_state=0).split(d, groups=d[REF]) if prot == "Unseen paper"
                 else KFold(5, shuffle=True, random_state=0).split(d))
        pred_grouped_tuned = np.zeros(len(d))
        pred_row_tuned = np.zeros(len(d))
        chosen = []
        for tr, te in outer:
            dtr = d.iloc[tr].reset_index(drop=True)
            scores_g, scores_r = [], []
            for g in grid:
                pg = np.zeros(len(dtr))
                pr = np.zeros(len(dtr))
                for itr, ite in GroupKFold(3, shuffle=True, random_state=1).split(dtr, groups=dtr[REF]):
                    pg[ite] = lgb_fp(dtr.iloc[itr], dtr.iloc[ite], 0, g)
                for itr, ite in KFold(3, shuffle=True, random_state=1).split(dtr):
                    pr[ite] = lgb_fp(dtr.iloc[itr], dtr.iloc[ite], 0, g)
                scores_g.append(np.mean(np.abs(dtr.y - pg)))
                scores_r.append(np.mean(np.abs(dtr.y - pr)))
            bg, br = grid[int(np.argmin(scores_g))], grid[int(np.argmin(scores_r))]
            chosen.append(dict(protocol=prot, grouped_choice=json.dumps(bg), row_choice=json.dumps(br),
                               inner_grouped_mae=min(scores_g), inner_row_mae=min(scores_r)))
            pred_grouped_tuned[te] = lgb_fp(dtr, d.iloc[te], 0, bg)
            pred_row_tuned[te] = lgb_fp(dtr, d.iloc[te], 0, br)
        for lab, p in [("Grouped inner CV", pred_grouped_tuned), ("Row-level inner CV", pred_row_tuned)]:
            rows.append(dict(protocol=prot, tuning=lab, R2=r2(d.y.values, p), MAE=np.mean(np.abs(d.y.values - p))))
        pd.DataFrame(chosen).to_csv(RESULTS / f"si_nested_choices_{prot.split()[0].lower()}.csv", index=False)
        print(prot, "done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "si_nested_tuning.csv", index=False)
    print(out.round(3))


def part_b():
    oof = pd.read_csv(RESULTS / "benchmark_oof.csv", low_memory=False)
    d = load_sp()
    o = oof[(oof.scenario == "Unseen material class")].copy()
    o["material"] = d.loc[o["row"], "material"].values
    pc = o.groupby(["model", "material"]).apply(
        lambda g: pd.Series(dict(n=len(g), R2=r2(g.y.values, g.pred.values), MAE=np.mean(np.abs(g.y - g.pred)),
                                 bias=np.mean(g.pred - g.y)))).reset_index()
    pc.to_csv(RESULTS / "si_per_class_heldout.csv", index=False)
    print(pc[pc.model == "LightGBM"].round(3))

    rows = []
    for tag in ("pure_solvent_permeance", "solution_permeance"):
        r = fsum.load(tag)
        r = r[r.k > 0]
        key = ["seed", "paper", "entity", "target", "k", "draw"]
        a = r[r.method == "HiFT − pairwise learner"].set_index(key)[["y", "pred"]]
        b = r[r.method == "Pairwise transfer GBM"].set_index(key)[["pred"]].rename(columns={"pred": "pp"})
        j = a.join(b, how="inner").reset_index()
        for w in np.linspace(0, 1, 11):
            j["e"] = (j.y - (w * j.pred + (1 - w) * j.pp)).abs()
            c = j.groupby(["seed", "paper", "entity", "target", "k"], as_index=False)["e"].mean()
            for k, g in c.groupby("k"):
                rows.append(dict(dataset=tag, w_backfit=round(w, 1), k=k, MAE=g.e.mean()))
    ew = pd.DataFrame(rows)
    ew.to_csv(RESULTS / "si_ensemble_weight.csv", index=False)
    print(ew.pivot_table(index=["dataset", "w_backfit"], columns="k", values="MAE").round(3))

    rel = []
    r = fsum.load("pure_solvent_permeance")
    h = r[(r.method == "HiFT")].copy()
    h["err"] = (h.y - h.pred).abs()
    h["bin"] = pd.qcut(h.sd, 8, duplicates="drop")
    for b, g in h.groupby("bin", observed=True):
        rel.append(dict(sd_mid=g.sd.median(), rmse=np.sqrt(np.mean((g.y - g.pred) ** 2)), mae=g.err.mean(), n=len(g)))
    pd.DataFrame(rel).to_csv(RESULTS / "si_reliability.csv", index=False)
    print(pd.DataFrame(rel).round(3))

    ce = fsum.cell_errors(r)
    sub = ce[((ce.method == "Zero-shot GBM") & (ce.k == 0)) | ((ce.method == "HiFT") & ce.k.isin([0, 1, 3]))]
    pcm = sub.groupby(["material", "method", "k"]).agg(MAE=("abs_err", "mean"), n_membranes=("entity", "nunique"))
    pcm.reset_index().to_csv(RESULTS / "si_fewshot_per_class.csv", index=False)
    print(pcm.round(3))


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("a", "all"):
        d = load_sp()
        learning_curve(d)
        cleaning_rules(d)
        mlp_control(d)
        nested_tuning(d)
    if part in ("b", "all"):
        part_b()
