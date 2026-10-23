"""OSN-Bench: zero-shot performance of five models under five deployment scenarios.

Uncertainty: paper-cluster bootstrap (B=1000) of the seed-averaged metric.
"""
import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from osn_core import DATA, REF, RESULTS, S1, featurize

warnings.filterwarnings("ignore")
SEEDS = (0, 1, 2)
B = 1000
LGB = dict(n_estimators=700, learning_rate=0.03, num_leaves=31, min_child_samples=10, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, n_jobs=8)


def make_model(name, seed):
    if name == "Ridge":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             RidgeCV(alphas=np.logspace(-2, 3, 20)))
    if name == "Random forest":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                             RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.33,
                                                   n_jobs=16, random_state=seed))
    if name == "LightGBM":
        return lgb.LGBMRegressor(random_state=seed, **LGB)
    raise ValueError(name)


MODELS = ["Global mean", "Solvent mean", "Ridge", "Random forest", "LightGBM"]


def fit_predict(name, dtr, dte, seed, extra_numeric=()):
    ytr = dtr["y"].to_numpy()
    if name == "Global mean":
        return np.full(len(dte), ytr.mean())
    if name == "Solvent mean":
        m = dtr.groupby(S1)["y"].mean()
        return dte[S1].map(m).fillna(ytr.mean()).to_numpy()
    Xtr, vocab = featurize(dtr, extra_numeric=extra_numeric)
    Xte, _ = featurize(dte, vocab=vocab, extra_numeric=extra_numeric)
    model = make_model(name, seed)
    model.fit(Xtr, ytr)
    return model.predict(Xte)


def run_splits(d, splits, models, extra_numeric=()):
    """splits: list of (seed, train_idx, test_idx). Returns long OOF frame."""
    rows = []
    for seed, tr, te in splits:
        dtr, dte = d.iloc[tr], d.iloc[te]
        for name in models:
            p = fit_predict(name, dtr, dte, seed, extra_numeric)
            rows.append(pd.DataFrame({"row": dte.index, "seed": seed, "model": name, "y": dte["y"].values,
                                      "pred": p, "paper": dte[REF].values, "solvent": dte[S1].values}))
    return pd.concat(rows, ignore_index=True)


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def boot_metrics(oof, rng):
    """Seed-averaged R2/MAE/Spearman with paper-cluster bootstrap CI for each model."""
    from scipy.stats import spearmanr
    out = []
    papers = oof["paper"].unique()
    pidx = {p: i for i, p in enumerate(papers)}
    for model, g in oof.groupby("model"):
        seeds = sorted(g["seed"].unique())
        per = [g[g.seed == s] for s in seeds]
        point = {k: np.mean([f(x["y"].values, x["pred"].values) for x in per])
                 for k, f in [("R2", r2), ("MAE", lambda y, p: np.mean(np.abs(y - p))),
                              ("rho", lambda y, p: spearmanr(y, p).statistic)]}
        codes = [x["paper"].map(pidx).values for x in per]
        bs_r2, bs_mae = [], []
        for _ in range(B):
            w = np.bincount(rng.integers(0, len(papers), len(papers)), minlength=len(papers))
            vals_r2, vals_mae = [], []
            for x, c in zip(per, codes):
                ww = w[c]
                if ww.sum() == 0:
                    continue
                y, p = x["y"].values, x["pred"].values
                ybar = np.average(y, weights=ww)
                vals_r2.append(1 - np.sum(ww * (y - p) ** 2) / np.sum(ww * (y - ybar) ** 2))
                vals_mae.append(np.average(np.abs(y - p), weights=ww))
            bs_r2.append(np.mean(vals_r2))
            bs_mae.append(np.mean(vals_mae))
        out.append(dict(model=model, n_rows=len(per[0]), n_papers=per[0]["paper"].nunique(), **point,
                        R2_lo=np.percentile(bs_r2, 2.5), R2_hi=np.percentile(bs_r2, 97.5),
                        MAE_lo=np.percentile(bs_mae, 2.5), MAE_hi=np.percentile(bs_mae, 97.5)))
    return pd.DataFrame(out)


def main():
    rng = np.random.default_rng(2026)
    d = pd.read_csv(DATA / "pure_solvent_permeance_domain.csv", low_memory=False)
    d["y"] = d["logP"]
    d = d.reset_index(drop=True)
    n = len(d)
    all_metrics, oofs = [], {}

    scen = {}
    scen["Random 5-fold"] = [(s, tr, te) for s in SEEDS for tr, te in KFold(5, shuffle=True, random_state=s).split(d)]
    scen["Unseen paper"] = [(s, tr, te) for s in SEEDS
                            for tr, te in GroupKFold(5, shuffle=True, random_state=s).split(d, groups=d[REF])]
    scen["Unseen material class"] = [(0, np.where(d.material != m)[0], np.where(d.material == m)[0])
                                     for m in d.material.unique() if (d.material == m).sum() >= 15]
    big = d[S1].value_counts()
    scen["Unseen solvent"] = [(0, np.where(d[S1] != s)[0], np.where(d[S1] == s)[0]) for s in big[big >= 40].index]
    known = d["year"].notna()
    scen["Future papers (≥2021)"] = [(0, np.where(known & (d.year <= 2020))[0], np.where(known & (d.year >= 2021))[0])]

    for name, splits in scen.items():
        print(f"running {name}: {len(splits)} splits", flush=True)
        oof = run_splits(d, splits, MODELS)
        oofs[name] = oof
        m = boot_metrics(oof, rng)
        m.insert(0, "scenario", name)
        all_metrics.append(m)
        print(m[["model", "R2", "R2_lo", "R2_hi", "MAE", "rho"]].round(3).to_string(index=False), flush=True)

    met = pd.concat(all_metrics, ignore_index=True)
    met.to_csv(RESULTS / "benchmark_metrics.csv", index=False)
    pd.concat([o.assign(scenario=k) for k, o in oofs.items()]).to_csv(RESULTS / "benchmark_oof.csv", index=False)

    print("\n== temporal drift vs training size (LightGBM) ==", flush=True)
    recent = d.loc[known & (d.year >= 2021), REF].unique()
    old = d.loc[known & (d.year <= 2020), REF].unique()
    rows = []
    for rep in range(10):
        r = np.random.default_rng(rep)
        perm = r.permutation(recent)
        A, Bp = perm[: len(perm) // 2], perm[len(perm) // 2:]
        te = np.where(d[REF].isin(A))[0]
        tr_temporal = np.where(d[REF].isin(old))[0]
        pool = np.concatenate([old, Bp])
        mixed = r.choice(pool, size=len(old), replace=False)
        tr_mixed = np.where(d[REF].isin(mixed))[0]
        for label, tr in [("Past only (≤2020)", tr_temporal), ("Same size, mixed years", tr_mixed)]:
            p = fit_predict("LightGBM", d.iloc[tr], d.iloc[te], rep)
            y = d.iloc[te]["y"].values
            rows.append(dict(rep=rep, training=label, R2=r2(y, p), MAE=np.mean(np.abs(y - p)),
                             n_train_rows=len(tr), n_train_papers=d.iloc[tr][REF].nunique()))
    tc = pd.DataFrame(rows)
    tc.to_csv(RESULTS / "temporal_control.csv", index=False)
    print(tc.groupby("training")[["R2", "MAE", "n_train_rows", "n_train_papers"]].agg(["mean", "std"]).round(3))

    print("\n== rejection: random vs unseen paper (LightGBM) ==", flush=True)
    rj = pd.read_csv(DATA / "rejection_long.csv", low_memory=False)
    rj["y"] = rj["R"] / 100
    rj["log_solute_MW"] = np.log10(rj["solute_MW"])
    for c in ["Dye", "Oligomer", "Alkanes", "Other", "Sugars", "Industrial Model Solute"]:
        rj[f"solcat_{c}"] = (rj["solute_cat"] == c).astype(int)
    extra = ["log_solute_MW"] + [c for c in rj.columns if c.startswith("solcat_")]
    rj = rj.reset_index(drop=True)
    rsc = {"Random 5-fold": [(s, tr, te) for s in SEEDS for tr, te in KFold(5, shuffle=True, random_state=s).split(rj)],
           "Unseen paper": [(s, tr, te) for s in SEEDS
                            for tr, te in GroupKFold(5, shuffle=True, random_state=s).split(rj, groups=rj[REF])]}
    rmet = []
    for name, splits in rsc.items():
        oof = run_splits(rj, splits, ["Solvent mean", "LightGBM"], extra_numeric=extra)
        m = boot_metrics(oof, rng)
        m.insert(0, "scenario", name)
        rmet.append(m)
        if name == "Unseen paper":
            oof.to_csv(RESULTS / "rejection_oof_unseen_paper.csv", index=False)
        else:
            oof.to_csv(RESULTS / "rejection_oof_random.csv", index=False)
    rmet = pd.concat(rmet)
    rmet.to_csv(RESULTS / "rejection_benchmark.csv", index=False)
    print(rmet[["scenario", "model", "R2", "R2_lo", "R2_hi", "MAE"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
