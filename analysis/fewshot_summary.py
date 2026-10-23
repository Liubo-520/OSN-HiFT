"""Summaries of few-shot results: draw-averaged cell errors with paper-cluster bootstrap CIs, and coverage."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from osn_core import RESULTS

MIN_SOLVENTS = {"pure_solvent_permeance": 4, "solution_permeance": 3}


def load(tag):
    r = pd.read_csv(RESULTS / f"fewshot_{tag}.csv", low_memory=False)
    return r[r.n_solvents >= MIN_SOLVENTS[tag]]


def cell_errors(r):
    return r.groupby(["seed", "method", "k", "paper", "entity", "target", "material"], as_index=False)["abs_err"].mean()


def boot_mean(df, value, B=2000, seed=0):
    papers = df["paper"].unique()
    idx = df["paper"].map({p: i for i, p in enumerate(papers)}).to_numpy()
    s = np.bincount(idx, weights=df[value].to_numpy(), minlength=len(papers))
    c = np.bincount(idx, minlength=len(papers)).astype(float)
    W = np.random.default_rng(seed).multinomial(len(papers), np.full(len(papers), 1 / len(papers)), size=B)
    bs = (W @ s) / (W @ c)
    return df[value].mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def summary(tag):
    ce = cell_errors(load(tag))
    rows = []
    for (m, k), g in ce.groupby(["method", "k"]):
        mean, lo, hi = boot_mean(g, "abs_err")
        rows.append(dict(method=m, k=k, mae=mean, lo=lo, hi=hi, n_cells=g.groupby(["entity", "target"]).ngroups,
                         n_membranes=g["entity"].nunique(), n_papers=g["paper"].nunique()))
    return pd.DataFrame(rows)


def paired_gain(tag, a, b, k):
    """Mean (a - b) of cell MAE at a given k with paper-bootstrap CI; positive means b is better."""
    ce = cell_errors(load(tag))
    x = ce[ce.k == k].pivot_table(index=["seed", "paper", "entity", "target"], columns="method", values="abs_err")
    d = (x[a] - x[b]).rename("d").reset_index()
    return boot_mean(d, "d") + (float((d["d"] > 0).mean()),)


def coverage(tag, method, levels=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95)):
    r = load(tag)
    r = r[r.method == method]
    z = (r["y"] - r["pred"]).abs() / r["sd"]
    out = []
    for k, g in r.assign(z=z).groupby("k"):
        for lv in levels:
            cov = (g["z"] <= norm.ppf(0.5 + lv / 2)).astype(float)
            mean, lo, hi = boot_mean(g.assign(c=cov), "c")
            out.append(dict(method=method, k=k, level=lv, coverage=mean, lo=lo, hi=hi,
                            half_width_log=float(norm.ppf(0.5 + lv / 2) * g["sd"].median())))
    return pd.DataFrame(out)
