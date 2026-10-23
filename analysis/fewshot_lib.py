"""Few-shot transfer library: cached fold preparation, within-membrane contrast model and HiFT predictor."""
import pickle
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from hift import HierarchicalTransfer
from osn_core import DATA, REF, RESULTS, S1, SOLV, featurize, transfer_design

warnings.filterwarnings("ignore")
CACHE = RESULTS / "cache"
CACHE.mkdir(exist_ok=True)
LGB = dict(n_estimators=700, learning_rate=0.03, num_leaves=31, min_child_samples=10, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, n_jobs=8)
WMC = dict(n_estimators=300, learning_rate=0.05, num_leaves=7, min_child_samples=5, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, verbose=-1, n_jobs=8)
PAIR = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=10, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=8)
SOLV_DESC = ["log_eta", "log_Vm", "dD", "dP", "dH", "eps", "gamma", "log_Vm_eta"]
AUX = {"pure_solvent_permeance": "solution_permeance", "solution_permeance": "pure_solvent_permeance"}


def load_domain(tag):
    d = pd.read_csv(DATA / f"{tag}_domain.csv", low_memory=False)
    d["y"] = d["logP"]
    d["log_solute_MW"] = np.log10(d["soluteChoice.solute1.molecularWeight"])
    return d.reset_index(drop=True)


def extra_for(tag):
    return ("log_solute_MW",) if tag == "solution_permeance" else ()


def gbm_fit_predict(dtr, dte, seed, extra):
    Xtr, vocab = featurize(dtr, extra_numeric=extra)
    Xte, _ = featurize(dte, vocab=vocab, extra_numeric=extra)
    return lgb.LGBMRegressor(random_state=seed, **LGB).fit(Xtr, dtr["y"]).predict(Xte)


def prepare_fold(tag, seed, fold, d):
    path = CACHE / f"{tag}_s{seed}_f{fold}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    tr, te = list(GroupKFold(5, shuffle=True, random_state=seed).split(d, groups=d[REF]))[fold]
    extra = extra_for(tag)
    dtr, dte = d.iloc[tr].reset_index(drop=True), d.iloc[te].reset_index(drop=True)
    f_te = gbm_fit_predict(dtr, dte, seed, extra)
    f_oof = np.zeros(len(dtr))
    for itr, ite in GroupKFold(5, shuffle=True, random_state=100 + seed).split(dtr, groups=dtr[REF]):
        f_oof[ite] = gbm_fit_predict(dtr.iloc[itr], dtr.iloc[ite], seed, extra)
    out = dict(tr=tr, te=te, f_te=f_te, f_oof=f_oof)
    pickle.dump(out, open(path, "wb"))
    return out


def cells_of(d, pred, source):
    t = d.assign(f=pred)
    c = t.groupby(["entity", S1], as_index=False).agg(
        y=("y", "median"), f=("f", "median"), paper=(REF, "first"), material=("material", "first"))
    c["source"] = source
    return c


def membrane_features(rows_by_entity, vocab):
    first = rows_by_entity.groupby("entity").head(1).reset_index(drop=True)
    X, _ = featurize(first, vocab=vocab)
    X.index = first["entity"].values
    return X[[c for c in X.columns if not c.startswith("solv_")]]


def design(cells, memX):
    S = SOLV.loc[cells[S1].values, SOLV_DESC].to_numpy(float)
    M = memX.reindex(cells["entity"].values).to_numpy(float)
    src = (cells["source"].values == "aux").astype(float)[:, None]
    return np.column_stack([S, src, M])


class WithinMembraneContrast:
    """GBM for within-membrane solvent response, fitted by backfitting against membrane fixed effects."""

    def __init__(self, seed, n_backfit=4):
        self.seed, self.n_backfit = seed, n_backfit

    def fit(self, cells, memX):
        X = design(cells, memX)
        y = cells["y"].to_numpy()
        key = (cells["entity"] + "##" + cells["source"]).values
        mu = pd.Series(y).groupby(key).transform("mean").to_numpy()
        for _ in range(self.n_backfit):
            self.model = lgb.LGBMRegressor(random_state=self.seed, **WMC).fit(X, y - mu)
            g = self.model.predict(X)
            mu = pd.Series(y - g).groupby(key).transform("mean").to_numpy()
        return self

    def predict(self, cells, memX):
        return self.model.predict(design(cells, memX))


def multi(cells):
    return cells[cells.groupby(["entity", "source"])[S1].transform("nunique") >= 2]


def estimate_variances(train_cells_multi, aux_cells_multi, memX, f_by_cell_all, seed):
    """sigma2: within-membrane noise around the contrast model (inner CV, LOO-centred);
    tau2: prior variance of the membrane level around the zero-shot level estimate."""
    errs, ns = [], []
    papers = train_cells_multi["paper"].unique()
    if len(papers) >= 5:
        for itr, ite in GroupKFold(5, shuffle=True, random_state=seed).split(papers, groups=papers):
            ptr, pte = set(papers[itr]), set(papers[ite])
            fit_cells = pd.concat([train_cells_multi[train_cells_multi.paper.isin(ptr)],
                                   aux_cells_multi[~aux_cells_multi.paper.isin(pte)]])
            w = WithinMembraneContrast(seed).fit(fit_cells, memX)
            tc = train_cells_multi[train_cells_multi.paper.isin(pte)].copy()
            tc["g"] = w.predict(tc, memX)
            for _, g in tc.groupby("entity"):
                n = len(g)
                r = (g["y"] - g["g"]).to_numpy()
                loo = (r.sum() - r) / (n - 1)
                errs.extend(((r - loo) ** 2 * (n - 1) / n).tolist())
                ns.append(n)
    sigma2 = float(np.mean(errs)) if errs else 0.05
    return sigma2, tau2_from(f_by_cell_all, sigma2)


def tau2_from(resid_cells, sigma2):
    """Moment estimate of the prior variance of the membrane level: E[d_m^2] - sigma2/n_m, d_m = mean zero-shot residual."""
    d = resid_cells.groupby("entity").agg(dm=("r", "mean"), n=("r", "size"))
    return float(max(np.mean(d["dm"] ** 2 - sigma2 / d["n"]), 0.02))


def hift_predict(g_t, g_a, y_a, m0, sigma2, tau2, shrink=True):
    k = len(y_a)
    if k == 0:
        return m0 + g_t, np.sqrt(tau2 + sigma2)
    if not shrink:
        mu = np.mean(y_a - g_a)
        return mu + g_t, np.sqrt(sigma2 / k + sigma2)
    prec = 1 / tau2 + k / sigma2
    mu = (m0 / tau2 + np.sum(y_a - g_a) / sigma2) / prec
    return mu + g_t, np.sqrt(1 / prec + sigma2)


def hift_ensemble_predict(g_t, g_a, pair_ta, y_a, m0, sigma2, tau2, shrink=True):
    """Bayesian membrane level + equal-weight average of two contrast learners (backfitted GBM, pairwise GBM).

    pair_ta[i] is the pairwise-model estimate of y_t - y_a for anchor i. Weights are fixed at 1/2, not tuned.
    """
    k = len(y_a)
    if k == 0:
        return m0 + g_t, np.sqrt(tau2 + sigma2)
    from_anchors = np.mean(y_a + 0.5 * (g_t - g_a) + 0.5 * pair_ta)
    if not shrink:
        return from_anchors, np.sqrt(sigma2 / k + sigma2)
    w0 = (1 / tau2) / (1 / tau2 + k / sigma2)
    return w0 * (m0 + g_t) + (1 - w0) * from_anchors, np.sqrt(1 / (1 / tau2 + k / sigma2) + sigma2)


def additive_offsets(cells):
    m = multi(cells)
    key = m["entity"] + "##" + m["source"]
    a = m.groupby(key)["y"].mean()
    b = pd.Series(0.0, index=m[S1].unique())
    for _ in range(100):
        b = (m["y"] - key.map(a)).groupby(m[S1]).mean()
        b -= b.mean()
        a = (m["y"] - m[S1].map(b)).groupby(key).mean()
    return b


def pairwise_model(cells, memX, seed):
    m = multi(cells)
    pairs = m.merge(m, on=["entity", "source"], suffixes=("_t", "_a"))
    pairs = pairs[pairs[f"{S1}_t"] != pairs[f"{S1}_a"]]
    X = pair_X(pairs[f"{S1}_t"].values, pairs[f"{S1}_a"].values, pairs["y_a"].values,
               memX.reindex(pairs["entity"].values).to_numpy(float))
    return lgb.LGBMRegressor(random_state=seed, **PAIR).fit(X, pairs["y_t"].values - pairs["y_a"].values)


def pair_X(t_solv, a_solv, y_a, mem):
    st = SOLV.loc[t_solv, SOLV_DESC].to_numpy(float)
    sa = SOLV.loc[a_solv, SOLV_DESC].to_numpy(float)
    return np.column_stack([st, st - sa, y_a, mem])


def linear_rc_model(train_cells, center, scale):
    groups = []
    for _, g in train_cells.groupby("entity"):
        Z, _, _ = transfer_design(g[S1], center, scale)
        groups.append((Z, (g["y"] - g["f"]).to_numpy()))
    return (HierarchicalTransfer().fit(groups), HierarchicalTransfer(intercept_only=True).fit(groups))
