"""Fitting and evaluation of few-shot transfer models, shared by CV, nested-selection and external validation.

The standard method set reproduces 13_fewshot.py exactly (same fitted components, same anchor draws).
`grid=True` additionally evaluates every model configuration explored during HiFT development, for
selection-aware (nested) validation.
"""
import itertools
import zlib

import numpy as np
import pandas as pd

import fewshot_lib as fl
from osn_core import REF, S1, SOLV, featurize, transfer_design

KS = (1, 2, 3)
DRAWS = 10
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
SHRINK = ("none", "tau_multi", "tau_all")
FROZEN = "w=0.50|shrink=none|aux=1"


def _quick_sigma2(cells, memX, wmc):
    """Moment within-membrane residual variance around a fitted contrast model (selection only)."""
    if cells.empty:
        return 0.05
    r = cells["y"].to_numpy() - wmc.predict(cells, memX)
    mu = pd.Series(r).groupby(cells["entity"].values).transform("mean").to_numpy()
    df = len(r) - cells["entity"].nunique()
    return float(max(np.sum((r - mu) ** 2) / max(df, 1), 0.01))


def fit_components(dtr, f_oof, test_rows, aux, seed, tag, center, scale, quick_var=False):
    ctr = fl.cells_of(dtr, f_oof, "target")
    caux = fl.cells_of(aux, np.full(len(aux), np.nan), "aux")
    _, vocab = featurize(dtr, extra_numeric=fl.extra_for(tag))
    memX = fl.membrane_features(pd.concat([dtr, test_rows, aux], ignore_index=True), vocab)
    ctr_m, caux_m = fl.multi(ctr), fl.multi(caux)
    c = dict(memX=memX, center=center, scale=scale)
    c["wmc"] = fl.WithinMembraneContrast(seed).fit(pd.concat([ctr_m, caux_m]), memX)
    c["wmc_noaux"] = fl.WithinMembraneContrast(seed).fit(ctr_m, memX)
    resid_multi = ctr_m.assign(r=ctr_m["y"] - ctr_m["f"])[["entity", "r"]]
    resid_all = ctr.assign(r=ctr["y"] - ctr["f"])[["entity", "r"]]
    if quick_var:
        c["s2"] = _quick_sigma2(ctr_m, memX, c["wmc"])
        c["s2n"] = _quick_sigma2(ctr_m, memX, c["wmc_noaux"])
        c["t2"] = fl.tau2_from(resid_multi, c["s2"])
        c["t2n"] = fl.tau2_from(resid_multi, c["s2n"])
    else:
        c["s2"], c["t2"] = fl.estimate_variances(ctr_m, caux_m, memX, resid_multi, seed)
        c["s2n"], c["t2n"] = fl.estimate_variances(ctr_m, caux_m.iloc[0:0], memX, resid_multi, seed)
    c["t2_all"] = fl.tau2_from(resid_all, c["s2"])
    c["t2n_all"] = fl.tau2_from(resid_all, c["s2n"])
    c["boff"] = fl.additive_offsets(pd.concat([ctr, caux]))
    c["pgbm"] = fl.pairwise_model(pd.concat([ctr, caux]), memX, seed)
    c["pgbm_noaux"] = fl.pairwise_model(ctr, memX, seed)
    c["lin_rc"], c["lin_int"] = fl.linear_rc_model(ctr, center, scale)
    c["n_train_multi"] = ctr_m["entity"].nunique()
    c["n_aux_multi"] = caux_m["entity"].nunique()
    return c


def config_predict(gt, ga, pta, ya, m0, s2, t2, w, shrink):
    k = len(ya)
    if k == 0:
        return m0 + gt, np.sqrt(t2 + s2)
    from_anchors = np.mean(ya + w * (gt - ga) + (1 - w) * pta)
    if shrink == "none":
        return from_anchors, np.sqrt(s2 / k + s2)
    w0 = (1 / t2) / (1 / t2 + k / s2)
    return w0 * (m0 + gt) + (1 - w0) * from_anchors, np.sqrt(1 / (1 / t2 + k / s2) + s2)


def evaluate(cte, comp, seed, fold, tag, grid=False, standard=True, min_solvents=2, ks=KS):
    """cte: test cells with columns entity, solvent, y, f, paper, material. Returns long-format prediction rows."""
    c = comp
    memX, center, scale = c["memX"], c["center"], c["scale"]
    cte = cte[cte.groupby("entity")[S1].transform("nunique") >= min_solvents].reset_index(drop=True)
    if cte.empty:
        return []
    cte["g"] = c["wmc"].predict(cte, memX)
    cte["g0"] = c["wmc_noaux"].predict(cte, memX)
    s2, t2, t2_all, s2n, t2n, t2n_all = (c[k] for k in ("s2", "t2", "t2_all", "s2n", "t2n", "t2n_all"))
    out = []
    for ent, g in cte.groupby("entity"):
        g = g.reset_index(drop=True)
        n = len(g)
        solv = g[S1].tolist()
        Z, _, _ = transfer_design(solv, center, scale)
        eta = SOLV.loc[solv, "log_eta"].to_numpy()
        vme = SOLV.loc[solv, "log_Vm_eta"].to_numpy()
        b = g[S1].map(c["boff"]).fillna(0.0).to_numpy()
        y, f, gg, g0 = (g[col].to_numpy() for col in ("y", "f", "g", "g0"))
        m0, m0n = np.mean(f - gg), np.mean(f - g0)
        mem = np.repeat(memX.loc[[ent]].to_numpy(float), n * n, axis=0)
        ti, ai = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        PX = fl.pair_X(np.array(solv)[ti.ravel()], np.array(solv)[ai.ravel()], y[ai.ravel()], mem)
        gpair = c["pgbm"].predict(PX).reshape(n, n)
        gpair0 = c["pgbm_noaux"].predict(PX).reshape(n, n)
        rng = np.random.default_rng(seed * 1000 + fold * 100 + zlib.crc32(ent.encode()) % 97)
        base = dict(tag=tag, seed=seed, fold=fold, entity=ent, paper=g["paper"][0], material=g["material"][0],
                    n_solvents=n)
        variants = {1: (gg, gpair, m0, s2, {"tau_multi": t2, "tau_all": t2_all}),
                    0: (g0, gpair0, m0n, s2n, {"tau_multi": t2n, "tau_all": t2n_all})}

        def grid_rows(common, k, di, anc, t, A, ra):
            rows = []
            for aux, (G, PP, M0, S2, T2s) in variants.items():
                for w in WEIGHTS:
                    for sh in SHRINK:
                        if k == 0 and (w != WEIGHTS[0] or sh != "none"):
                            continue
                        T2 = T2s.get(sh, T2s["tau_multi"])
                        p, s = config_predict(G[t], G[A], PP[t, A], y[A], M0, S2, T2, w, sh)
                        name = f"w={w:.2f}|shrink={sh}|aux={aux}" if k > 0 else f"k0|aux={aux}"
                        rows.append(dict(common, k=k, draw=di, anchors=anc, method=name, pred=p, sd=s))
            ml, sl = c["lin_rc"].predict(Z[[t]], Z[A], ra) if k > 0 else c["lin_rc"].predict(Z[[t]])
            mi, si = c["lin_int"].predict(Z[[t]], Z[A], ra) if k > 0 else c["lin_int"].predict(Z[[t]])
            rows.append(dict(common, k=k, draw=di, anchors=anc, method="linear_rc", pred=f[t] + ml[0], sd=sl[0]))
            rows.append(dict(common, k=k, draw=di, anchors=anc, method="level_shift", pred=f[t] + mi[0], sd=si[0]))
            return rows

        for t in range(n):
            common = dict(base, target=solv[t], y=y[t])
            if standard:
                p_h, s_h = fl.hift_predict(gg[t], gg[[]], y[[]], m0, s2, t2)
                p_hn, s_hn = fl.hift_predict(g0[t], g0[[]], y[[]], m0n, s2n, t2n)
                ml, sl = c["lin_rc"].predict(Z[[t]])
                mi, si = c["lin_int"].predict(Z[[t]])
                for meth, p, s in [("Zero-shot GBM", f[t], np.nan), ("HiFT", p_h, s_h),
                                   ("HiFT − auxiliary data", p_hn, s_hn),
                                   ("Linear random-coefficient", f[t] + ml[0], sl[0]),
                                   ("Bayesian level shift", f[t] + mi[0], si[0])]:
                    out.append(dict(common, k=0, draw=0, anchors="", method=meth, pred=p, sd=s))
            if grid:
                out.extend(grid_rows(common, 0, 0, "", t, [], y[[]]))
            others = [j for j in range(n) if j != t]
            for k in ks:
                if len(others) < k:
                    continue
                combos = list(itertools.combinations(others, k))
                if k > 1 and len(combos) > DRAWS:
                    combos = [combos[i] for i in rng.choice(len(combos), DRAWS, replace=False)]
                for di, A in enumerate(combos):
                    A = list(A)
                    ra = y[A] - f[A]
                    anc = "|".join(solv[j] for j in A)
                    if grid:
                        out.extend(grid_rows(common, k, di, anc, t, A, ra))
                    if not standard:
                        continue
                    ml, sl = c["lin_rc"].predict(Z[[t]], Z[A], ra)
                    mi, si = c["lin_int"].predict(Z[[t]], Z[A], ra)
                    p_h, s_h = fl.hift_ensemble_predict(gg[t], gg[A], gpair[t, A], y[A], m0, s2, t2, shrink=False)
                    p_hs, s_hs = fl.hift_ensemble_predict(gg[t], gg[A], gpair[t, A], y[A], m0, s2, t2)
                    p_hc, s_hc = fl.hift_predict(gg[t], gg[A], y[A], m0, s2, t2, shrink=False)
                    p_hn, s_hn = fl.hift_ensemble_predict(g0[t], g0[A], gpair0[t, A], y[A], m0n, s2n, t2n,
                                                          shrink=False)
                    if k >= 2 and np.ptp(eta[A]) > 0.05:
                        slope, icpt = np.polyfit(eta[A], y[A], 1)
                        p_fit = icpt + slope * eta[t]
                    else:
                        p_fit = np.mean(y[A] + eta[A]) - eta[t]
                    preds = [
                        ("Zero-shot GBM", f[t], np.nan),
                        ("Viscosity scaling", np.mean(y[A] + eta[A]) - eta[t], np.nan),
                        ("Vm/η scaling", np.mean(y[A] - vme[A]) + vme[t], np.nan),
                        ("Per-membrane η fit", p_fit, np.nan),
                        ("Additive solvent offset", np.mean(y[A] - b[A]) + b[t], np.nan),
                        ("GBM + mean shift", f[t] + ra.mean(), np.nan),
                        ("Pairwise transfer GBM", np.mean(y[A] + gpair[t, A]), np.nan),
                        ("Bayesian level shift", f[t] + mi[0], si[0]),
                        ("Linear random-coefficient", f[t] + ml[0], sl[0]),
                        ("HiFT − pairwise learner", p_hc, s_hc),
                        ("HiFT − auxiliary data", p_hn, s_hn),
                        ("HiFT + zero-shot prior shrinkage", p_hs, s_hs),
                        ("HiFT", p_h, s_h),
                    ]
                    for meth, p, s in preds:
                        out.append(dict(common, k=k, draw=di, anchors=anc, method=meth, pred=p, sd=s))
    return out
