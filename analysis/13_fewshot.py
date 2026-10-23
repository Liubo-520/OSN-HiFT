"""Few-shot cross-solvent transfer for membranes from unseen papers.

Each membrane in held-out papers measured in >=2 solvents: every solvent is predicted from k anchor solvents of the
same membrane (k=1: all single anchors; k=2,3: up to 10 random anchor sets). Auxiliary within-membrane contrasts come
from the other permeance type, restricted to training papers.
"""
import itertools
import sys
import zlib

import numpy as np
import pandas as pd

import fewshot_lib as fl
from osn_core import REF, RESULTS, S1, SOLV, featurize, transfer_design

KS = (1, 2, 3)
DRAWS = 10


def run(tag, seeds=(0, 1, 2)):
    d = fl.load_domain(tag)
    aux_all = fl.load_domain(fl.AUX[tag])
    dom_solvents = sorted(d[S1].unique())
    _, center, scale = transfer_design(dom_solvents)
    out, params = [], []
    for seed in seeds:
        for fold in range(5):
            prep = fl.prepare_fold(tag, seed, fold, d)
            dtr = d.iloc[prep["tr"]].reset_index(drop=True)
            dte = d.iloc[prep["te"]].reset_index(drop=True)
            test_papers = set(dte[REF])
            aux = aux_all[~aux_all[REF].isin(test_papers)].reset_index(drop=True)

            ctr = fl.cells_of(dtr, prep["f_oof"], "target")
            cte = fl.cells_of(dte, prep["f_te"], "target")
            caux = fl.cells_of(aux, np.full(len(aux), np.nan), "aux")
            _, vocab = featurize(dtr, extra_numeric=fl.extra_for(tag))
            memX = fl.membrane_features(pd.concat([dtr, dte, aux], ignore_index=True), vocab)

            ctr_m, caux_m = fl.multi(ctr), fl.multi(caux)
            wmc = fl.WithinMembraneContrast(seed).fit(pd.concat([ctr_m, caux_m]), memX)
            wmc_noaux = fl.WithinMembraneContrast(seed).fit(ctr_m, memX)
            resid_multi = ctr_m.assign(r=ctr_m["y"] - ctr_m["f"])[["entity", "r"]]
            resid_all = ctr.assign(r=ctr["y"] - ctr["f"])[["entity", "r"]]
            s2, t2 = fl.estimate_variances(ctr_m, caux_m, memX, resid_multi, seed)
            t2_all = fl.tau2_from(resid_all, s2)
            s2n, t2n = fl.estimate_variances(ctr_m, caux_m.iloc[0:0], memX, resid_multi, seed)
            boff = fl.additive_offsets(pd.concat([ctr, caux]))
            pgbm = fl.pairwise_model(pd.concat([ctr, caux]), memX, seed)
            pgbm_noaux = fl.pairwise_model(ctr, memX, seed)
            lin_rc, lin_int = fl.linear_rc_model(ctr, center, scale)
            params.append(dict(tag=tag, seed=seed, fold=fold, sigma2=s2, tau2=t2, tau2_all=t2_all,
                               sigma2_noaux=s2n, tau2_noaux=t2n,
                               n_train_multi=ctr_m["entity"].nunique(), n_aux_multi=caux_m["entity"].nunique()))

            cte = cte[cte.groupby("entity")[S1].transform("nunique") >= 2].reset_index(drop=True)
            cte["g"] = wmc.predict(cte, memX)
            cte["g0"] = wmc_noaux.predict(cte, memX)
            for ent, g in cte.groupby("entity"):
                g = g.reset_index(drop=True)
                n = len(g)
                solv = g[S1].tolist()
                Z, _, _ = transfer_design(solv, center, scale)
                eta = SOLV.loc[solv, "log_eta"].to_numpy()
                vme = SOLV.loc[solv, "log_Vm_eta"].to_numpy()
                b = g[S1].map(boff).fillna(0.0).to_numpy()
                y, f, gg, g0 = (g[c].to_numpy() for c in ("y", "f", "g", "g0"))
                m0, m0n = np.mean(f - gg), np.mean(f - g0)
                mem = np.repeat(memX.loc[[ent]].to_numpy(float), n * n, axis=0)
                ti, ai = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
                PX = fl.pair_X(np.array(solv)[ti.ravel()], np.array(solv)[ai.ravel()], y[ai.ravel()], mem)
                gpair = pgbm.predict(PX).reshape(n, n)
                gpair0 = pgbm_noaux.predict(PX).reshape(n, n)
                rng = np.random.default_rng(seed * 1000 + fold * 100 + zlib.crc32(ent.encode()) % 97)
                base = dict(tag=tag, seed=seed, fold=fold, entity=ent, paper=g["paper"][0],
                            material=g["material"][0], n_solvents=n)
                for t in range(n):
                    common = dict(base, target=solv[t], y=y[t])
                    p_h, s_h = fl.hift_predict(gg[t], gg[[]], y[[]], m0, s2, t2)
                    p_ha, s_ha = fl.hift_predict(gg[t], gg[[]], y[[]], m0, s2, t2_all)
                    p_hn, s_hn = fl.hift_predict(g0[t], g0[[]], y[[]], m0n, s2n, t2n)
                    ml, sl = lin_rc.predict(Z[[t]])
                    mi, si = lin_int.predict(Z[[t]])
                    for meth, p, s in [("Zero-shot GBM", f[t], np.nan), ("HiFT", p_h, s_h),
                                       ("HiFT − auxiliary data", p_hn, s_hn),
                                       ("Linear random-coefficient", f[t] + ml[0], sl[0]),
                                       ("Bayesian level shift", f[t] + mi[0], si[0])]:
                        out.append(dict(common, k=0, draw=0, anchors="", method=meth, pred=p, sd=s))
                    others = [j for j in range(n) if j != t]
                    for k in KS:
                        if len(others) < k:
                            continue
                        combos = list(itertools.combinations(others, k))
                        if k > 1 and len(combos) > DRAWS:
                            combos = [combos[i] for i in rng.choice(len(combos), DRAWS, replace=False)]
                        for di, A in enumerate(combos):
                            A = list(A)
                            ra = y[A] - f[A]
                            ml, sl = lin_rc.predict(Z[[t]], Z[A], ra)
                            mi, si = lin_int.predict(Z[[t]], Z[A], ra)
                            p_h, s_h = fl.hift_ensemble_predict(gg[t], gg[A], gpair[t, A], y[A], m0, s2, t2,
                                                                shrink=False)
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
                            anc = "|".join(solv[j] for j in A)
                            for meth, p, s in preds:
                                out.append(dict(common, k=k, draw=di, anchors=anc, method=meth, pred=p, sd=s))
            print(f"{tag} seed {seed} fold {fold}: test membranes={cte['entity'].nunique()} "
                  f"sigma={np.sqrt(s2):.3f} tau={np.sqrt(t2):.3f} train multi={ctr_m['entity'].nunique()} "
                  f"aux multi={caux_m['entity'].nunique()}", flush=True)
    res = pd.DataFrame(out)
    res["abs_err"] = (res["y"] - res["pred"]).abs()
    res.to_csv(RESULTS / f"fewshot_{tag}.csv", index=False)
    pd.DataFrame(params).to_csv(RESULTS / f"hift_params_{tag}.csv", index=False)
    return res


def summarize(res, min_solvents):
    r = res[res.n_solvents >= min_solvents]
    cell = r.groupby(["seed", "method", "k", "entity", "target"], as_index=False)["abs_err"].mean()
    return cell.groupby(["method", "k"])["abs_err"].mean().unstack("k").round(3)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    seeds = tuple(int(s) for s in sys.argv[2].split(",")) if len(sys.argv) > 2 else (0, 1, 2)
    if which in ("both", "sp"):
        print(summarize(run("pure_solvent_permeance", seeds), 4).to_string(), flush=True)
    if which in ("both", "pm"):
        print(summarize(run("solution_permeance", seeds), 3).to_string(), flush=True)
