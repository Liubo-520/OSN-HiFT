"""Split-conformal intervals: does the calibration split need to respect paper structure?

Test scenarios: random rows (in-distribution), unseen papers, future papers (>=2021).
Calibration: random 25% of training rows vs 25% of training papers held out as a group.
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, ShuffleSplit

from osn_core import DATA, REF, RESULTS, featurize

warnings.filterwarnings("ignore")
LGB = dict(n_estimators=700, learning_rate=0.03, num_leaves=31, min_child_samples=10, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, n_jobs=8)
LEVELS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])


def conformal_q(scores, level):
    n = len(scores)
    k = min(int(np.ceil((n + 1) * level)), n)
    return np.sort(scores)[k - 1]


def evaluate(d, tr, te, seed, rows, scenario):
    dtr = d.iloc[tr].reset_index(drop=True)
    dte = d.iloc[te]
    for cal_mode in ("Random-row calibration", "Paper-held-out calibration"):
        if cal_mode == "Random-row calibration":
            fit_i, cal_i = next(ShuffleSplit(1, test_size=0.25, random_state=seed).split(dtr))
        else:
            fit_i, cal_i = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(dtr, groups=dtr[REF]))
        dfit, dcal = dtr.iloc[fit_i], dtr.iloc[cal_i]
        Xf, vocab = featurize(dfit)
        m = lgb.LGBMRegressor(random_state=seed, **LGB).fit(Xf, dfit["y"])
        s_cal = np.abs(dcal["y"].values - m.predict(featurize(dcal, vocab=vocab)[0]))
        err = np.abs(dte["y"].values - m.predict(featurize(dte, vocab=vocab)[0]))
        for lv in LEVELS:
            q = conformal_q(s_cal, lv)
            cov = err <= q
            rows.append(dict(scenario=scenario, calibration=cal_mode, seed=seed, level=lv, q=q, width=2 * q,
                             coverage=cov.mean(), n_test=len(err),
                             paper_coverage_median=pd.Series(cov).groupby(dte[REF].values).mean().median()))


def main():
    d = pd.read_csv(DATA / "pure_solvent_permeance_domain.csv", low_memory=False)
    d["y"] = d["logP"]
    d = d.reset_index(drop=True)
    rows = []
    for seed in (0, 1, 2):
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(d):
            evaluate(d, tr, te, seed, rows, "Random rows")
        for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(d, groups=d[REF]):
            evaluate(d, tr, te, seed, rows, "Unseen papers")
        known = d["year"].notna()
        evaluate(d, np.where(known & (d.year <= 2020))[0], np.where(known & (d.year >= 2021))[0], seed, rows,
                 "Future papers (≥2021)")
        print("seed", seed, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "conformal_coverage.csv", index=False)
    summ = res.groupby(["scenario", "calibration", "level"])[["coverage", "width"]].mean().round(3)
    print(summ.to_string())


if __name__ == "__main__":
    main()
