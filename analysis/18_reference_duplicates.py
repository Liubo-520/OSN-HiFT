"""Audit aliases and refit a raw-reference sensitivity control from fresh folds."""
import importlib.util
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from osn_core import DATA, HERE, REF, RESULTS

spec = importlib.util.spec_from_file_location("bench", HERE / "11_benchmark.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
out = {"protocol": "canonical IDs in all production stages; raw-group control uses fresh refits"}
for tag, f in [("pure_solvent_permeance", "pure_solvent_permeance_domain.csv"),
               ("solution_permeance", "solution_permeance_domain.csv"), ("rejection", "rejection_long.csv")]:
    d = pd.read_csv(DATA / f, low_memory=False)
    g = d.groupby(REF)["report.reference_raw"].nunique()
    dup = g[g > 1].index
    out[tag] = dict(papers_raw=int(d["report.reference_raw"].nunique()),
                    papers_distinct=int(d[REF].nunique()), duplicated_papers=len(dup),
                    records_affected=int(d[REF].isin(dup).sum()), rows_retained=len(d))
d = pd.read_csv(DATA / "pure_solvent_permeance_domain.csv", low_memory=False).reset_index(drop=True)
d["y"] = d["logP"]
scores = []
for seed in bench.SEEDS:
    pred = np.full(len(d), np.nan)
    for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(d, groups=d["report.reference_raw"]):
        pred[te] = bench.fit_predict("LightGBM", d.iloc[tr], d.iloc[te], seed)
    scores.append(float(bench.r2(d.y.values, pred)))
    print("raw-reference sensitivity", seed, scores[-1], flush=True)
metrics = pd.read_csv(RESULTS / "benchmark_metrics.csv")
canonical = metrics.query("scenario == 'Unseen paper' and model == 'LightGBM'").R2.iloc[0]
out["unseen_paper_lightgbm_r2"] = {"raw": float(np.mean(scores)), "merged": float(canonical)}
(RESULTS / "reference_duplicates.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2), flush=True)
