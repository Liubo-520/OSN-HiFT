"""Independent verification of HiFT, addressing selection on the reported folds.

Layers
------
1. Frozen configuration. The reported HiFT (equal-weight ensemble, no prior
   shrinkage, auxiliary contrasts included) is locked as FROZEN and is the only
   configuration used for confirmatory numbers.
2. Untouched internal subset. Membranes with 2–3 solvents (pure-solvent) or 2
   solvents (solution permeance) were excluded from the primary tables and from
   configuration decisions; they are scored with the already-computed frozen HiFT.
3. Selection-free nested CV. Every candidate (w, shrinkage, auxiliary) is scored
   on inner grouped folds of training papers only; the winner is evaluated on the
   outer test papers. Frozen and an oracle (best on the outer fold) are compared
   on the same cells.
4. Post-snapshot external papers. Models are trained on the entire 2025-04-08
   snapshot and tested on papers that first appear in the live OMD4SRNF export of
   2026-10-23 (49 new papers). The frozen configuration is not re-tuned.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import fewshot_lib as fl
import fewshot_summary as fsum
import transfer_eval as te
from reference_identity import normalize_references
from osn_core import (DATA, NUMERIC, PM, R1, RAW, REF, RESULTS, S1, SOLVENT_ALIASES, SP,
                      load as load_snapshot, material_class, organic_domain, reconstruct_years,
                      transfer_design)

EXT_RAW = RAW.parent / "external" / "OMD_SRNF_2026-10-23.csv"
SEEDS = (0, 1, 2)
MIN_REPORT = {"pure_solvent_permeance": 4, "solution_permeance": 3}
MIN_UNTOUCH = {"pure_solvent_permeance": (2, 3), "solution_permeance": (2, 2)}
HEADLINE = ["Zero-shot GBM", "Viscosity scaling", "Vm/η scaling", "Additive solvent offset",
            "GBM + mean shift", "Pairwise transfer GBM", "Bayesian level shift",
            "Linear random-coefficient", "HiFT"]


def _mae(df, method, k, min_solvents):
    r = df[(df.method == method) & (df.k == k) & (df.n_solvents >= min_solvents)]
    if r.empty:
        return np.nan
    return float(r.groupby(["seed", "entity", "target"], as_index=False)["abs_err"].mean()["abs_err"].mean())


def _boot(df, method, k, min_solvents):
    r = df[(df.method == method) & (df.k == k) & (df.n_solvents >= min_solvents)].copy()
    if r.empty:
        return dict(MAE=None, CI=[None, None], n_cells=0, n_membranes=0, n_papers=0)
    cell = r.groupby(["seed", "paper", "entity", "target"], as_index=False)["abs_err"].mean()
    mean = float(cell["abs_err"].mean())
    if cell["paper"].nunique() < 2:
        lo = hi = None
    else:
        _, lo, hi = fsum.boot_mean(cell, "abs_err")
        lo, hi = round(lo, 3), round(hi, 3)
    return dict(MAE=round(mean, 3), CI=[lo, hi],
                n_cells=int(cell.groupby(["entity", "target"]).ngroups),
                n_membranes=int(cell["entity"].nunique()), n_papers=int(cell["paper"].nunique()))


def _gain(zs, h):
    if zs is None or h is None or zs == 0:
        return None
    return round(100 * (1 - h / zs), 1)


def load_omd(path):
    df = normalize_references(pd.read_csv(path, low_memory=False))
    num = [SP, PM, R1, "soluteChoice.solute1.molecularWeight", "testConditions.hydraulicP",
           "testConditions.temperature"] + [c for c in NUMERIC if c in df.columns]
    for c in num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[S1] = df[S1].replace(SOLVENT_ALIASES)
    years = reconstruct_years(df)
    df["year"] = df[REF].map(years["year"]).astype(float)
    df["year_source"] = df[REF].map(years["year_source"])
    df["entity"] = df[REF].astype(str) + " || " + df["name"].astype(str)
    df["material"] = df.apply(material_class, axis=1)
    return df


def domain_from(df, tag):
    target = SP if tag == "pure_solvent_permeance" else PM
    out = organic_domain(df, target)
    out["y"] = out["logP"]
    out["log_solute_MW"] = np.log10(pd.to_numeric(out["soluteChoice.solute1.molecularWeight"], errors="coerce"))
    return out.reset_index(drop=True)


def inventory():
    snap = load_snapshot()
    ext = load_omd(EXT_RAW)
    new_papers = sorted(set(ext[REF]) - set(snap[REF]))
    info = dict(snapshot_papers=int(snap[REF].nunique()), live_papers=int(ext[REF].nunique()),
                new_papers=len(new_papers), new_paper_list=new_papers)
    for tag in ("pure_solvent_permeance", "solution_permeance"):
        d0, de = domain_from(snap, tag), domain_from(ext, tag)
        de_new = de[de[REF].isin(new_papers)].copy()
        ns = de_new.groupby("entity")[S1].nunique() if len(de_new) else pd.Series(dtype=int)
        info[tag] = dict(
            snapshot_rows=int(len(d0)), live_rows=int(len(de)), new_rows=int(len(de_new)),
            new_domain_papers=int(de_new[REF].nunique()) if len(de_new) else 0,
            new_membranes=int(de_new["entity"].nunique()) if len(de_new) else 0,
            multi_ge2=int((ns >= 2).sum()), multi_ge3=int((ns >= 3).sum()),
            multi_ge4=int((ns >= 4).sum()), n_solvents=ns.value_counts().sort_index().to_dict(),
            materials=de_new["material"].value_counts().to_dict() if len(de_new) else {},
            year_min=None if de_new["year"].isna().all() else float(de_new["year"].min()),
            year_max=None if de_new["year"].isna().all() else float(de_new["year"].max()),
        )
        de_new.to_csv(DATA / f"{tag}_external_new.csv", index=False)
    (RESULTS / "indep_inventory.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps({k: info[k] for k in info if k != "new_paper_list"}, indent=2))
    return info


def untouched():
    out = {}
    for tag in ("pure_solvent_permeance", "solution_permeance"):
        raw = pd.read_csv(RESULTS / f"fewshot_{tag}.csv", low_memory=False)
        lo, hi = MIN_UNTOUCH[tag]
        r = raw[raw.n_solvents.between(lo, hi)].copy()
        r.to_csv(RESULTS / f"fewshot_untouched_{tag}.csv", index=False)
        snap = fl.load_domain(tag)
        ns = snap.groupby("entity")[S1].nunique()
        papers4 = set(snap.loc[snap.entity.isin(ns[ns >= MIN_REPORT[tag]].index), REF])
        ents = r["entity"].unique()
        papers_u = set(r["paper"].unique())
        block = {}
        for meth in HEADLINE:
            block[meth] = {f"k={k}": _boot(r, meth, k, lo) for k in (0, 1, 2, 3) if (r.method == meth).any()}
        zs, h1 = block["Zero-shot GBM"].get("k=0", {}).get("MAE"), block["HiFT"].get("k=1", {}).get("MAE")
        out[tag] = dict(
            n_membranes=int(r["entity"].nunique()), n_papers=int(r["paper"].nunique()),
            n_from_primary_papers=int(sum(p in papers4 for p in papers_u)),
            n_exclusive_papers=int(sum(p not in papers4 for p in papers_u)),
            n_solvents=r.groupby("entity")["n_solvents"].first().value_counts().sort_index().to_dict(),
            metrics=block, gain_k1=_gain(zs, h1),
        )
        print(tag, "untouched", out[tag]["n_membranes"], "membranes", out[tag]["n_papers"],
              "papers; HiFT k=1", h1, "gain", out[tag]["gain_k1"])
    (RESULTS / "indep_untouched.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _cell_mae(df, min_solvents, k=1):
    r = df[(df.k == k) & (df.n_solvents >= min_solvents)]
    if r.empty:
        return pd.Series(dtype=float)
    return r.groupby("method")["abs_err"].mean()


def nested_base_predictions(dtr, dte, seed, tag):
    """Refit within this inner split; no outer-training OOF predictions are reused.

    Training residuals use paper-disjoint cross-fitting restricted to dtr.
    Validation predictions use a model fitted to all and only dtr.
    """
    if set(dtr[REF]) & set(dte[REF]):
        raise ValueError("Inner training and validation papers overlap")
    extra = fl.extra_for(tag)
    f_te = fl.gbm_fit_predict(dtr, dte, seed, extra)
    f_oof = np.full(len(dtr), np.nan)
    n_splits = min(5, dtr[REF].nunique())
    if n_splits < 2:
        raise ValueError("At least two inner-training papers are required")
    for tr, te_idx in GroupKFold(n_splits, shuffle=True, random_state=300 + seed).split(dtr, groups=dtr[REF]):
        f_oof[te_idx] = fl.gbm_fit_predict(dtr.iloc[tr], dtr.iloc[te_idx], seed, extra)
    if not np.isfinite(f_oof).all():
        raise RuntimeError("Incomplete inner-training cross-fitted predictions")
    return f_oof, f_te


def nested(tag, seeds=SEEDS):
    path_outer = RESULTS / f"indep_outer_grid_{tag}.csv"
    path_inner = RESULTS / f"indep_nested_inner_{tag}.csv"
    path_choice = RESULTS / f"indep_nested_choices_{tag}.csv"
    d = fl.load_domain(tag)
    aux_all = fl.load_domain(fl.AUX[tag])
    center, scale = transfer_design(sorted(d[S1].unique()))[1:]
    min_rep = MIN_REPORT[tag]
    outer_rows, inner_rows, choices = [], [], []

    if path_outer.exists():
        outer = pd.read_csv(path_outer, low_memory=False)
        print(tag, "reusing", path_outer.name, len(outer), "rows")
    else:
        outer = None
    done_choices = set()
    if path_choice.exists():
        old_ch = pd.read_csv(path_choice)
        done_choices = set(zip(old_ch.seed.astype(int), old_ch.fold.astype(int)))
        print(tag, "already selected", sorted(done_choices))

    for seed in seeds:
        for fold in range(5):
            prep = fl.prepare_fold(tag, seed, fold, d)
            dtr = d.iloc[prep["tr"]].reset_index(drop=True)
            dte = d.iloc[prep["te"]].reset_index(drop=True)
            test_papers = set(dte[REF])
            aux = aux_all[~aux_all[REF].isin(test_papers)].reset_index(drop=True)
            have = (outer is not None and ((outer.seed == seed) & (outer.fold == fold)).any())
            if not have:
                comp = te.fit_components(dtr, prep["f_oof"], dte, aux, seed, tag, center, scale)
                cte = fl.cells_of(dte, prep["f_te"], "target")
                rows = te.evaluate(cte, comp, seed, fold, tag, grid=True, standard=True, min_solvents=2)
                part = pd.DataFrame(rows)
                part["abs_err"] = (part["y"] - part["pred"]).abs()
                outer_rows.append(part)
                print(f"{tag} outer seed {seed} fold {fold}: membranes={cte.entity.nunique()} "
                      f"grid_rows={len(part)}", flush=True)
            else:
                part = outer[(outer.seed == seed) & (outer.fold == fold)]

            if (seed, fold) in done_choices:
                print(f"{tag} skip inner seed {seed} fold {fold}", flush=True)
                continue

            papers = dtr[REF].unique()
            n_inner = 3 if len(papers) >= 6 else 2
            inner_mae = {}
            for inner_i, (i_tr, i_te) in enumerate(
                    GroupKFold(n_inner, shuffle=True, random_state=200 + seed).split(papers, groups=papers)):
                ptr, pte = set(papers[i_tr]), set(papers[i_te])
                mask_tr = dtr[REF].isin(ptr).to_numpy()
                mask_te = dtr[REF].isin(pte).to_numpy()
                if mask_te.sum() == 0 or mask_tr.sum() == 0:
                    continue
                inner_dtr = dtr.iloc[mask_tr].reset_index(drop=True)
                inner_dte = dtr.iloc[mask_te].reset_index(drop=True)
                inner_aux = aux_all[~aux_all[REF].isin(test_papers | pte)].reset_index(drop=True)
                inner_f_oof, inner_f_te = nested_base_predictions(inner_dtr, inner_dte, seed, tag)
                icomp = te.fit_components(inner_dtr, inner_f_oof, inner_dte, inner_aux,
                                          seed, tag, center, scale, quick_var=True)
                icte = fl.cells_of(inner_dte, inner_f_te, "target")
                irows = te.evaluate(icte, icomp, seed, fold, tag, grid=True, standard=False,
                                    min_solvents=2, ks=(1,))
                if not irows:
                    continue
                idf = pd.DataFrame(irows)
                idf["abs_err"] = (idf["y"] - idf["pred"]).abs()
                idf["inner"] = inner_i
                inner_rows.append(idf)
                mae = _cell_mae(idf, min_solvents=2, k=1)
                for meth, v in mae.items():
                    inner_mae.setdefault(meth, []).append(v)
                print(f"{tag} inner seed {seed} fold {fold}.{inner_i}: "
                      f"test_mem={icte['entity'].nunique()} configs={len(mae)}", flush=True)

            mean_inner = {m: float(np.mean(vs)) for m, vs in inner_mae.items() if m.startswith("w=")}
            selected = min(mean_inner, key=mean_inner.get) if mean_inner else te.FROZEN
            grid_part = part[part.method.str.startswith("w=")]
            oracle_mae = _cell_mae(grid_part, min_rep, k=1)
            oracle = oracle_mae.idxmin() if len(oracle_mae) else te.FROZEN
            choices.append(dict(tag=tag, seed=seed, fold=fold, selected=selected,
                                selected_inner_mae=mean_inner.get(selected),
                                frozen_inner_mae=mean_inner.get(te.FROZEN),
                                oracle=oracle,
                                frozen_outer_mae=_mae(part, te.FROZEN, 1, min_rep),
                                selected_outer_mae=_mae(part, selected, 1, min_rep),
                                oracle_outer_mae=_mae(part, oracle, 1, min_rep),
                                n_inner_configs=len(mean_inner)))
            print(f"{tag} select seed {seed} fold {fold}: {selected}  oracle={oracle}", flush=True)

    if outer_rows:
        outer = pd.concat([outer] + outer_rows, ignore_index=True) if outer is not None else pd.concat(outer_rows, ignore_index=True)
        outer.to_csv(path_outer, index=False)
    if inner_rows:
        prev = pd.read_csv(path_inner, low_memory=False) if path_inner.exists() else None
        inner = pd.concat(([prev] if prev is not None else []) + inner_rows, ignore_index=True)
        inner.to_csv(path_inner, index=False)
    ch = pd.DataFrame(choices)
    if path_choice.exists() and not choices:
        ch = pd.read_csv(path_choice)
    else:
        if path_choice.exists() and choices:
            old = pd.read_csv(path_choice)
            key = ["seed", "fold"]
            ch = pd.concat([old, ch], ignore_index=True).drop_duplicates(key, keep="last")
        ch.to_csv(path_choice, index=False)
    return summarize_nested(tag, outer, ch)


def summarize_nested(tag, outer, ch):
    min_rep = MIN_REPORT[tag]
    mapped = []
    for row in ch.itertuples():
        part = outer[(outer.seed == row.seed) & (outer.fold == row.fold)]
        for lab, meth in [("Frozen HiFT", te.FROZEN), ("Nested-selected HiFT", row.selected),
                          ("Oracle (test-tuned)", row.oracle)]:
            sl = part[part.method == meth].copy()
            sl["method"] = lab
            mapped.append(sl)
        for meth in HEADLINE:
            mapped.append(part[part.method == meth])
    cat = pd.concat(mapped, ignore_index=True)
    cat.to_csv(RESULTS / f"indep_nested_mapped_{tag}.csv", index=False)
    block = {}
    for meth in ["Frozen HiFT", "Nested-selected HiFT", "Oracle (test-tuned)"] + HEADLINE:
        block[meth] = {f"k={k}": _boot(cat, meth, k, min_rep) for k in (0, 1, 2, 3)}
    n_sel = int((ch["selected"] == te.FROZEN).sum())
    out = dict(n_folds=int(len(ch)), n_selected_frozen=n_sel,
               frac_selected_frozen=round(n_sel / max(len(ch), 1), 3),
               selected_counts=ch["selected"].value_counts().to_dict(),
               fold_table=ch.to_dict(orient="records"), metrics=block,
               optimism_k1=None)
    fz, nest, ora = (block[m]["k=1"]["MAE"] for m in
                     ("Frozen HiFT", "Nested-selected HiFT", "Oracle (test-tuned)"))
    if None not in (fz, nest, ora):
        out["optimism_k1"] = dict(frozen_minus_nested=round(fz - nest, 3),
                                  oracle_minus_nested=round(ora - nest, 3),
                                  frozen=fz, nested=nest, oracle=ora)
    print(tag, "nested", json.dumps(out["optimism_k1"]), "frozen-selected", n_sel, "/", len(ch))
    return out


def external(tag, seed=0):
    info = json.loads((RESULTS / "indep_inventory.json").read_text(encoding="utf-8"))
    new_papers = set(info["new_paper_list"])
    d = fl.load_domain(tag)
    aux = fl.load_domain(fl.AUX[tag])
    d_ext = pd.read_csv(DATA / f"{tag}_external_new.csv", low_memory=False)
    d_ext["y"] = d_ext["logP"]
    d_ext["log_solute_MW"] = np.log10(pd.to_numeric(d_ext.get("soluteChoice.solute1.molecularWeight"),
                                                    errors="coerce"))
    extra = fl.extra_for(tag)
    f_oof = np.zeros(len(d))
    for itr, ite in GroupKFold(5, shuffle=True, random_state=100 + seed).split(d, groups=d[REF]):
        f_oof[ite] = fl.gbm_fit_predict(d.iloc[itr], d.iloc[ite], seed, extra)
    f_te = fl.gbm_fit_predict(d, d_ext, seed, extra)
    center, scale = transfer_design(sorted(d[S1].unique()))[1:]
    comp = te.fit_components(d, f_oof, d_ext, aux, seed, tag, center, scale)
    cte = fl.cells_of(d_ext, f_te, "target")
    rows = te.evaluate(cte, comp, seed, 0, tag, grid=True, standard=True, min_solvents=2)
    res = pd.DataFrame(rows)
    res["abs_err"] = (res["y"] - res["pred"]).abs()
    res = res[res.method != te.FROZEN]
    res.to_csv(RESULTS / f"fewshot_external_{tag}.csv", index=False)
    print(f"{tag} external: test membranes={cte['entity'].nunique()} papers={cte['paper'].nunique()} "
          f"rows={len(res)}", flush=True)
    out = {}
    for min_s, lab in [(2, "multi_ge2"), (MIN_REPORT[tag], "primary_threshold")]:
        block = {meth: {f"k={k}": _boot(res, meth, k, min_s) for k in (0, 1, 2, 3)} for meth in HEADLINE}
        zs, h1 = block["Zero-shot GBM"]["k=0"]["MAE"], block["HiFT"]["k=1"]["MAE"]
        out[lab] = dict(metrics=block, gain_k1=_gain(zs, h1))
    return out


def compile_summary(inv, unt, nest, ext):
    summary = dict(frozen=te.FROZEN, inventory=inv, untouched=unt, nested=nest, external=ext)
    (RESULTS / "indep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=== independent validation ===")
    print("frozen", te.FROZEN)
    for tag in nest:
        print(tag, "nested", nest[tag].get("optimism_k1"))
    for tag, g in ext.items():
        print(tag, "external ge2 gain", g.get("multi_ge2", {}).get("gain_k1"),
              "primary gain", g.get("primary_threshold", {}).get("gain_k1"))
    return summary


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    inv = json.loads((RESULTS / "indep_inventory.json").read_text(encoding="utf-8")) if (
        (RESULTS / "indep_inventory.json").exists() and which != "all") else inventory()
    unt = json.loads((RESULTS / "indep_untouched.json").read_text(encoding="utf-8")) if (
        (RESULTS / "indep_untouched.json").exists() and which not in ("all", "untouched")) else untouched()
    nest, ext = {}, {}
    if which in ("all", "nested", "nested_sp"):
        nest["pure_solvent_permeance"] = nested("pure_solvent_permeance")
    elif (RESULTS / "indep_nested_choices_pure_solvent_permeance.csv").exists():
        outer = pd.read_csv(RESULTS / "indep_outer_grid_pure_solvent_permeance.csv", low_memory=False)
        ch = pd.read_csv(RESULTS / "indep_nested_choices_pure_solvent_permeance.csv")
        nest["pure_solvent_permeance"] = summarize_nested("pure_solvent_permeance", outer, ch)
    if which in ("all", "nested", "nested_pm"):
        nest["solution_permeance"] = nested("solution_permeance")
    elif (RESULTS / "indep_nested_choices_solution_permeance.csv").exists():
        outer = pd.read_csv(RESULTS / "indep_outer_grid_solution_permeance.csv", low_memory=False)
        ch = pd.read_csv(RESULTS / "indep_nested_choices_solution_permeance.csv")
        nest["solution_permeance"] = summarize_nested("solution_permeance", outer, ch)
    if which in ("all", "external", "external_sp"):
        ext["pure_solvent_permeance"] = external("pure_solvent_permeance")
    elif (RESULTS / "fewshot_external_pure_solvent_permeance.csv").exists():
        res = pd.read_csv(RESULTS / "fewshot_external_pure_solvent_permeance.csv", low_memory=False)
        ext["pure_solvent_permeance"] = {
            lab: dict(metrics={m: {f"k={k}": _boot(res, m, k, ms) for k in (0, 1, 2, 3)} for m in HEADLINE},
                      gain_k1=_gain(_boot(res, "Zero-shot GBM", 0, ms)["MAE"], _boot(res, "HiFT", 1, ms)["MAE"]))
            for ms, lab in [(2, "multi_ge2"), (MIN_REPORT["pure_solvent_permeance"], "primary_threshold")]}
    if which in ("all", "external", "external_pm"):
        ext["solution_permeance"] = external("solution_permeance")
    elif (RESULTS / "fewshot_external_solution_permeance.csv").exists():
        res = pd.read_csv(RESULTS / "fewshot_external_solution_permeance.csv", low_memory=False)
        ext["solution_permeance"] = {
            lab: dict(metrics={m: {f"k={k}": _boot(res, m, k, ms) for k in (0, 1, 2, 3)} for m in HEADLINE},
                      gain_k1=_gain(_boot(res, "Zero-shot GBM", 0, ms)["MAE"], _boot(res, "HiFT", 1, ms)["MAE"]))
            for ms, lab in [(2, "multi_ge2"), (MIN_REPORT["solution_permeance"], "primary_threshold")]}
    compile_summary(inv, unt, nest, ext)


if __name__ == "__main__":
    main()
