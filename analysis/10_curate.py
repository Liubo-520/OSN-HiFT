"""Curate analysis tables and record an auditable count at every exclusion step."""
import json

import numpy as np
import pandas as pd

from osn_core import DATA, PM, R1, RAW, REF, S1, SOLV, SP, load, organic_domain, reconstruct_years
from reference_identity import normalize_references

df = load()
audit = {"raw_rows": len(df), "raw_papers": int(df[REF].nunique()),
         "raw_reference_strings": int(df["report.reference_raw"].nunique()),
         "reference_identity": "canonical DOI/text; raw strings retained; no measurement rows removed"}
df[["report.reference_raw", REF]].drop_duplicates().to_csv(DATA / "paper_reference_map.csv", index=False)
years = reconstruct_years(normalize_references(pd.read_csv(RAW, low_memory=False)))
years.index.name = "reference"
years.to_csv(DATA / "year_audit.csv")
audit["year_sources_papers"] = years["year_source"].value_counts().to_dict()
audit["year_changed_papers"] = int((years["year"].fillna(-1) != years["recorded_year"].fillna(-1)).sum())


def step(name, frame, base):
    audit[name] = {"rows": int(len(frame)), "papers": int(frame[REF].nunique()),
                   "removed_from_previous": int(len(base) - len(frame))}
    return frame


for target, tag in [(SP, "pure_solvent_permeance"), (PM, "solution_permeance")]:
    s0 = step(f"{tag}:numeric", df[df[target].notna()], df)
    s1 = step(f"{tag}:accepted", s0[s0["status"] == "Accepted"], s0)
    s2 = step(f"{tag}:single_solvent", s1[~s1["testConditions.solventMixture"].astype(bool) & s1["testConditions.solvent2"].isna()], s1)
    s3 = step(f"{tag}:organic", s2[s2[S1] != "water"], s2)
    s4 = step(f"{tag}:descriptor_known", s3[s3[S1].isin(SOLV.index)], s3)
    s5 = step(f"{tag}:positive", s4[s4[target] > 0], s4)
    dom = organic_domain(df, target)
    assert len(dom) == len(s5)
    dom.to_csv(DATA / f"{tag}_domain.csv", index=False)
    ns = dom.groupby("entity")[S1].nunique()
    audit[f"{tag}:membranes"] = int(dom["entity"].nunique())
    audit[f"{tag}:membranes_by_n_solvents"] = {int(k): int(v) for k, v in ns.value_counts().sort_index().items()}
    audit[f"{tag}:material_counts"] = dom["material"].value_counts().to_dict()
    audit[f"{tag}:solvents"] = int(dom[S1].nunique())
    audit[f"{tag}:year_missing"] = int(dom["year"].isna().sum())

base = df[(df["status"] == "Accepted") & (~df["testConditions.solventMixture"].astype(bool))
          & df["testConditions.solvent2"].isna() & (df[S1] != "water") & df[S1].isin(SOLV.index)]
parts = []
for i in range(1, 13):
    rc, sc = f"filtrationResults.rejection{i}", f"soluteChoice.solute{i}"
    if rc not in base or f"{sc}.molecularWeight" not in base:
        continue
    p = base.copy()
    p["R"] = pd.to_numeric(p[rc], errors="coerce")
    p["solute_name"] = p[f"{sc}.name"]
    p["solute_cat"] = p.get(f"{sc}.category")
    p["solute_MW"] = pd.to_numeric(p[f"{sc}.molecularWeight"], errors="coerce")
    p["solute_slot"] = i
    parts.append(p[p["R"].notna()])
rej = pd.concat(parts, ignore_index=True)
audit["rejection:long_values"] = int(len(rej))
audit["rejection:from_slot1"] = int((rej.solute_slot == 1).sum())
audit["rejection:negative"] = int((rej.R < 0).sum())
audit["rejection:above_100"] = int((rej.R > 100).sum())
rej_m = rej[(rej.R >= 0) & (rej.R <= 100) & rej.solute_MW.notna()].copy()
audit["rejection:modelled_rows"] = int(len(rej_m))
audit["rejection:papers"] = int(rej_m[REF].nunique())
audit["rejection:solutes"] = int(rej_m["solute_name"].nunique())
rej_m.to_csv(DATA / "rejection_long.csv", index=False)

tab = SOLV.copy()
tab["source"] = "Field-level provenance: reports/verification/solvent_descriptor_audit.csv; ambient reference temperatures vary"
tab.to_csv(DATA / "solvent_descriptors.csv", index_label="solvent")

with open(DATA / "curation_audit.json", "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2, ensure_ascii=False)
print(json.dumps(audit, indent=2, ensure_ascii=False))
