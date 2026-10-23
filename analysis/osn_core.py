"""Core data layer for the OSN study: loading, analysis domains, entities, descriptors, features."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from reference_identity import normalize_references

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "data" / "raw" / "OMD_SRNF_2025-04-08.csv"
DATA = HERE / "data"
RESULTS = HERE / "results"
FIGS = HERE / "figures"
for p in (DATA, RESULTS, FIGS):
    p.mkdir(exist_ok=True)

SP = "filtrationResults.solventPermeance"
PM = "filtrationResults.permeance"
R1 = "filtrationResults.rejection1"
S1 = "testConditions.solvent1"
REF = "report.reference"

SOLVENT_ALIASES = {"Methylene chloride": "Dichloromethane (DCM)", "2-butanone": "Methyl ethyl ketone (MEK)"}

SHORT = {
    "Methanol": "MeOH", "Ethanol": "EtOH", "Propanol": "1-PrOH", "Isopropanol (IPA)": "IPA", "Butanol": "1-BuOH",
    "2-Butanol": "2-BuOH", "Isobutanol": "i-BuOH", "1-Pentanol": "1-PeOH", "1-Hexanol": "1-HxOH",
    "Acetone": "Acetone", "Methyl ethyl ketone (MEK)": "MEK", "Cyclohexanone": "CyH-one", "Ethyl acetate": "EtOAc",
    "Methyl acetate": "MeOAc", "Isopropyl acetate": "i-PrOAc", "Butyl acetate": "BuOAc", "Tetrahydrofuran (THF)": "THF",
    "1,4-dioxane": "Dioxane", "Acetonitrile": "MeCN", "Dimethylformamide (DMF)": "DMF",
    "Dimethylacetamide (DMAC)": "DMAc", "1-Methyl-2-pyrrolidinone (NMP)": "NMP", "Dimethylsulfoxide (DMSO)": "DMSO",
    "Toluene": "Toluene", "Xylene": "Xylene", "Anisole": "Anisole", "Hexane": "Hexane", "Heptane": "Heptane",
    "Pentane": "Pentane", "Octane": "Octane", "Decane": "Decane", "Cyclohexane": "CyHex",
    "Dichloromethane (DCM)": "DCM", "Chloroform": "CHCl3", "Carbon Tetrachloride": "CCl4",
    "Diethyl ether": "Et2O", "dimethyl carbonate": "DMC", "Propylene glycol methyl ether acetate": "PGMEA",
    "water": "Water",
}

# Inputs are frozen from the field-level source audit. See data/reference and reports/verification.
RAW_DESC = ["MW", "rho", "eta", "dD", "dP", "dH", "eps", "gamma"]


def solvent_table():
    t = pd.read_csv(ROOT / "data/reference/solvent_descriptors_verified.csv", index_col="solvent")
    if t[["MW", "rho", "eta", "dD", "dP", "dH"]].isna().any().any():
        raise ValueError("Source-backed transfer descriptors are incomplete")
    t["Vm"] = t["MW"] / t["rho"]
    t["log_eta"] = np.log10(t["eta"])
    t["log_Vm"] = np.log10(t["Vm"])
    t["dT"] = np.sqrt(t["dD"] ** 2 + t["dP"] ** 2 + t["dH"] ** 2)
    t["log_Vm_eta"] = np.log10(t["Vm"] / t["eta"])
    t["d_mol_nm"] = (6 * t["Vm"] * 1e-6 / (np.pi * 6.02214076e23)) ** (1 / 3) * 1e9
    t["short"] = [SHORT.get(s, s) for s in t.index]
    return t


SOLV = solvent_table()
GBM_SOLVENT_FEATURES = ["log_eta", "log_Vm", "dD", "dP", "dH", "dT", "eps", "gamma", "log_Vm_eta", "d_mol_nm"]
TRANSFER_DESC = ["log_eta", "dP", "dH", "log_Vm"]

MATERIAL_ORDER = ["Ceramic", "PDMS", "Glassy ISA", "TFC/TFN", "2D/framework", "Commercial polymer", "Other"]


def _tokens(v):
    if pd.isna(v):
        return []
    s = str(v).strip()
    if s.startswith("["):
        try:
            return [str(t) for t in json.loads(s) if t is not None and str(t).strip()]
        except json.JSONDecodeError:
            return [s]
    return [s] if s else []


CERAMIC_COMMERCIAL = ("inopor", "hitk", "ceram", "tio2", "zro2")


def material_class(row):
    chem = " ".join(_tokens(row["chemistry"])).lower()
    struct = " ".join(_tokens(row["structure"])).lower()
    name = f"{row.get('commercialName', '')} {row.get('name', '')}".lower()
    if row["family"] == "Commercial":
        return "Ceramic" if any(k in name for k in CERAMIC_COMMERCIAL) else "Commercial polymer"
    if "inorganic" in struct or "ceramic" in chem:
        return "Ceramic"
    if any(k in chem for k in ("graphene", "mxene", "metal organic", "mof", "covalent organic", "carbon nanotube")):
        return "2D/framework"
    if "polydimethylsiloxane" in chem:
        return "PDMS"
    if "tfc" in struct or "tfn" in struct:
        return "TFC/TFN"
    if "isa" in struct or "dense" in struct:
        return "Glassy ISA"
    return "Other"


def doi_year(doi):
    """Year implied by publisher DOI conventions, or None."""
    import re
    d = str(doi).lower()
    rules = [
        (r"10\.1016/j\.[a-z]+\.(\d{4})\.", lambda m: int(m.group(1))),
        (r"10\.1016/s\d{4}-\d{3}[\dx]\((\d{2})\)", lambda m: 1900 + int(m.group(1)) if int(m.group(1)) > 50
         else 2000 + int(m.group(1))),
        (r"10\.1021/acs\.?[a-z]+\.(\d)[a-z]\d", lambda m: 2020 + int(m.group(1)) if int(m.group(1)) <= 5
         else 2010 + int(m.group(1))),
        (r"10\.1039/([b-d])(\d)", lambda m: {"b": 2000, "c": 2010, "d": 2020}[m.group(1)] + int(m.group(2))),
        (r"10\.1002/[a-z]+\.(\d{4})\d{3}", lambda m: int(m.group(1))),
        (r"10\.1038/s\d+-(\d{3})-", lambda m: 2000 + int(m.group(1))),
        (r"10\.1007/s\d{5}-(\d{3})-", lambda m: 2000 + int(m.group(1))),
        (r"10\.3390/membranes(\d{1,2})\d{6}", lambda m: 2010 + int(m.group(1))),
        (r"10\.3390/separations(\d{1,2})\d{6}", lambda m: 2013 + int(m.group(1))),
    ]
    for pat, f in rules:
        m = re.search(pat, d)
        if m:
            y = f(m)
            return y if 1990 <= y <= 2026 else None
    return None


def old_acs_year(doi):
    """Approximate year of pre-2013 ACS manuscript IDs (e.g. jp805117z); only used when no other year exists."""
    import re
    m = re.search(r"10\.1021/[a-z]{2}(\d)\d{5}[a-z]$", str(doi).lower())
    if not m:
        return None
    d = int(m.group(1))
    return 2000 + d if d >= 3 else 2010 + d


def reconstruct_years(df):
    """Publication year per source paper. The snapshot stores some years as epoch milliseconds
    (1970-01-01T00:00:02.022Z = 2022) and some curation timestamps (dated late 2023) instead of publication
    dates; genuine years are stored as Jan-1 dates. DOI-implied years override recorded years that disagree by
    more than one year; curation timestamps without a DOI-implied year are treated as unknown."""
    first = df.groupby(REF)["report.publicationDate"].first().astype(str)
    out = {}
    for ref, s in first.items():
        rec = pd.to_datetime(s, errors="coerce")
        rec_year = None if pd.isna(rec) else rec.year
        source = "recorded"
        if rec_year == 1970:
            ms = rec.microsecond // 1000 + 1000 * rec.second
            rec_year = ms if 1990 <= ms <= 2026 else None
            source = "decoded epoch-ms" if rec_year else "missing"
        curation_stamp = rec_year is not None and source == "recorded" and s[5:10] != "01-01"
        dy = doi_year(ref)
        if curation_stamp:
            dy = dy if dy is not None else old_acs_year(ref)
            out[ref] = (dy, "DOI (curation timestamp)" if dy else "missing (curation timestamp)", rec_year, dy)
        elif dy is not None and (rec_year is None or not 0 <= rec_year - dy <= 1):
            out[ref] = (dy, "DOI" if rec_year is None else "DOI (recorded year inconsistent)", rec_year, dy)
        else:
            out[ref] = (rec_year, source if rec_year is not None else "missing", rec_year, dy)
    return pd.DataFrame.from_dict(out, orient="index", columns=["year", "year_source", "recorded_year", "doi_year"])


def load():
    df = normalize_references(pd.read_csv(RAW, low_memory=False))
    num = [SP, PM, R1, "soluteChoice.solute1.molecularWeight", "testConditions.hydraulicP",
           "testConditions.temperature"] + NUMERIC
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[S1] = df[S1].replace(SOLVENT_ALIASES)
    years = reconstruct_years(df)
    df["year"] = df[REF].map(years["year"]).astype(float)
    df["year_source"] = df[REF].map(years["year_source"])
    df["entity"] = df[REF].astype(str) + " || " + df["name"].astype(str)
    df["material"] = df.apply(material_class, axis=1)
    return df


def organic_domain(df, target):
    """Accepted, single organic solvent with known descriptors, no mixture, positive numeric target."""
    m = ((df["status"] == "Accepted") & (~df["testConditions.solventMixture"].astype(bool))
         & df["testConditions.solvent2"].isna() & (df[S1] != "water") & df[S1].isin(SOLV.index)
         & df[target].notna() & (df[target] > 0))
    out = df[m].copy()
    out["logP"] = np.log10(out[target])
    return out


MULTI_LABEL = ["structure", "chemistry", "supportLayerChemistry", "postTreatment",
               "isaSupportLayer.solvent", "interfacialPolymerization.phase1", "interfacialPolymerization.phase2"]
SINGLE_CAT = ["family", "topLayerDepositionMethod", "supportLayerType", "testConditions.filtrationMode",
              "supportLayer.postTreatment", "chemistryOther", "tfnData.nanomaterialName",
              "interfacialPolymerization.monomer1Smiles", "interfacialPolymerization.monomer2Smiles",
              "commercialName"]
NUMERIC = ["testConditions.hydraulicP", "testConditions.temperature",
           "isaSupportLayer.polymerWeightConcentrationInDopeSolution", "isaSupportLayer.castingThickness",
           "interfacialPolymerization.monomer1Concentration", "interfacialPolymerization.monomer2Concentration",
           "tfnData.nanomaterialConcentration"]


def _clean(c):
    for a, b in (("[", "("), ("]", ")"), (",", ";"), (":", "-"), ('"', ""), ("{", "("), ("}", ")")):
        c = c.replace(a, b)
    return c[:110]


def featurize(d, vocab=None, min_count=10, extra_numeric=()):
    """Design-stage membrane descriptors + test conditions + solvent physics. Vocabulary is fit on `d` when None."""
    fit = vocab is None
    vocab = {} if fit else vocab
    cols = {}
    for c in MULTI_LABEL + SINGLE_CAT:
        toks = d[c].map(_tokens)
        if fit:
            counts = pd.Series([t for ts in toks for t in ts], dtype=object).value_counts()
            vocab[c] = list(counts[counts >= min_count].index)
        for t in vocab[c]:
            cols[_clean(f"{c}={t}")] = toks.map(lambda ts, t=t: int(t in ts)).values
        cols[_clean(f"{c}__missing")] = toks.map(lambda ts: int(len(ts) == 0)).values
    X = pd.DataFrame(cols, index=d.index)
    X = X.loc[:, ~X.columns.duplicated()]
    for c in list(NUMERIC) + list(extra_numeric):
        X[_clean(c)] = d[c].values
    X["has_tfn"] = d["tfnData.nanomaterialName"].notna().astype(int).values
    sv = SOLV.loc[d[S1].values, GBM_SOLVENT_FEATURES].set_axis(d.index)
    X = pd.concat([X, sv.add_prefix("solv_")], axis=1)
    return X, vocab


def cell_table(d, value="logP"):
    """Median per (membrane entity, solvent) cell, carrying paper and material."""
    return (d.groupby(["entity", S1], as_index=False)
             .agg(y=(value, "median"), paper=(REF, "first"), material=("material", "first"),
                  n_rows=(value, "size")))


def transfer_design(solvents, center=None, scale=None):
    """Design matrix [1, standardized TRANSFER_DESC] for the hierarchical transfer model."""
    z = SOLV.loc[list(solvents), TRANSFER_DESC].to_numpy(float)
    if center is None:
        center, scale = z.mean(0), z.std(0)
    return np.column_stack([np.ones(len(z)), (z - center) / scale]), center, scale
