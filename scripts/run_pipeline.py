"""Run the production pipeline; fresh by default, hash-bound explicit resume."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "analysis"
STEPS = [("10_curate.py",), ("11_benchmark.py",), ("12_variance_ceiling.py",),
         ("13_fewshot.py",), ("14_conformal.py",), ("15_fingerprints_anchors.py",),
         ("16_si_analyses.py", "all"), ("17_independent_validation.py", "all"),
         ("18_reference_duplicates.py",), ("fig4_hift.py",), ("20_key_numbers.py",),
         ("fig1_landscape.py",), ("fig2_generalization.py",), ("fig3_study_effects.py",),
         ("fig5_uncertainty.py",), ("fig6_mechanism.py",), ("fig_si.py",),
         ("fig0_graphical_abstract.py",), ("fig_toc.py",), ("21_si_tables.py",),
         ("23_build_paper.py",)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--through")
    args = ap.parse_args()
    state_path = ROOT / "reports" / "pipeline_run.json"
    bound = sorted([ROOT / "data/raw/OMD_SRNF_2025-04-08.csv",
                    ROOT / "data/raw/external/OMD_SRNF_2026-10-23.csv"] +
                   list((ROOT / "data/reference").glob("*.csv")) +
                   [p for p in A.glob("*.py") if not p.name.startswith(("fig", "20_", "21_", "23_"))])
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in bound}
    authoring_files = list(A.glob("fig*.py")) + [A / name for name in
        ["20_key_numbers.py", "21_si_tables.py", "23_build_paper.py"]] + list((ROOT / "paper_src").rglob("*"))
    authoring = {str(p.relative_to(ROOT)): digest(p) for p in sorted(authoring_files) if p.is_file()}
    authoring_steps = {" ".join(step) for step in STEPS[9:]}
    if args.resume:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["inputs"] != inputs:
            raise RuntimeError("Input/code hashes changed: start fresh instead of using cached folds")
        if state.get("authoring_inputs") != authoring:
            # Numerical model results remain bound; changed presentation inputs
            # invalidate their downstream chain. Manuscript-only edits do not
            # change numerical summaries or figure inputs.
            previous = state.get("authoring_inputs", {})
            changed = {name for name in set(previous) | set(authoring)
                       if previous.get(name) != authoring.get(name)}
            manuscript_only = previous and all(name.replace('\\', '/').startswith('paper_src/') or
                                               Path(name).name == '23_build_paper.py' for name in changed)
            leaf_figures = {"fig1_landscape.py", "fig2_generalization.py", "fig3_study_effects.py",
                            "fig5_uncertainty.py", "fig6_mechanism.py", "fig_si.py", "fig_toc.py"}
            presentation_only = previous and all(name.replace('\\', '/').startswith('paper_src/') or
                Path(name).name in leaf_figures | {'23_build_paper.py'} for name in changed)
            invalidated = ({"23_build_paper.py"} | {Path(name).name for name in changed if Path(name).name in leaf_figures}
                           if presentation_only else authoring_steps)
            state["steps"] = [s for s in state["steps"] if s["step"] not in invalidated]
            state["authoring_regeneration_reason"] = ("Manuscript source changed; rebuild manuscript only" if
                manuscript_only else "Leaf figure/manuscript sources changed; rebuild affected figures and manuscripts" if
                presentation_only else "Authoring inputs changed or were not previously registered")
        state["authoring_inputs"] = authoring
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = ROOT / "archive" / ("outputs_before_" + stamp)
        for sub in [A / "data", A / "results", A / "figures", ROOT / "paper"]:
            if sub.exists() and any(sub.iterdir()):
                target = archive / sub.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(sub), str(target))
            sub.mkdir(parents=True, exist_ok=True)
        state = {"started_local": stamp, "timezone": "Asia/Tokyo", "inputs": inputs,
                 "python": sys.executable, "steps": [], "status": "running", "authoring_inputs": authoring}
    logdir = ROOT / "reports" / "run_logs"
    logdir.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    for step in STEPS:
        key = " ".join(step)
        if any(x["step"] == key and x["returncode"] == 0 for x in state["steps"]):
            continue
        print("START", key, flush=True)
        started = time.time()
        log = logdir / (step[0].replace(".py", ".log"))
        with log.open("w", encoding="utf-8") as f:
            result = subprocess.run([sys.executable, "-u", str(A / step[0]), *step[1:]],
                                    cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        state["steps"].append({"step": key, "returncode": result.returncode,
                               "seconds": round(time.time() - started, 2),
                               "log": str(log.relative_to(ROOT))})
        protocol_order = {" ".join(item): i for i, item in enumerate(STEPS)}
        state["steps"].sort(key=lambda item: protocol_order[item["step"]])
        state["status"] = "failed" if result.returncode else "running"
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print("END", key, result.returncode, state["steps"][-1]["seconds"], flush=True)
        if result.returncode:
            print(log.read_text(encoding="utf-8")[-4000:], flush=True)
            raise SystemExit(result.returncode)
        if args.through == step[0]:
            state["status"] = "staged_stop"
            break
    else:
        state["status"] = "complete"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
