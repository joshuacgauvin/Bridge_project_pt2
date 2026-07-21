"""End-to-end reproducibility check for the national bridge-survival pipeline.

Runs the notebook chain headlessly and verifies it reproduces the tracked artifacts.

Default (smoke) mode — ~15-30 min:
    python reproduce_check.py
  Each notebook is executed with its smoke switch patched IN MEMORY ONLY
  (STATE_SUBSET = ["MA", "RI"] for the dataset build — MA must be present because the
  ablation notebook's Arm C/D are anchored to it — and SMOKE_TEST = True elsewhere;
  the .ipynb files on disk are never modified). The regenerated *_smoke.json
  artifacts are then compared field-by-field against the versions committed at git
  HEAD. Smoke artifacts are deleted before their producer runs, so stale keys from
  older schema versions cannot linger through the stage-save merge pattern.

Full mode — the real pipeline, ~13-16 h:
    python reproduce_check.py --full
  Runs the chain unpatched and compares the tracked full-scale JSONs and the four
  tracked CSVs (hazard ratios, AFT coefficients, per-state C-index, feature
  importance).

Other flags:
    --run-only       execute the chain but skip the comparison (used to regenerate
                     reference artifacts before committing them)
    --compare-only   skip execution; compare what is on disk against git HEAD
    --only n4,n6     run/compare a comma-separated subset of steps (see CHAIN below)
    --tol 1e-3       absolute tolerance for float comparisons (default 0 = exact;
                     the pipeline is deterministic on a single machine, and every
                     stored value is rounded before writing)

Exit code 0 = every compared artifact matches; 1 = any mismatch, missing reference,
or execution error. Volatile fields (generated_utc timestamps, fit-time entries) are
ignored. Requires the gitignored raw inputs (see README "Data") to be present.
"""
import argparse
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parent
DATADIR = REPO / "Bridges_all_of_US"

# (step, notebook, flag patch, smoke artifacts, full artifacts, in smoke chain?)
SMOKE_PATCH = (r"^SMOKE_TEST\s*=\s*False", "SMOKE_TEST = True")
N4_PATCH = (r"^STATE_SUBSET\s*=\s*None", 'STATE_SUBSET = ["MA", "RI"]')
CHAIN = [
    ("n4", "build_national_rsf_dataset.ipynb", N4_PATCH,
     ["us_culvert_diagnostic_smoke.json", "us_reconstruction_measurement_smoke.json",
      "us_traffic_leakage_measurement_smoke.json"],
     ["us_culvert_diagnostic.json", "us_reconstruction_measurement.json",
      "us_traffic_leakage_measurement.json"], True),
    # us_bridge_risk_map[_smoke].json is a multi-MB per-bridge visualization payload
    # (the full one is gitignored) — not compared here; N5's scientific artifact is the metrics JSON.
    ("n5", "train_national_rsf.ipynb", SMOKE_PATCH,
     ["us_rsf_metrics_smoke.json"],
     ["us_rsf_metrics.json", "us_rsf_cindex_by_state.csv", "us_rsf_feature_importance.csv"],
     True),
    ("n6", "train_national_parametric.ipynb", SMOKE_PATCH,
     ["us_parametric_metrics_smoke.json", "us_cox_ph_check_smoke.json"],
     ["us_parametric_metrics.json", "us_cox_ph_check.json",
      "us_cox_hazard_ratios.csv", "us_aft_coefficients.csv"], True),
    # The map notebook renders Notebook 5's outputs and has no smoke mode / no tracked
    # artifact of its own (us_bridge_risk_map.json is gitignored for size).
    ("map", "bridge_map_national.ipynb", None, [], [], False),
    ("n8", "ablation_state_dummies.ipynb", SMOKE_PATCH,
     ["us_ablation_state_dummies_smoke.json"], ["us_ablation_state_dummies.json"], True),
    ("leakage", "leakage_ablation_national.ipynb", SMOKE_PATCH,
     ["us_leakage_ablation_smoke.json"], ["us_leakage_ablation.json"], True),
    ("temporal", "temporal_validation_national.ipynb", SMOKE_PATCH,
     ["us_temporal_validation_smoke.json"], ["us_temporal_validation.json"], True),
    ("bootstrap", "bootstrap_model_comparison.ipynb", SMOKE_PATCH,
     ["us_model_bootstrap_smoke.json"], ["us_model_bootstrap.json"], True),
]

VOLATILE = re.compile(r"(^generated_utc$|_utc$|minutes$|seconds$)")


def git_show(relpath):
    """Bytes of the committed version at HEAD, or None if not tracked there."""
    r = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{relpath}"],
                       capture_output=True)
    return r.stdout if r.returncode == 0 else None


def diff_json(ref, new, tol, path="$"):
    """Recursive field diff, skipping volatile keys. Returns list of mismatch strings."""
    out = []
    if isinstance(ref, dict) and isinstance(new, dict):
        for k in sorted(set(ref) | set(new)):
            if VOLATILE.search(str(k)):
                continue
            if k not in ref:
                out.append(f"{path}.{k}: only in regenerated")
            elif k not in new:
                out.append(f"{path}.{k}: only in reference")
            else:
                out += diff_json(ref[k], new[k], tol, f"{path}.{k}")
    elif isinstance(ref, list) and isinstance(new, list):
        if len(ref) != len(new):
            out.append(f"{path}: length {len(ref)} != {len(new)}")
        else:
            for i, (r, n) in enumerate(zip(ref, new)):
                out += diff_json(r, n, tol, f"{path}[{i}]")
    elif isinstance(ref, (int, float)) and isinstance(new, (int, float)) \
            and not isinstance(ref, bool) and not isinstance(new, bool):
        if abs(float(ref) - float(new)) > tol:
            out.append(f"{path}: {ref} != {new}")
    elif ref != new:
        out.append(f"{path}: {ref!r} != {new!r}")
    return out


def diff_csv(ref_bytes, disk_path, tol):
    """Compare a tracked CSV against its regenerated version (parsed, not byte-wise)."""
    import numpy as np
    import pandas as pd
    ref = pd.read_csv(io.BytesIO(ref_bytes))
    new = pd.read_csv(disk_path)
    if list(ref.columns) != list(new.columns):
        return [f"columns differ: {sorted(set(ref.columns) ^ set(new.columns))}"]
    if len(ref) != len(new):
        return [f"row count {len(ref)} != {len(new)}"]
    out = []
    for c in ref.columns:
        r, n = ref[c], new[c]
        if pd.api.types.is_numeric_dtype(r) and pd.api.types.is_numeric_dtype(n):
            bad = ~(np.isclose(r, n, rtol=0, atol=tol, equal_nan=True))
        else:
            bad = ~((r == n) | (r.isna() & n.isna()))
        if bad.any():
            i = int(bad.idxmax())
            out.append(f"col {c}: {int(bad.sum())} mismatched rows "
                       f"(first at row {i}: {r.iloc[i]!r} != {n.iloc[i]!r})")
    return out


def compare_artifact(name, tol):
    disk = DATADIR / name
    rel = f"Bridges_all_of_US/{name}"
    ref = git_show(rel)
    if ref is None:
        return False, [f"no tracked reference at HEAD — run with --run-only and commit {rel}"]
    if not disk.exists():
        return False, ["not regenerated (missing on disk)"]
    if name.endswith(".json"):
        problems = diff_json(json.loads(ref.decode("utf-8")),
                             json.loads(disk.read_text(encoding="utf-8")), tol)
    else:
        problems = diff_csv(ref, disk, tol)
    return not problems, problems


def run_notebook(nb_file, patch, cell_timeout):
    nb = nbformat.read(DATADIR / nb_file, as_version=4)
    if patch:
        pattern, repl = patch
        hits = 0
        for c in nb.cells:
            if c.cell_type != "code":
                continue
            src, n = re.subn(pattern, repl, c.source, flags=re.M)
            hits += n
            c.source = src
        assert hits == 1, (f"{nb_file}: expected exactly one '{pattern}' match, got {hits} "
                           "— is the flag already set, or was the config cell changed?")
    kernel = nb.metadata.get("kernelspec", {}).get("name", "python3")
    client = NotebookClient(nb, timeout=cell_timeout, kernel_name=kernel,
                            resources={"metadata": {"path": str(DATADIR)}})
    client.execute()   # in-memory notebook object only; the .ipynb on disk is untouched


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="run the real pipeline (~13-16 h)")
    ap.add_argument("--run-only", action="store_true", help="execute chain, skip comparison")
    ap.add_argument("--compare-only", action="store_true", help="compare disk vs HEAD, no execution")
    ap.add_argument("--only", default="", help="comma-separated step subset, e.g. n4,n6")
    ap.add_argument("--tol", type=float, default=0.0, help="float tolerance (default exact)")
    args = ap.parse_args()
    assert not (args.run_only and args.compare_only), "--run-only and --compare-only conflict"

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    steps = [s for s in CHAIN if (not only or s[0] in only)]
    if not args.full:
        steps = [s for s in steps if s[5]]           # smoke chain skips the map notebook
    if args.full:
        print("FULL mode: this reruns the real pipeline (~13-16 h of model fits).")

    failures = 0
    t_start = time.time()
    for step, nb_file, patch, smoke_arts, full_arts, _ in steps:
        arts = full_arts if args.full else smoke_arts
        if not args.compare_only:
            if not args.full:
                for a in arts:                        # clean regeneration (refs live in git)
                    (DATADIR / a).unlink(missing_ok=True)
            print(f"[{step}] executing {nb_file} ...", flush=True)
            t0 = time.time()
            try:
                run_notebook(nb_file, None if args.full else patch,
                             cell_timeout=21600 if args.full else 3600)
            except Exception as e:
                print(f"[{step}] EXECUTION FAILED: {type(e).__name__}: {e}")
                sys.exit(1)
            print(f"[{step}] done in {(time.time() - t0) / 60:.1f} min")
        if args.run_only:
            continue
        for a in arts:
            ok, problems = compare_artifact(a, args.tol)
            print(f"[{step}] {a}: {'PASS' if ok else 'FAIL'}")
            for p in problems[:20]:
                print(f"         {p}")
            if len(problems) > 20:
                print(f"         ... and {len(problems) - 20} more")
            failures += (not ok)

    mode = "run-only" if args.run_only else ("full" if args.full else "smoke")
    print(f"\n{mode} chain finished in {(time.time() - t_start) / 60:.1f} min", end="")
    if args.run_only:
        print(" — no comparison requested (commit the regenerated artifacts as references)")
        return
    print(f"; {failures} artifact(s) failed" if failures else "; all artifacts reproduced")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
