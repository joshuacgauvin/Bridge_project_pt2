"""Diff every tracked pipeline artifact against a git tag, as old -> new pairs.

Written for the record-split (structure-renumbering) fix: the stitched rebuild changes
essentially every headline number, and each one has to be re-quoted in README.md,
draft.ipynb and the manuscript. This prints the substitution list to work from, so the
edit is mechanical rather than a hand audit.

    python compare_to_tag.py                      # vs pre-stitch-2026-07-29
    python compare_to_tag.py --tag v1 --min-delta 0.001
    python compare_to_tag.py --smoke              # compare the *_smoke artifacts instead
"""
import argparse
import io
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

FULL = ["us_culvert_diagnostic.json", "us_reconstruction_measurement.json",
        "us_traffic_leakage_measurement.json", "us_rsf_metrics.json",
        "us_parametric_metrics.json", "us_cox_ph_check.json",
        "us_ablation_state_dummies.json", "us_leakage_ablation.json",
        "us_temporal_validation.json", "us_model_bootstrap.json"]
CSVS = ["us_rsf_cindex_by_state.csv", "us_rsf_feature_importance.csv",
        "us_cox_hazard_ratios.csv", "us_aft_coefficients.csv"]

# generated_utc / fit_seconds / fit_minutes carry no scientific content.
VOLATILE = re.compile(r"(^generated_utc$|_utc$|minutes$|seconds$)")


def at_tag(tag, relpath):
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{tag}:{relpath}"],
                       capture_output=True)
    return r.stdout if r.returncode == 0 else None


def walk(old, new, min_delta, path="", out=None):
    """Collect (jsonpath, old, new) for every leaf that changed."""
    out = [] if out is None else out
    if isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(set(old) | set(new)):
            if VOLATILE.search(str(k)):
                continue
            if k not in old:
                out.append((f"{path}.{k}", "—", new[k]))
            elif k not in new:
                out.append((f"{path}.{k}", old[k], "—"))
            else:
                walk(old[k], new[k], min_delta, f"{path}.{k}", out)
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            out.append((f"{path}[]", f"len {len(old)}", f"len {len(new)}"))
        else:
            for i, (o, n) in enumerate(zip(old, new)):
                walk(o, n, min_delta, f"{path}[{i}]", out)
    elif isinstance(old, (int, float)) and isinstance(new, (int, float)) \
            and not isinstance(old, bool) and not isinstance(new, bool):
        if abs(float(old) - float(new)) > min_delta:
            out.append((path, old, new))
    elif old != new:
        out.append((path, old, new))
    return out


def fmt(v):
    return f"{v:,}" if isinstance(v, int) and abs(v) >= 10000 else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pre-stitch-2026-07-29")
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-per-file", type=int, default=60)
    a = ap.parse_args()

    names = [n.replace(".json", "_smoke.json") for n in FULL] if a.smoke else FULL + CSVS
    missing, total = [], 0

    for name in names:
        rel = f"Bridges_all_of_US/{name}"
        ref, disk = at_tag(a.tag, rel), HERE / name
        if ref is None or not disk.exists():
            missing.append(f"{name} ({'not at tag' if ref is None else 'not on disk'})")
            continue

        if name.endswith(".json"):
            rows = walk(json.loads(ref.decode("utf-8")),
                        json.loads(disk.read_text(encoding="utf-8")), a.min_delta)
        else:
            import pandas as pd
            o, n = pd.read_csv(io.BytesIO(ref)), pd.read_csv(disk)
            rows = []
            if list(o.columns) != list(n.columns) or len(o) != len(n):
                rows.append((f"shape {o.shape}", f"{o.shape}", f"{n.shape}"))
            else:
                key = o.columns[0]
                for c in o.columns[1:]:
                    if not pd.api.types.is_numeric_dtype(o[c]):
                        continue
                    d = (o[c] - n[c]).abs()
                    for i in d[d > a.min_delta].sort_values(ascending=False).index[:5]:
                        rows.append((f"{c} @ {o[key].iloc[i]}", o[c].iloc[i], n[c].iloc[i]))

        total += len(rows)
        print(f"\n{'=' * 78}\n{name}  —  {len(rows)} changed value(s)\n{'=' * 78}")
        for p, o, n in rows[:a.max_per_file]:
            print(f"  {p:<58} {fmt(o):>12}  ->  {fmt(n)}")
        if len(rows) > a.max_per_file:
            print(f"  ... and {len(rows) - a.max_per_file} more")

    if missing:
        print(f"\nSKIPPED: {', '.join(missing)}")
    print(f"\n{total} changed values vs {a.tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
