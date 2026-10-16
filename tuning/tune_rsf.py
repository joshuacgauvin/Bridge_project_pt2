"""Random-search hyperparameter tuning for the national Random Survival Forest.

Built to run on a cloud machine: the defaults are the full-size search a laptop could
not finish (see "Search size"), and everything the run needs is in this folder.

    bash run_tuning.sh               # Linux/cloud: venv + pinned installs + detached run
    python tune_rsf.py               # the same search, in the foreground
    python tune_rsf.py --smoke       # <1 min end-to-end check -> *_smoke outputs

Input:  rsf_tuning_train.csv (this folder) — Notebook 5's training split only, built by
        prepare_tuning_data.py. The test split is never read, so tuning cannot touch the
        headline test C-index.
Output: rsf_best_hyperparameters.json (this folder) — the winning parameters, an
        `rsf_params` block ready to paste into Notebook 5's RSF_PARAMS, the production
        baseline scored on the same folds, an estimate of the RAM the production refit
        will need, and every trial.

Search size — 50 configurations (as many as the Massachusetts search in
ma_bridges/RSF.ipynb) x 3 folds x 50-tree forests, on all 641,836 training rows:
    max_samples        rows per tree: 5/10/25/50% of the training split, or None
                       (full bootstrap)                      [laptop search: 2.5-15%]
    min_samples_leaf   3, 5, 10, 20, 35, 50, 75, 100         [laptop: 20-99]
    min_samples_split  6, 10, 20, 50, 100, 200               [laptop: 50-299]
    max_features       log2, sqrt, 0.2, 0.3, 0.4             [laptop: sqrt, 0.2, 0.3]
    max_depth          None, 10, 15, 20 (as in MA)           [laptop: not searched]
The fine end (leaf 3-5, large or full bootstrap) is where the forest scores best — a
16-tree probe at leaf 3 / 160k rows per tree reached C = 0.83 vs 0.77 for the shipped
settings — and the laptop-sized search could not reach it. Folds and trees only set how
precisely each configuration is scored: fold-to-fold spread is ~0.002, far below the
gaps between configurations, so 3 x 50 ranks them reliably. `--cv 5 --n-estimators 100`
gives exact MA parity at ~4x the compute.

Method (replaces the RUN_TUNING cell at the end of train_national_rsf.ipynb):
  - Folds: StratifiedKFold on `event`, SHUFFLED, precomputed once and shared by every
    configuration. (The notebook cell drew its subsample with groupby(state, event)
    .sample, which returns rows sorted by state, then used RandomizedSearchCV's
    unshuffled KFold — so each validation fold was a block of whole states.)
  - max_samples is searched as an ABSOLUTE number of rows per tree (fractions of the
    full 641,836-row split; production's 0.10 = 64,183 rows), so the winner transfers to
    the full refit. None = full bootstrap: the fold's training rows in CV, all 641,836
    at the production refit.
  - Scoring: Harrell's C-index (RandomSurvivalForest.score) — the headline metric.
  - n_estimators is not tuned (more trees only reduce variance); the exported
    rsf_params uses production's 200.
  - The search fits with low_memory=True: same trees and risk scores, but one stored
    value per node instead of ~440. Notebook 5 needs low_memory=False for survival
    curves, so the JSON's `production_refit` block estimates the RAM that refit needs.
  - A configuration's folds are fitted in parallel threads, sharing the cores, so a
    many-core machine stays busy even when a forest has fewer trees than cores.
  - Each finished configuration is appended to rsf_tuning_progress.jsonl; rerunning
    the same command skips completed ones (a larger --n-iter extends the search). The
    file is folded into the JSON and deleted when the search finishes.

Flags:
    --n-iter 50          random configurations to try (plus the production baseline)
    --cv 3               cross-validation folds
    --n-estimators 50    trees per forest during the search
    --n-tune N           stratified (state x event) subsample of the training rows
                         (default: all 641,836)
    --n-jobs -1          total cores to use (default: all the machine/container has)
    --fold-jobs K        folds fitted at once (default: all; lower it if RAM is short)
    --seed 42            seed for the subsample, folds, parameter draws and forests
    --smoke              5,000 rows, 3 configurations, 10 trees; writes *_smoke files
    --fresh              discard any saved progress and start over

Requires the pinned packages in requirements.txt (Python 3.12 on x86-64).
"""
import argparse
import ctypes
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import cpu_count
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

HERE = Path(__file__).resolve().parent
DATA = HERE / "rsf_tuning_train.csv"
KEYS = ["STATE_FIPS", "STRUCTURE_NUMBER_008"]

# Rows per tree as fractions of the FULL training split (converted to absolute counts);
# 0.10 is production's setting, None is a full bootstrap.
MAX_SAMPLES_FRACS = [0.05, 0.10, 0.25, 0.50, None]
SPACE = {
    "min_samples_leaf":  [3, 5, 10, 20, 35, 50, 75, 100],
    "min_samples_split": [6, 10, 20, 50, 100, 200],
    "max_features":      ["log2", "sqrt", 0.2, 0.3, 0.4],
    "max_depth":         [None, 10, 15, 20],
}
# Notebook 5's shipped configuration; max_samples (10%) is filled in once rows are known.
BASELINE = dict(min_samples_split=100, min_samples_leaf=50, max_features="sqrt", max_depth=None)
PRODUCTION_N_ESTIMATORS = 200
NODE_BYTES = 64   # sklearn's per-node struct, on top of the stored survival values


def keep_awake(on):
    """Stop Windows from sleeping mid-search (sleep has killed long runs here before)."""
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def load(n_tune, seed):
    """Typed load as in Notebook 5 cell-03, plus an optional stratified subsample."""
    if not DATA.exists():
        raise FileNotFoundError(f"{DATA} not found — copy it into this folder "
                                "(or rebuild it with prepare_tuning_data.py)")
    cols = pd.read_csv(DATA, nrows=0).columns
    dtypes = {c: "float32" for c in cols}
    dtypes.update({k: str for k in KEYS})
    dtypes["event"] = "int8"
    df = pd.read_csv(DATA, dtype=dtypes)
    n_full = len(df)

    feats = [c for c in df.columns if c not in KEYS + ["event", "time"]]
    leak = [c for c in feats if "_COND_" in c or c == "OPEN_CLOSED_POSTED_041"
            or c in ("bridge_age", "STRUCTURE_TYPE_043B_19")]
    assert not leak, f"leakage / culvert columns present: {leak} — rebuild with prepare_tuning_data.py"
    assert (df["time"] > 0).all(), "non-positive survival times"

    if n_tune and n_tune < n_full:
        pos = (df.groupby(["STATE_FIPS", "event"], group_keys=False)
                 .sample(frac=n_tune / n_full, random_state=seed)).index.to_numpy()
        df = df.loc[np.sort(pos)].reset_index(drop=True)

    y = Surv.from_arrays(event=df["event"].astype(bool).to_numpy(),
                         time=df["time"].astype("float64").to_numpy())
    X = df[feats].to_numpy(dtype=np.float32)
    return X, y, n_full


def clean(params):
    """JSON-safe params with a stable key order."""
    return {k: (int(v) if isinstance(v, np.integer) else v) for k, v in sorted(params.items())}


def key(params):
    return json.dumps(clean(params), sort_keys=True)


def draw_configs(space, n_iter, seed):
    """n_iter distinct configurations from one seeded stream, so a larger --n-iter
    extends a search instead of reshuffling it."""
    rng = np.random.default_rng(seed)
    n_grid = int(np.prod([len(v) for v in space.values()]))
    out, seen = [], set()
    while len(out) < min(n_iter, n_grid):
        p = clean({k: v[rng.integers(len(v))] for k, v in sorted(space.items())})
        if key(p) not in seen:
            seen.add(key(p))
            out.append(p)
    return out


def unique_rows(m, n):
    """Expected distinct rows in a bootstrap of m draws from n rows."""
    return n * (1 - np.exp(-m / n))


def split_features(max_features, p):
    """Features tried per split, as sklearn resolves max_features."""
    if max_features == "sqrt":
        return max(1, int(np.sqrt(p)))
    if max_features == "log2":
        return max(1, int(np.log2(p)))
    return max(1, int(max_features * p))


def read_progress(path, settings):
    """Completed trials from a previous run of the same settings, keyed by params."""
    done = {}
    if not path.exists():
        return done
    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0]) if lines else {}
    if header.get("settings") != settings:
        sys.exit(f"{path.name} was written with different settings:\n  saved: "
                 f"{header.get('settings')}\n  now:   {settings}\n"
                 "Rerun with the original flags to resume, or add --fresh to start over.")
    for line in lines[1:]:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:          # a run killed mid-write leaves a partial line
            print(f"  ignoring a truncated line in {path.name}")
            continue
        done[key(rec["params"])] = rec
    return done


def run_trial(label, params, X, y, folds, args, forest_jobs):
    """Cross-validate one configuration, its folds fitted in parallel threads."""
    def one_fold(fold):
        tr, te = fold
        rsf = RandomSurvivalForest(n_estimators=args.n_estimators, low_memory=True,
                                   n_jobs=forest_jobs, random_state=args.seed, **params)
        t0 = time.time()
        rsf.fit(X[tr], y[tr])
        fit_seconds = time.time() - t0
        c = rsf.score(X[te], y[te])
        nodes = float(np.mean([t.tree_.node_count for t in rsf.estimators_]))
        return c, fit_seconds, nodes

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.fold_jobs) as pool:
        res = list(pool.map(one_fold, folds))
    scores = np.array([r[0] for r in res])
    return {"trial": label, "params": clean(params),
            "mean_c_index": float(scores.mean()), "std_c_index": float(scores.std()),
            "fold_c_index": [float(s) for s in scores],
            "nodes_per_tree": float(np.mean([r[2] for r in res])),
            "fit_minutes": sum(r[1] for r in res) / 60,
            "wall_minutes": (time.time() - t0) / 60}


def describe(p):
    rows = "full" if p["max_samples"] is None else f"{p['max_samples']:,}"
    return (f"rows/tree={rows} leaf={p['min_samples_leaf']} split={p['min_samples_split']} "
            f"max_features={p['max_features']} max_depth={p['max_depth']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-iter", type=int, default=50)
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--n-estimators", type=int, default=50)
    ap.add_argument("--n-tune", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--fold-jobs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_tune, args.n_iter, args.n_estimators = 5_000, 3, 10

    total_jobs = cpu_count() if args.n_jobs == -1 else args.n_jobs
    args.fold_jobs = max(1, min(args.fold_jobs or args.cv, args.cv, total_jobs))
    forest_jobs = max(1, total_jobs // args.fold_jobs)

    suffix = "_smoke" if args.smoke else ""
    out_json = HERE / f"rsf_best_hyperparameters{suffix}.json"
    progress = HERE / f"rsf_tuning_progress{suffix}.jsonl"

    t_load = time.time()
    X, y, n_full = load(args.n_tune, args.seed)
    n_features = X.shape[1]
    n_times = len(np.unique(y["time"]))      # survival values stored per node when low_memory=False
    print(f"loaded {DATA.name} in {time.time() - t_load:.0f}s: tuning on {len(X):,} of "
          f"{n_full:,} training bridges x {n_features} features   "
          f"event rate {y['event'].mean()*100:.1f}%", flush=True)
    print(f"{total_jobs} cores: {args.fold_jobs} folds at once x {forest_jobs} threads per forest",
          flush=True)

    folds = list(StratifiedKFold(args.cv, shuffle=True, random_state=args.seed)
                 .split(X, y["event"]))
    fold_train = min(len(tr) for tr, _ in folds)

    # Rows-per-tree grid. Smoke scales it to its tiny pool; real runs keep the counts
    # anchored to the full training split so they transfer to the production refit.
    base_n = len(X) if args.smoke else n_full
    grid = [None if f is None else int(f * base_n) for f in MAX_SAMPLES_FRACS]
    too_big = [g for g in grid if g is not None and g > fold_train]
    grid = [g for g in grid if g is None or g <= fold_train]
    if too_big:
        print(f"  max_samples {too_big} exceed the {fold_train:,}-row fold training size and are "
              "skipped — raise --n-tune or --cv to search them")
    baseline = clean({**BASELINE, "max_samples": int(0.10 * base_n)})
    assert baseline["max_samples"] <= fold_train, \
        f"production's {baseline['max_samples']:,} rows/tree exceeds the fold size — raise --n-tune"

    space = {"max_samples": grid, **SPACE}
    # n_iter is left out on purpose: rerunning with a larger --n-iter extends a search.
    # Parallelism flags are left out too — they change speed, not results.
    settings = {"n_rows": len(X), "n_full": n_full, "cv": args.cv,
                "n_estimators": args.n_estimators, "seed": args.seed, "smoke": args.smoke,
                "space": space}

    if args.fresh:
        progress.unlink(missing_ok=True)
    done = read_progress(progress, settings)
    if done:
        print(f"resuming: {len(done)} configuration(s) already scored in {progress.name}")
    elif not progress.exists():
        progress.write_text(json.dumps({"settings": settings}) + "\n", encoding="utf-8")

    draws = draw_configs(space, args.n_iter, args.seed)
    todo = [("baseline", baseline)] + [(f"{i + 1}/{len(draws)}", p) for i, p in enumerate(draws)]

    def cost(p):
        """Relative fit cost for the ETA — a power law fitted to timed fits at three corners
        of the space; depth caps cut it in proportion to the unconstrained depth."""
        rows = unique_rows(p["max_samples"] or fold_train, fold_train)
        c = (rows ** 1.2 * split_features(p["max_features"], n_features) ** 0.62
             * p["min_samples_leaf"] ** -0.45)
        natural_depth = 3.1 * np.log2(max(rows / p["min_samples_leaf"], 2))
        return c * min(1.0, p["max_depth"] / natural_depth) if p["max_depth"] else c

    keep_awake(True)
    try:
        session_minutes, session_cost = 0.0, 0.0
        pending = {key(p): p for _, p in todo if key(p) not in done}
        for label, params in todo:
            k = key(params)
            if k in done:
                r = done[k]
                print(f"[{label:>8}] done   C={r['mean_c_index']:.4f} +/- {r['std_c_index']:.4f}   "
                      f"{describe(r['params'])}", flush=True)
                continue
            rec = run_trial(label, params, X, y, folds, args, forest_jobs)
            done[k] = rec
            with progress.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())
            pending.pop(k)
            session_minutes += rec["wall_minutes"]
            session_cost += cost(params)
            eta = session_minutes / session_cost * sum(cost(p) for p in pending.values()) / 60
            print(f"[{label:>8}] C={rec['mean_c_index']:.4f} +/- {rec['std_c_index']:.4f}   "
                  f"{rec['wall_minutes']:6.1f} min   {describe(params)}   "
                  f"ETA {eta:.1f} h", flush=True)
    finally:
        keep_awake(False)

    # ── Assemble the export ──────────────────────────────────────────────────
    trials = sorted(done.values(), key=lambda r: r["mean_c_index"], reverse=True)
    best = trials[0]
    base = done[key(baseline)]
    fold_delta = [b - a for b, a in zip(best["fold_c_index"], base["fold_c_index"])]
    r4 = lambda v: round(float(v), 4)
    cidx = lambda r: {"mean": r4(r["mean_c_index"]), "std": r4(r["std_c_index"]),
                      "folds": [r4(s) for s in r["fold_c_index"]]}

    def refit_ram_gb(r):
        """Notebook 5's low_memory=False forest: nodes/tree (scaled from fold to full
        training size) x 200 trees x (2 survival values per unique time + node struct)."""
        m = r["params"]["max_samples"]
        scale = (unique_rows(m or base_n, base_n) / unique_rows(m or fold_train, fold_train))
        return r["nodes_per_tree"] * scale * PRODUCTION_N_ESTIMATORS * (16 * n_times + NODE_BYTES) / 1e9

    best_ms = best["params"]["max_samples"]
    result = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "smoke_test": args.smoke,
        "data": {
            "file": DATA.name,
            "source": "Bridges_all_of_US/us_rsf_data_clean.csv, Notebook 5's 80/20 "
                      "train_test_split(random_state=42) training rows only",
            "n_training_rows": n_full,
            "n_tune": len(X),
            "n_features": n_features,
            "event_rate": r4(y["event"].mean()),
        },
        "search": {
            "method": "random search over a discrete grid + production baseline",
            "n_iter": args.n_iter,
            "n_configs_scored": len(trials),
            "cv": f"StratifiedKFold(n_splits={args.cv}, shuffle=True, random_state={args.seed}) on event",
            "n_estimators_during_search": args.n_estimators,
            "scoring": "Harrell's C-index (RandomSurvivalForest.score) on held-out folds",
            "param_space": space,
            "max_samples_note": "absolute rows per tree (fractions "
                                f"{[f for f in MAX_SAMPLES_FRACS if f is not None]} of {base_n:,} "
                                f"rows); null = full bootstrap ({fold_train:,} rows per tree in CV, "
                                f"{base_n:,} at the production refit)",
            "total_wall_hours": r4(sum(r["wall_minutes"] for r in trials) / 60),
        },
        "best_params": best["params"],
        "best_is_baseline": best is base,
        "best_cv_c_index": cidx(best),
        "baseline": {"params": base["params"], "cv_c_index": cidx(base)},
        "delta_vs_baseline": {"mean": r4(np.mean(fold_delta)),
                              "folds": [r4(d) for d in fold_delta]},
        # Paste into Notebook 5's RSF_PARAMS (max_samples as an int keeps rows-per-tree
        # fixed whatever the training-set size; None = full bootstrap).
        "rsf_params": {
            "n_estimators": PRODUCTION_N_ESTIMATORS,
            **best["params"],
            "low_memory": False,
            "n_jobs": -1,
            "random_state": 42,
        },
        "max_samples_fraction_of_train": 1.0 if best_ms is None else r4(best_ms / base_n),
        "production_refit": {
            "n_estimators": PRODUCTION_N_ESTIMATORS,
            "low_memory": False,
            "est_forest_ram_gb": round(refit_ram_gb(best), 1),
            "baseline_est_forest_ram_gb": round(refit_ram_gb(base), 1),
            "note": "Estimate for Notebook 5's refit on all training rows: nodes per tree "
                    "measured in this search x 200 trees x (16 B x "
                    f"{n_times} unique times + {NODE_BYTES} B) per node. Prediction and "
                    "the data add a few GB on top.",
        },
        "trials": [{"trial": r["trial"], "params": r["params"], **cidx(r),
                    "nodes_per_tree": round(r["nodes_per_tree"]),
                    "wall_minutes": round(r["wall_minutes"], 1)} for r in trials],
    }
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    progress.unlink(missing_ok=True)

    print(f"\nsaved {out_json.name}")
    print(f"best: C={best['mean_c_index']:.4f} +/- {best['std_c_index']:.4f}   {describe(best['params'])}"
          f"{'   (the production baseline)' if best is base else ''}")
    print(f"production baseline: C={base['mean_c_index']:.4f}   "
          f"delta {np.mean(fold_delta):+.4f} (per fold: {', '.join(f'{d:+.4f}' for d in fold_delta)})")
    print(f"production refit (200 trees, low_memory=False) needs ~{refit_ram_gb(best):.1f} GB "
          f"for the forest (baseline config: ~{refit_ram_gb(base):.1f} GB)")
    print("rsf_params:", json.dumps(result["rsf_params"]))


if __name__ == "__main__":
    main()
