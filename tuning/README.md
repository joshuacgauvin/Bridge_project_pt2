# RSF hyperparameter tuning (cloud bundle)

A self-contained random search for the national Random Survival Forest's
hyperparameters. It is sized for a cloud machine: the laptop-sized search had to stay
within leaf ≥ 20 and ≤ 15% of rows per tree. This one reaches leaf 3 and full bootstrap,
where the forest scores best. Copy this folder to the machine, run one command, and
bring back one JSON file.

## What's in the folder

| File | What it is | Needed on the cloud? |
|---|---|---|
| `rsf_tuning_train.csv` | Cleaned data: Notebook 5's 641,836 training bridges × 255 features + keys, event, time (473 MB). The test split is not in it. | Yes |
| `tune_rsf.py` | The search | Yes |
| `requirements.txt` | Pinned packages (Python 3.12, x86-64) | Yes |
| `run_tuning.sh` | Linux/macOS launcher: venv, installs, detached run | Yes (Linux) |
| `prepare_tuning_data.py` | Rebuilds the CSV from `../Bridges_all_of_US/us_rsf_data_clean.csv` (already run) | No |

**Copy the folder itself, not a git clone.** `*.csv` is gitignored, and the file is
over GitHub's 100 MB limit, so a clone arrives without the data. For example:

```
scp -r tuning/ user@your-vm:~/
```

or use the cloud console's upload or a storage bucket.

## Run it

```
cd tuning
bash run_tuning.sh --smoke     # ~1 min: installs packages, checks everything end to end
bash run_tuning.sh             # the full search, detached (survives SSH disconnects)
tail -f tuning.log             # progress: one line per configuration, with an ETA
```

**Use an x86-64 (Intel/AMD) machine with Python 3.12.** Only that combination has
prebuilt wheels for every pinned package. On anything else pip tries to compile from
source and usually fails:

- numpy 2.5 dropped Python 3.11;
- `ecos` has no Python 3.13 wheel;
- scikit-survival has no ARM wheel, which rules out Graviton/Ampere/Axion instances.

**Ubuntu 24.04 is the easy choice:** it ships Python 3.12. Ubuntu 22.04 ships 3.10.

- If 3.12 is installed under another name, point the launcher at it:
  `PYTHON=python3.12 bash run_tuning.sh`.
- `python3 -m venv` may need `sudo apt install python3.12-venv`.

The launcher checks the CPU type and Python version before it installs anything.

Without bash, for example on a Windows VM or in a hosted notebook:

```
pip install -r requirements.txt
python tune_rsf.py
```

The smoke run writes `*_smoke` files, which you can delete.

## Choosing a machine

**CPU drives the runtime; RAM barely matters for the search.**

- **Measured:** the script's peak at full scale was 2.4 GB, with all 3 folds fitted at
  once.
- **Worst case:** the largest configurations (leaf 3, full bootstrap) and a many-core
  machine's thread buffers add about 3 GB, for roughly 6 GB in total.
- **Recommendation:** 8 GB is the minimum and 16 GB is comfortable. Any 64+ vCPU machine
  has far more.
- **If RAM is short,** `--fold-jobs 1` fits one fold at a time.

Estimated runtimes for the default search (≈1,600 thread-hours). These are rough, ±2×,
fitted from timed fits at three corners of the space:

| vCPUs | Estimated wall time |
|---|---|
| 64 | ~25 h |
| 96 | ~17 h |
| 192 | ~8 h |

- **Choose a compute-optimised x86-64 instance.** Each configuration fits 150 trees at once
  (3 folds × 50), so vCPUs beyond about 150 sit idle.
- **Expect a per-configuration floor of about 2 h.** A single tree runs on one core,
  and in the most expensive configurations one tree takes roughly 2 hours.
- **Spot or preemptible instances are fine** as long as the folder is on a persistent
  disk. Rerunning resumes the search.

## Stopping and resuming

- After every configuration, the result is appended to `rsf_tuning_progress.jsonl`.
- Rerun the same command to continue. Only the configuration that was in flight is
  redone.
- Raising `--n-iter` extends a finished or partial search.
- Changing `--cv`, `--n-estimators`, `--n-tune` or `--seed` requires `--fresh`, which
  discards the saved progress.
- When the search finishes, the progress file is folded into the JSON and deleted.

## The output: `rsf_best_hyperparameters.json`

| Field | What it holds |
|---|---|
| `rsf_params` | Ready to paste into `RSF_PARAMS`. Uses production's 200 trees and `low_memory=False`; `max_samples` is an absolute row count, or `null` for a full bootstrap. |
| `best_cv_c_index`, `baseline`, `delta_vs_baseline` | The winner against today's shipped settings, scored on the same folds (per fold and mean). |
| `production_refit.est_forest_ram_gb` | RAM the Notebook 5 refit of the winner will need (see below). |
| `trials` | Every configuration, best first. |

Copy the JSON back into this folder in the repo.

## Using the result

1. **Check `production_refit.est_forest_ram_gb` first.** This is the memory wall the
   laptop hit.
   - Notebook 5 needs `low_memory=False` for survival curves, which stores about 3.4 KB
     per tree node.
   - Fine configurations grow far more nodes. Probe estimates were ~25 GB for leaf 3 with
     160k rows per tree, and ~80 GB for leaf 3 with a full bootstrap. The shipped
     settings need 1.1 GB.
   - Run the refit on a machine with that much RAM plus about 10 GB of headroom.
2. **Update `RSF_PARAMS` in all four notebooks that define it**, or the model
   comparisons mix configurations:
   - `train_national_rsf.ipynb`
   - `leakage_ablation_national.ipynb`
   - `temporal_validation_national.ipynb`
   - `ablation_state_dummies.ipynb` (`FULL_RSF_PARAMS`)
3. **Regenerate the RSF headline numbers.** Every RSF figure in the paper changes.

## How the search works

| Parameter | This search | Laptop search |
|---|---|---|
| `max_samples` (rows per tree) | 5/10/25/50% of 641,836, or full bootstrap | 2.5–15% |
| `min_samples_leaf` | 3, 5, 10, 20, 35, 50, 75, 100 | 20–99 |
| `min_samples_split` | 6, 10, 20, 50, 100, 200 | 50–299 |
| `max_features` | log2, sqrt, 0.2, 0.3, 0.4 | sqrt, 0.2, 0.3 |
| `max_depth` | None, 10, 15, 20 | not searched |
| Configurations × folds × trees | 50 × 3 × 50 | 15 × 3 × 50 on a 100k subsample |

- **Configurations:** 50 drawn at random, as many as the Massachusetts search
  (`ma_bridges/RSF.ipynb`), plus today's shipped settings as a baseline.
- **Folds:** shuffled `StratifiedKFold` on the event, shared by every configuration. The
  notebook's old `RUN_TUNING` cell used unshuffled folds over rows sorted by state, so
  each validation fold was a block of whole states.
- **Precision:** fold-to-fold spread is about ±0.002, far smaller than the gaps between
  configurations (a 16-tree probe at leaf 3 scored 0.83 vs 0.77 shipped). 3 folds × 50
  trees therefore ranks configurations reliably. `--cv 5 --n-estimators 100` gives exact
  Massachusetts parity at about 4× the compute.
- **Rows per tree** are searched as absolute counts, so the winner transfers to the full
  refit unchanged.
- **Scoring** is Harrell's C-index, the headline metric.
- **Low-memory mode** grows identical trees and risk scores to `low_memory=False`
  (verified), so the search's scores hold for the production refit.

Run `python tune_rsf.py --help` for every flag. The most useful ones are `--n-iter`,
`--cv`, `--n-estimators`, and `--fold-jobs` (fewer folds at once if RAM is short).
