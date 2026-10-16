"""Build the cleaned RSF tuning matrix -> tuning/rsf_tuning_train.csv.

One-time step (already run; re-run only if Notebook 4 regenerates the matrix):
    python tuning/prepare_tuning_data.py

Reads Notebook 4's encoded model matrix (Bridges_all_of_US/us_rsf_data_clean.csv) and
keeps ONLY the training rows of Notebook 5's 80/20 split (random_state=42), so the
hyperparameter search can never see a test bridge — the headline test C-index stays
untouched by tuning. Cleaning applied:

  - bridge_age dropped: it equals `time` on every row (target leak; Notebook 5 drops it too)
  - the Notebook 5 cell-03 guards re-checked: positive integer times, binary event, no
    condition-code / posting columns, no culvert dummy, all-numeric features, no inf,
    unique (STATE_FIPS, STRUCTURE_NUMBER_008) keys
  - NaNs are KEPT (1.4% of cells): scikit-survival's RSF splits on them natively and the
    production model trains with them, so imputing would tune on different data
  - all 255 features are kept (including two constant columns) so max_features
    fractions resolve to the same feature counts as in production

Output columns: STATE_FIPS, STRUCTURE_NUMBER_008, event, time, then the 255 features in
Notebook 5's X column order. Rows are in Notebook 5's X_train order. Values are written
back unconverted (parsed with float_precision="round_trip"), so they are exactly the
source CSV's values.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "Bridges_all_of_US" / "us_rsf_data_clean.csv"
TEST_RISK = HERE.parent / "Bridges_all_of_US" / "us_rsf_test_risk.csv"   # N5's test-set handoff
OUT = HERE / "rsf_tuning_train.csv"

KEYS = ["STATE_FIPS", "STRUCTURE_NUMBER_008"]


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"{SOURCE} not found — it is produced by "
                                "Bridges_all_of_US/build_national_rsf_dataset.ipynb (Notebook 4)")
    # round_trip parsing: pandas' default fast parser can land 1 ULP off on long decimals
    # (seen in dist_to_coast_km), which would make the written copy differ from the source.
    df = pd.read_csv(SOURCE, dtype={k: str for k in KEYS}, float_precision="round_trip")
    for k in KEYS:
        df[k] = df[k].str.strip()
    print(f"read {SOURCE.name}: {len(df):,} bridges x {df.shape[1]} columns")

    # ── Target and key guards ────────────────────────────────────────────────
    assert (df["time"] > 0).all(), "non-positive survival times"
    assert (df["time"] % 1 == 0).all(), "fractional survival times (RSF survival matrices would blow up)"
    assert df["event"].isin([0, 1]).all(), "event is not binary"
    assert not df.duplicated(KEYS).any(), "duplicate (STATE_FIPS, STRUCTURE_NUMBER_008) keys"
    assert (df["bridge_age"] == df["time"]).all(), \
        "bridge_age no longer equals time — re-check the leak logic before dropping it"

    # bridge_age == time: must drop or it leaks the target.
    X = df.drop(columns=["event", "time", "bridge_age"] + KEYS)

    # ── Feature guards (as in Notebook 5 cell-03) ────────────────────────────
    leak = [c for c in X.columns if "_COND_" in c or c == "OPEN_CLOSED_POSTED_041"]
    assert not leak, f"leakage columns present: {leak}"
    assert "STRUCTURE_TYPE_043B_19" not in X.columns, \
        "culvert dummy present — the matrix predates the Notebook 4 culvert exclusion; rebuild it"
    assert X.select_dtypes(exclude="number").shape[1] == 0, "non-numeric feature columns in X"
    assert not np.isinf(X.to_numpy(dtype="float64")).any(), "infinite feature values"

    # ── Notebook 5's 80/20 split ─────────────────────────────────────────────
    # train_test_split's partition depends only on (n, test_size, random_state), so
    # splitting row positions reproduces the notebook's X_train rows in the same order.
    train_pos, test_pos = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)

    if TEST_RISK.exists():
        ref = pd.read_csv(TEST_RISK, usecols=KEYS, dtype=str)
        held_out = df.iloc[test_pos][KEYS]
        same = (len(ref) == len(held_out)
                and set(zip(ref[KEYS[0]], ref[KEYS[1]])) == set(zip(held_out[KEYS[0]], held_out[KEYS[1]])))
        assert same, (f"split mismatch: {TEST_RISK.name} has {len(ref):,} test bridges that do not "
                      "match this split — was us_rsf_data_clean.csv rebuilt after Notebook 5 ran?")
        print(f"split check: held-out rows match {TEST_RISK.name} ({len(ref):,} test bridges)")
    else:
        print(f"split check skipped ({TEST_RISK.name} not found)")

    out = pd.concat([df[KEYS + ["event", "time"]], X], axis=1).iloc[train_pos]
    out.to_csv(OUT, index=False)

    n_feat = out.shape[1] - 4
    print(f"saved {OUT.name}: {len(out):,} training bridges x {n_feat} features "
          f"(+ 2 keys, event, time)   event rate: {out['event'].mean()*100:.1f}%   "
          f"NaN share: {out.iloc[:, 4:].isna().mean().mean()*100:.2f}%   "
          f"{OUT.stat().st_size/1e6:.0f} MB")


if __name__ == "__main__":
    main()
