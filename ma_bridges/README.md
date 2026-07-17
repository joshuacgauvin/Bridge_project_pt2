# ma_bridges/ — Massachusetts prototype (superseded)

**Status: earlier exploratory work, superseded by the national pipeline in
[`../Bridges_all_of_US/`](../Bridges_all_of_US/).** Nothing in the current results
depends on this folder; it is kept because the national notebooks document their
lineage against it ("ports the validated Massachusetts pipeline") and because it
contains the one analysis the national pipeline cannot reproduce: MA-DOT maintenance
records (`maintenance.csv`) exist only for Massachusetts.

| Notebook | What it does |
|---|---|
| `bridge_project.ipynb` | Downloads/builds the MA NBI panel (1992–2025), merges MA-DOT maintenance records, pulls Daymet climate, and produces the cleaned `massachusetts_bridges_rsf_ready.csv` |
| `Build_RSF.ipynb` | Collapses the panel to survival format (one row per bridge) and one-hot encodes it |
| `RSF.ipynb` | Trains the standalone MA Random Survival Forest (1000 trees, leaf=5) + per-bridge predictions |
| `bridge_map.ipynb` | Plotly risk map for MA |

Caveats for anyone reading these notebooks today:

- **Numbers in here are historical.** They predate two later corrections applied to
  the national pipeline: the exclusion of culverts (whose deck/superstructure/
  substructure ratings are "N", so the poor-condition event was undefined for them)
  and the panel-entry measurement of the reconstruction covariates. See the root
  [`README.md`](../README.md) "Leakage audit" and
  `../Bridges_all_of_US/build_national_rsf_dataset.ipynb` for the current definitions.
- The CSVs these notebooks read/write are gitignored, so they are not runnable
  without regenerating the data.
- The MA-specialist benchmark the national work cites is re-derived inside the
  national pipeline itself (`../Bridges_all_of_US/ablation_state_dummies.ipynb`,
  Arm C) — not from this folder.
