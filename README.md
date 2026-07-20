  # Survival Models for Bridge Deterioration Prediction


### Comparing a Random Survival Forest against Cox PH and Weibull AFT on ~1M non-culvert NBI bridges and quantifying what climate data adds.


## Motivation
Infrastructure maintenance can either be reactive, after a fault or issue occurs, or proactive, fixing small issues before the infrastructure has a large fault. The latter has often been shown to be far less costly in terms of both resources required for the proactive maintenance and the impact the maintenance has on traffic in the context of bridge and roadway repairs.

The question concerning proactive maintenance, however, is when to conduct it. Maintenance that is too frequent can be inefficient and costly, diminishing the reason for preferring a proactive maintenance schedule over a reactive one. On the other hand, infrequent maintenance can also lead to inefficiencies where bridges aren't properly maintained as often as they should be. Finding the correct maintenance schedule is also not a one-bridge-fits-all problem. Differing structures, lengths, traffic, weather, or exposure to sea salt can each affect a bridge's lifespan in different, nonlinear ways. For this reason, a tailored algorithm can be used to predict bridge deterioration.

I tested which models are best for predicting bridge deterioration at the national level using the Random Survival Forest, Cox Proportional Hazards, and Weibull AFT models.

I am also including climate data in the models to account for deterioration due to weather and exposure in addition to data on the physical bridges themselves.

## Repository structure
| Folder | Role |
| --- | --- |
| [`Bridges_all_of_US/`](Bridges_all_of_US/) | **The main national pipeline** — data download/build (Notebooks 1–4), model training (5–6), risk map (7), ablations (8), plus the leakage ablation (`leakage_ablation_national.ipynb`) and bootstrap comparison (`bootstrap_model_comparison.ipynb`) |
| [`Lu+Guler_comparison/`](Lu%2BGuler_comparison/) | Self-contained Pennsylvania case study: a Lu & Guler (2022)-style deck analysis + the climate-contribution measurement |
| [`ma_bridges/`](ma_bridges/) | **Superseded** Massachusetts prototype the national pipeline was ported from — kept for lineage; not part of the current results (see its [README](ma_bridges/README.md)) |

## Data
- National Bridge Inventory (1992 - 2025): Bridge data. **Culverts (NBI item 43B code 19; 19.2% of structures) are excluded**: under NBI convention they report the deck/superstructure/substructure condition ratings the event is built from as "N" (not applicable — 99.3% of culverts), so the deterioration event is undefined for them (see the leakage audit below).


- Daymet, retrieved via pydaymet: Environmental data


- Geopandas: Distance to coastline


## Methods
|Model | Role | Library|
| --- | --- | --- |
|Random Survival Forest|Machine Learning Model|scikit-survival|
|Cox Proportional Hazards|Semiparametric Model|lifelines|
|Weibull AFT|Parametric Model|lifelines|


## Evaluation Metrics


- Concordance Index (C-Index)


- Mean AUC


- Integrated Brier Score (IBS)


## Findings

### National model comparison
All three models were trained on the same 973,905-bridge NBI dataset (first transition into "poor", structural + traffic + climate + coastal features; culverts and other structures without rateable deck/superstructure/substructure components excluded) and scored on the identical 194,781-bridge held-out test split.

|Model | C-Index | IBS | Mean AUC (25/50/75 yr) | Fit time|
| --- | --- | --- | --- | --- |
|Weibull AFT|**0.8839**|0.0565|0.9505|4.6 min|
|Cox Proportional Hazards|0.8835|**0.0563**|**0.9518**|50 s|
|Random Survival Forest|0.7547|0.0758|0.9381|154 min|

Uncertainty on the split (paired bootstrap over the shared test set, B = 1000, `bootstrap_model_comparison.ipynb` → `us_model_bootstrap.json`): AFT [0.8825, 0.8851], Cox [0.8822, 0.8847], RSF [0.7525, 0.7570]. The parametric-vs-RSF gap is decisive (Cox−RSF +0.1288, 95% CI [+0.1271, +0.1305], p < 0.002), while Cox and AFT are practically tied (AFT−Cox +0.0004, 95% CI [+0.0001, +0.0006]; statistically nonzero, practically nothing).

The parametric models outperform the RSF at national scale. However, the RSF's hyperparameters were deliberately coarsened to fit ~1M bridges in memory (`min_samples_leaf=50`, 10% per-tree subsampling); a **cost-matched capacity check** (`ablation_state_dummies.ipynb`, Arm D: MA-specialist hyperparameters — 300 trees, leaf 5, full bootstrap — scored on the full test split) reaches pooled C = 0.7879 ± 0.0041 on 15,057-row national subsamples (5 seeds) versus 0.7547 for the full coarsened forest. Pushing the same fine-grained forest up a **capacity curve** — 50k: 0.7973, 100k: 0.8026 ± 0.0025, 200k: 0.8062 (leaf 5 with a full bootstrap is memory-bound past ~200k, the same wall that forced the coarsening) — confirms the coarsening does understate the RSF, but with sharply diminishing returns: the successive gains are +0.0094, +0.0053, then +0.0036, so each doubling of the data now adds under ~0.005. The gap to Cox narrows from +0.129 (full coarse) to +0.096 (cost-matched) to +0.077 at the 200k ceiling; that flattening curve stalls well short of Cox's 0.8835, so the parametric advantage at national scale is a genuine capacity limit rather than primarily a hyperparameter artifact.

The Cox model additionally yields interpretable per-feature hazard ratios, 206 of 266 covariates are significant at p < 0.001, yet only 157 of those carry hazard ratios more than 10% away from 1.0 (and 171 more than 5%). Among the material effects: orthotropic-deck designs (NBI 43B code 8) carry ~3.3x the baseline hazard, while epoxy-coated-reinforcement decks (item 108C code 1) carry ~0.40x (both p < 0.001). A Schoenfeld-residual check (`lifelines.proportional_hazard_test`, rank transform; `us_cox_ph_check.json`) flags 40 of 266 covariates at p < 0.001, construction year is by far the largest offender, i.e. era effects are genuinely time-varying so Cox is reported as a discriminative benchmark and its hazard ratios read as time-averaged effects, not as a validated proportional-hazards structural model.

**Leakage audit.** Climate features are joined as per-cell **1992–2025 normals**, not per-year values, and reconstruction covariates are measured at each bridge's **panel-entry row**, not the kept row. The dataset keeps one row per bridge — the first at-poor observation for failures (median survey year 1996) but the last observation for censored bridges (median 2025) — so per-year climate joined at that row's survey year fingerprints the observation era and leaks censoring status; the kept-row reconstruction year is likewise truncated at the kept year (among reconstructed bridges, 1.0% of events vs 10.8% of censored bridges show a post-2010 reconstruction under the old definition, an impossible-by-construction asymmetry; before/after: `us_reconstruction_measurement.json`). Measured on the shared test split by refitting all three models on the leaky build (`leakage_ablation_national.ipynb` and `us_leakage_ablation.json`), the leaky construction inflates the C-Index by **+0.0210 (RSF), +0.0132 (Cox), and +0.0155 (AFT)**; the Pennsylvania case study measured the same mechanism at +0.09–0.13 for single-state tree models. All numbers reported here are from the corrected, era-free build. The same audit removed the earlier "culverts carry ~0.19x the baseline hazard" claim: 99.3% of culverts report all three event-defining condition ratings as "N" (event rate 0.12% vs 29.2% for non-culverts), so that coefficient was a labeling artifact of an event culverts could barely register, not a finding, therefore culverts are excluded from the study population (`us_culvert_diagnostic.json`).

Metrics: `us_rsf_metrics.json`, `us_parametric_metrics.json`, `us_model_bootstrap.json`, `us_leakage_ablation.json`, `us_culvert_diagnostic.json`, `us_reconstruction_measurement.json`, `us_cox_ph_check.json`. Hazard ratios / AFT coefficients: `us_cox_hazard_ratios.csv`, `us_aft_coefficients.csv` (regenerable from `train_national_parametric.ipynb`).

### Climate contribution (Pennsylvania case study)
The [Lu & Guler comparison study](Lu%2BGuler_comparison/pa_deck_climate_study.ipynb) isolates what the environmental features add over a feature-matched structural baseline on PA state-owned bridges:

- Adding climate normals (1992–2025 Daymet cell means) + coastal distance improves the deck first-hit-poor C-Index by **+0.025** (95% CI +0.019 to +0.031) and the composite outcome by **+0.023** (95% CI +0.018 to +0.027) — paired bootstrap over the shared test split, B = 1000, p < 0.002 for both.
- Caveat established along the way: joining climate by *observation year* inflates these deltas to +0.09–0.13 on first-hit outcomes, because the climate year leaks the observation era. Climate normals carry only the spatial exposure signal and give the honest estimate above.
- **Comparison with Lu & Guler (2022)** — a similar analysis on NBI-derived deck spells (not a replication: public NBI panel rather than their PennDOT records, and a static ADTT-at-spell-start × years traffic clock rather than their accumulated time-varying ADTT). All three published numbers, compared honestly: their traffic-clock RSF 0.836 vs 0.876 here and calendar-clock RSF 0.591 vs 0.666 here agree qualitatively; their traffic-clock AFT 0.693 vs **0.832 here diverges** — plausibly the static-rate clock, which a linear AFT can exploit as easily as a forest. The conclusion that survives both datasets is qualitative: **the clock (cumulative traffic vs calendar time), not the model family, drives the headline number.**

### Ablations (state indicators + RSF capacity)
Including the 51 `STATE_*` dummies is a near-tie (median per-state C-Index delta -0.0010 *without* them; 21 of 50 states marginally better without) — they are kept since geography also enters through the continuous climate and coastal features. The same notebook benchmarks a single-state specialist: an RSF trained only on Massachusetts bridges (same features, same test rows) scores 0.8376 within MA versus the national model's 0.6646 — pooled training wins between-state comparisons, but a dedicated state model ranks better within its own state. Arm D (described above) bounds how much of the RSF-vs-parametric gap is explained by the memory-driven hyperparameter coarsening.

## References
- Lu, M., & Guler, S. I. (2022). Comparison of Random Survival Forest with Accelerated Failure Time-Weibull Model for Bridge Deck Deterioration. *Transportation Research Record*, 2676(8). https://doi.org/10.1177/03611981221078281
- Federal Highway Administration. *National Bridge Inventory* (1992–2025 delimited ASCII files). U.S. Department of Transportation. https://www.fhwa.dot.gov/bridge/nbi.cfm
- Thornton, M. M., et al. (2022). *Daymet: Daily Surface Weather Data on a 1-km Grid for North America, Version 4 R1.* ORNL DAAC. https://doi.org/10.3334/ORNLDAAC/2129 (retrieved via `pydaymet`)
- Pölsterl, S. (2020). scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn. *Journal of Machine Learning Research*, 21(212), 1–6.
- Davidson-Pilon, C. (2019). lifelines: survival analysis in Python. *Journal of Open Source Software*, 4(40), 1317. https://doi.org/10.21105/joss.01317
- Natural Earth. *10m Coastline* vector dataset (used for coastal distances via geopandas). https://www.naturalearthdata.com/

