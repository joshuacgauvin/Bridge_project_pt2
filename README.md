# Random Survival Forests for Bridge Deterioration Prediction


### Predicting time-to-poor of bridges listed in the National Bridge Inventory (NBI) benchmarked against Cox Survival models.


## Motivation
Infrastructure maintenance can either be reactive, after a fault or issue occurs, or proactive, fixing small issues before the infrastructure has a large fault. The latter has been often shown to be far less costly in terms of both resources required for the proactive maintenance, and the impact the maintenance has on traffic in the context of bridge and roadway repairs.

The question concerning proactive maintenance, however, is when to conduct it. Maintenance too frequent can be inefficient and costly, diminishing the reason for preferring a proactive maintenance schedule over a reactive one. On the other hand, infrequent maintenance can also lead to inefficiencies where bridges aren't properly maintained as often as they should be. Finding the correct bridge schedule is also not a one-bridge-fits-all differing structures, lengths, traffic, weather, or exposure to sea salt can each affect a bridge's lifespan in different, nonlinear ways. For this reason, a tailored algorithm can be used to predict bridge deterioration.

For the reasons above, I am using a Random Survival Forest to predict the deterioration of a bridge to the "poor" category, as classified by the NBI. This model is best for interpreting nonlinear trends across large data sets, as one would expect to find in the wide variety of variables that impact bridge deterioration.


## Data
- National Bridge Inventory (1992 - 2025): Bridge data


- Daymet, retrieved via pydaymet: Environmental data


- Geopandas: Distance to coastline


## Methods
|Model | Role | Library|
| --- | --- | --- |
|Random Survival Forest|Proposed model|scikit-survival|
|Cox Proportional Hazards|Baseline|lifelines|
|Weibull AFT|Parametric baseline|lifelines|


## Evaluation Metrics


- Concordance Index (C-Index)


- Time-dependent AUC


- Integrated Brier Score (IBS)


## Findings

### National model comparison
All three models were trained on the same 1,206,337-bridge NBI dataset (first transition into "poor", structural + traffic + climate + coastal features) and scored on the identical 241,268-bridge held-out test split.

|Model | C-Index | IBS | Mean AUC (25/50/75 yr) | Fit time|
| --- | --- | --- | --- | --- |
|Weibull AFT|**0.9070**|**0.0475**|**0.9599**|5.7 min|
|Cox Proportional Hazards|0.9052|0.0483|0.9595|46 s|
|Random Survival Forest|0.8040|0.0645|0.9463|101 min|

The parametric baselines outperform the RSF at national scale. The RSF's hyperparameters were deliberately coarsened to fit 1.2M bridges in memory (`min_samples_leaf=50`, 10% per-tree subsampling), and the deterioration signal at this scale is captured well by a linear index over ~267 features — the RSF's advantage on nonlinearities and interactions does not overcome its capacity constraints here. The Cox model additionally yields interpretable per-feature hazard ratios (212 of 267 significant at p < 0.001) — e.g. orthotropic-deck designs (NBI 43B code 8) carry ~4.5x the baseline hazard, while culverts (code 19) carry ~0.19x.

**Leakage audit.** Climate features are joined as per-cell **1992–2025 normals**, not per-year values. The dataset keeps one row per bridge — the first at-poor observation for failures (median survey year 1996) but the last observation for censored bridges (median 2025) — so per-year climate joined at that row's survey year fingerprints the observation era and leaks censoring status. Measured on the shared test split, the per-year join (together with a survey-year-anchored reconstruction covariate, fixed the same way) had inflated the C-Index by **+0.016 (RSF), +0.011 (Cox), and +0.013 (AFT)**; the Pennsylvania case study measured the same mechanism at +0.09–0.13 for single-state tree models. All numbers reported here are from the corrected, era-free build.

Metrics: `us_rsf_metrics.json`, `us_parametric_metrics.json`. Hazard ratios / AFT coefficients: `us_cox_hazard_ratios.csv`, `us_aft_coefficients.csv` (regenerable from `train_national_parametric.ipynb`).

### Climate contribution (Pennsylvania case study)
The [Lu & Guler comparison study](Lu%26Guler_comparison/pa_deck_climate_study.ipynb) isolates what the environmental features add over a feature-matched structural baseline on PA state-owned bridges:

- Adding climate normals (1992–2025 Daymet cell means) + coastal distance improves the deck first-hit-poor C-Index by **+0.025** (95% CI +0.019 to +0.031) and the composite outcome by **+0.023** (95% CI +0.018 to +0.027); paired bootstrap over the shared test split, B = 1000, p < 0.002 for both.
- Caveat established along the way: joining climate by *observation year* inflates these deltas to +0.09–0.13 on first-hit outcomes, because the climate year leaks the observation era. Climate normals carry only the spatial exposure signal and give the honest estimate above.
- Replicating Lu & Guler (2022, TRR): their cumulative-truck-traffic clock result (RSF 0.836) reproduces on NBI-derived deck spells (0.876 here), and their calendar-clock finding also holds (0.666 here vs their 0.591) — the traffic clock, not the model, drives their headline number.

### Ablation (state indicators)
Including the 51 `STATE_*` dummies is a near-tie (median per-state C-Index delta +0.0017 *without* them; 31 of 50 states marginally better without) — they are kept since geography also enters through the continuous climate and coastal features. The same notebook also benchmarks a single-state specialist: an RSF trained only on Massachusetts bridges (same features, same test rows) scores 0.8323 within MA versus the national model's 0.6889 — pooled training wins between-state comparisons, but a dedicated state model ranks better within its own state.