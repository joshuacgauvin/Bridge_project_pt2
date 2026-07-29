  # Survival Models for Bridge Deterioration Prediction


### Comparing a Random Survival Forest against Cox PH and Weibull AFT on ~1M non-culvert NBI bridges and quantifying what climate data adds.


## Motivation
Infrastructure maintenance can either be reactive, after a fault or issue occurs, or proactive, fixing small issues before the infrastructure has a large fault. The latter has often been shown to be far less costly in terms of both resources required for the proactive maintenance and the impact the maintenance has on traffic in the context of bridge and roadway repairs.

The question concerning proactive maintenance, however, is when to conduct it. Maintenance that is too frequent can be inefficient and costly, diminishing the reason for preferring a proactive maintenance schedule over a reactive one. On the other hand, infrequent maintenance can also lead to inefficiencies where bridges aren't properly maintained as often as they should be. Finding the correct maintenance schedule is also not a one-bridge-fits-all problem. Differing structures, lengths, traffic, weather, or exposure to sea salt can each affect a bridge's lifespan in different, nonlinear ways. For this reason, a tailored algorithm can be used to predict bridge deterioration.

I tested which models are best for predicting bridge deterioration at the national level using the Random Survival Forest, Cox Proportional Hazards, and Weibull AFT models.

I am also including climate data in the models to account for deterioration due to weather and exposure in addition to data on the physical bridges themselves. Ablating that environmental block improves every model tested — by +0.0171 C-Index for the Random Survival Forest and about +0.0005 for the parametric models nationally, and by +0.025 for the Pennsylvania deck case study — so the climate features earn their place, though how much they buy depends heavily on the model family (see [Climate contribution](#climate-contribution-national-ablation)).

## Repository structure
| Folder | Role |
| --- | --- |
| [`Bridges_all_of_US/`](Bridges_all_of_US/) | **The main national pipeline** — data download/build (Notebooks 1–4), model training (5–6), risk map (7), ablations (8), plus the leakage ablation (`leakage_ablation_national.ipynb`), temporal validation (`temporal_validation_national.ipynb`), and bootstrap comparison (`bootstrap_model_comparison.ipynb`) |
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
|Weibull AFT|**0.8833**|0.0567|0.9502|5.2 min|
|Cox Proportional Hazards|0.8830|**0.0565**|**0.9515**|53 s|
|Random Survival Forest|0.7526|0.0761|0.9383|39 min|

Uncertainty on the split (paired bootstrap over the shared test set, B = 1000, `bootstrap_model_comparison.ipynb` → `us_model_bootstrap.json`): AFT [0.8819, 0.8846], Cox [0.8816, 0.8843], RSF [0.7504, 0.7549]. The parametric-vs-RSF gap is decisive (Cox−RSF +0.1304, 95% CI [+0.1287, +0.1322], p < 0.002), while Cox and AFT are practically tied (AFT−Cox +0.0003, 95% CI [+0.0001, +0.0005]; statistically nonzero, practically nothing).

The parametric models outperform the RSF at national scale. However, the RSF's hyperparameters were deliberately coarsened to fit ~1M bridges in memory (`min_samples_leaf=50`, 10% per-tree subsampling); a **cost-matched capacity check** (`ablation_state_dummies.ipynb`, Arm D: MA-specialist hyperparameters — 300 trees, leaf 5, full bootstrap — scored on the full test split) reaches pooled C = 0.7872 ± 0.0038 on 15,057-row national subsamples (5 seeds) versus 0.7526 for the full coarsened forest. Pushing the same fine-grained forest up a **capacity curve** — 50k: 0.7963, 100k: 0.8017 ± 0.0026, 200k: 0.8063 (leaf 5 with a full bootstrap is memory-bound past ~200k, the same wall that forced the coarsening) — confirms the coarsening does understate the RSF, but with sharply diminishing returns: the successive gains are +0.0091, +0.0054, then +0.0046, so each doubling of the data now adds under ~0.005. The gap to Cox narrows from +0.130 (full coarse) to +0.096 (cost-matched) to +0.077 at the 200k ceiling; that flattening curve stalls well short of Cox's 0.8830, so the parametric advantage at national scale is a genuine capacity limit rather than primarily a hyperparameter artifact.

The Cox model additionally yields interpretable per-feature hazard ratios, 204 of 266 covariates are significant at raw p < 0.001 and because significance is cheap at N ≈ 974k, this count is robust to multiple-testing correction (218 survive a Benjamini–Hochberg FDR at q < 0.05; 198 even under the far stricter Bonferroni family-wise bound at α = 0.05; counts derived from `us_cox_hazard_ratios.csv`). Significance is therefore not the discriminating lens: only 155 of those carry hazard ratios more than 10% away from 1.0 (and 168 more than 5%). Among the material effects: orthotropic-deck designs (NBI 43B code 8) carry ~3.2x the baseline hazard, while epoxy-coated-reinforcement decks (item 108C code 1) carry ~0.40x. A Schoenfeld-residual check (`lifelines.proportional_hazard_test`, rank transform; `us_cox_ph_check.json`) flags 42 of 266 covariates at p < 0.001, construction year is by far the largest offender, i.e. era effects are genuinely time-varying so Cox is reported as a discriminative benchmark and its hazard ratios read as time-averaged effects, not as a validated proportional-hazards structural model.

**Fit times** were aquired from a single run of each model. The specifications for the machine the models were ran on are detailed below:

- CPU: AMD Ryzen 7 7700X 8-Core Processor (4.50 GHz)

- RAM: 32 GB

- OS: Windows 11

**Leakage audit.** Three feature groups are measured era-free to avoid fingerprinting the observation window: climate joins as per-cell **1992–2025 normals** (not per-year values), and both reconstruction and **traffic** covariates (ADT, lane counts, truck share) are read at each bridge's **panel-entry row**, not the kept row. The dataset keeps one row per bridge — the first at-poor observation for failures (median survey year 1996) but the last observation for censored bridges (median 2025) — so any covariate re-measured over time leaks censoring status if read at the kept row. Per-year climate joined at that row's survey year does this; the kept-row reconstruction year is truncated at the kept year (among reconstructed bridges, 1.0% of events vs 10.8% of censored bridges show a post-2010 reconstruction under the old definition, an impossible-by-construction asymmetry; re-anchoring to each bridge's panel-entry row narrows this to 0.56% (events) vs 1.69% (censored) — roughly 3x apart rather than 11x, not fully flat, and reported honestly as a residual, smaller leak rather than a closed issue; before/after: `us_reconstruction_measurement.json`); and kept-row ADT carries the same fingerprint (measurement year `YEAR_ADT_030` median 1992 for events vs 2011 for censored; kept-row ADT median 225 vs 850; `us_traffic_leakage_measurement.json`). Measured on the shared test split by refitting all three models on the leaky build (`leakage_ablation_national.ipynb` and `us_leakage_ablation.json`), the fully leaky construction (per-year climate + kept-row reconstruction + kept-row traffic, 23 columns swapped) inflates the C-Index by **+0.0232 (RSF), +0.0137 (Cox), and +0.0161 (AFT)**. Swapping **only** the four traffic columns isolates a near-zero effect — **+0.0010 (RSF), +0.0005 (Cox), +0.0006 (AFT)** — so despite the stark measurement-era gap, kept-row traffic barely moves discrimination (ADT is one feature among 266); the climate and reconstruction constructions carry essentially all of the inflation. The Pennsylvania case study measured the climate mechanism at +0.09–0.13 for single-state tree models. All numbers reported here are from the corrected, era-free build. The same audit removed the earlier "culverts carry ~0.19x the baseline hazard" claim: 99.3% of culverts report all three event-defining condition ratings as "N" (event rate 0.12% vs 29.2% for non-culverts), so that coefficient was a labeling artifact of an event culverts could barely register, not a finding, therefore culverts are excluded from the study population (`us_culvert_diagnostic.json`).

Metrics: `us_rsf_metrics.json`, `us_parametric_metrics.json`, `us_model_bootstrap.json`, `us_ablation_state_dummies.json`, `us_leakage_ablation.json`, `us_culvert_diagnostic.json`, `us_reconstruction_measurement.json`, `us_traffic_leakage_measurement.json`, `us_cox_ph_check.json`, `us_temporal_validation.json`. Hazard ratios / AFT coefficients: `us_cox_hazard_ratios.csv`, `us_aft_coefficients.csv` (regenerable from `train_national_parametric.ipynb`).

### Predicting new bridges (temporal validation)
The headline split is random, so every construction era appears on both sides. To test the stricter deployment question, can a model trained on today's bridges rank *tomorrow's*? `temporal_validation_national.ipynb` trains on the 752,897 bridges built **before 2015** (227,941 events) and scores the 32,769 bridges built **in 2015 or later** (96 events, none seen in training). An in-era control (the standard test split restricted to pre-2015 bridges) reproduces the headline C-Index for all three models (RSF 0.7536, Cox 0.8835, AFT 0.8834), confirming the setup.

On the never-seen-era bridges all three models rank well above chance, and the random survival forest outperforms the parametric models: RSF **0.8365** (95% CI [0.786, 0.880]), Cox **0.7931** ([0.731, 0.843]), AFT **0.7972** ([0.739, 0.843]) (percentile bootstrap, B = 1000). With only 96 post-2015 events those intervals overlap heavily, so the 2015 cutoff alone does not resolve the ordering.

**A 2010 cutoff (54,282 bridges, 303 events) resolves it, for all three models:**

|Model | 2015 cutoff (96 events) | 2010 cutoff (303 events) | in-era control (2010) |
| --- | --- | --- | --- |
|Random Survival Forest|0.8365 [0.786, 0.880]|**0.8574** [0.8316, 0.8819]|0.7541|
|Cox Proportional Hazards|0.7931 [0.731, 0.843]|0.7804 [0.7515, 0.8097]|0.8838|
|Weibull AFT|0.7972 [0.739, 0.843]|0.7880 [0.7600, 0.8151]|0.8833|

At 303 events the intervals **separate** — the RSF's lower bound (0.8316) sits above the AFT's upper bound (0.8151). All three models score the identical 54,282 rows, so non-overlapping marginal intervals is a conservative criterion here and a paired test would only be more decisive. The movement is symmetric: going from their own era to an unseen one, Cox and AFT each give up ~0.10 C-Index, while the RSF *gains* ~0.10 (0.7541 in-era → 0.8574 new-era). The forest ranks bridges it has never seen better than the era it trained on, at both cutoffs. The likely reason is that these are different ranking problems: the post-cutoff cohort is young and narrow in age, which strips out most of the variance in `YEAR_BUILT` — the covariate the parametric models lean on hardest and the worst offender in the Schoenfeld check — leaving the structural and environmental variation where the forest's interaction modelling pays off. Caveats: the observable horizon for new bridges is ≤ 10 years (this validates early-life risk ranking, not long-horizon prediction), and the RSF cannot extrapolate `YEAR_BUILT` past the training range, so its new-bridge ranking rests on the non-age features.

### Climate contribution (national ablation)
`ablation_state_dummies.ipynb` (Arm E) refits **all three models with and without the environmental block** — the 17 Daymet 1992–2025 cell normals plus `dist_to_coast_km`, 18 columns — at full shipped fidelity on the 779,124-bridge training split, scored on the identical 194,781-bridge test set. The geocoding-quality flag `location_imputed` stays in *both* arms (it records data provenance, not exposure). Both "with environment" arms reproduce the headline C-Indices above to within 1e-3, confirming the ablation is measured against the shipped models.

|Model | Without environment | With environment | Delta | 95% CI |
| --- | --- | --- | --- | --- |
|Random Survival Forest|0.7355|0.7526|**+0.0171**|[+0.0165, +0.0176]|
|Cox Proportional Hazards|0.8825|0.8830|+0.0005|[+0.0004, +0.0006]|
|Weibull AFT|0.8828|0.8833|+0.0004|[+0.0003, +0.0006]|

Paired bootstrap over the shared test split, B = 1000, seed 2026; all three intervals exclude zero at p < 0.002 (`us_ablation_state_dummies.json` → `arm_E_climate_contribution`). **Environmental data measurably improves prediction for every model tested** — but the size of the improvement splits sharply by model family. The raw ~30:1 ratio between the forest's gain and the parametric models' overstates the split, because the two start from very different baselines and so have very different room left to improve; normalising each gain by its own remaining headroom (1 − C) gives 6.5% of the forest's headroom against 0.4% of Cox's, a ratio nearer **fifteen**. Quote the headroom-normalised figure, not the raw ratio.

Climate normals describe a smooth spatial field whose effect is plausibly nonlinear and interactive (freeze-thaw cycling matters more for some deck types, coastal salt for some protection systems), which a forest represents directly through splits and a single linear term per covariate cannot. The Cox coefficients corroborate this: per-unit hazard ratios for the climate features sit essentially at 1.0 (snow days 1.0034/day, freeze-thaw 1.0009/cycle) and `dist_to_coast_km` is not significant at all (p = 0.19) — even though it ranks 19th of 255 features in RSF permutation importance. Two caveats: the parametric null is **specification-dependent** (explicit climate-by-structure interaction terms could plausibly recover part of what the forest finds), and no size-matched non-environmental control block was run, so this does not isolate environmental data from feature removal in general. Dropping the 51 `STATE_*` indicators costs the forest only 0.0010 against 0.0171 for the 18 environmental columns, which is suggestive but not a matched control (different `n_train`, coarser hyperparameters, no parametric counterpart).

### Climate contribution (Pennsylvania case study)
The [Lu & Guler comparison study](Lu%2BGuler_comparison/pa_deck_climate_study.ipynb) isolates the same quantity on a single state's decks, over a feature-matched structural baseline on PA state-owned bridges:

- Adding climate normals (1992–2025 Daymet cell means) + coastal distance improves the deck first-hit-poor C-Index by **+0.025** (95% CI +0.019 to +0.031) and the composite outcome by **+0.023** (95% CI +0.018 to +0.027) — paired bootstrap over the shared test split, B = 1000, p < 0.002 for both.
- Caveat established along the way: joining climate by *observation year* inflates these deltas to +0.09–0.13 on first-hit outcomes, because the climate year leaks the observation era. Climate normals carry only the spatial exposure signal and give the honest estimate above.
- **Comparison with Lu & Guler (2022)** — a similar analysis on NBI-derived deck spells (not a replication: public NBI panel rather than their PennDOT records, and a static ADTT-at-spell-start × years traffic clock rather than their accumulated time-varying ADTT). All three published numbers, compared honestly: their traffic-clock RSF 0.836 vs 0.876 here and calendar-clock RSF 0.591 vs 0.666 here agree qualitatively; their traffic-clock AFT 0.693 vs **0.832 here diverges** — plausibly the static-rate clock, which a linear AFT can exploit as easily as a forest. The conclusion that survives both datasets is qualitative: **the clock (cumulative traffic vs calendar time), not the model family, drives the headline number.**

### Ablations (state indicators + RSF capacity)
Including the 51 `STATE_*` dummies is a near-tie that leans marginally toward dropping them (median per-state C-Index delta +0.0019 *without* them; 34 of 50 states marginally better without) — they are kept because the gap is within noise and they cost the RSF nothing, and geography also enters through the continuous climate and coastal features. The same notebook benchmarks a single-state specialist: an RSF trained only on Massachusetts bridges (same features, same test rows) scores 0.8356 within MA versus the national model's 0.6755 — pooled training wins between-state comparisons, but a dedicated state model ranks better within its own state. Arm D (described above) bounds how much of the RSF-vs-parametric gap is explained by the memory-driven hyperparameter coarsening, and Arm E (the national climate ablation above) measures what the environmental block contributes to each of the three models. All five arms write to `us_ablation_state_dummies.json`.

## Reproducibility
`reproduce_check.py` (repo root) reruns the notebook chain end-to-end and verifies it reproduces the tracked `.json` / `.csv` artifacts.

- **Smoke check** (`python reproduce_check.py`, ~15–30 min) patches each notebook's smoke switch **in memory only** — `STATE_SUBSET = ["MA", "RI"]` for the dataset build (Massachusetts is required because the ablation's specialist arms are anchored to it), `SMOKE_TEST = True` for the rest — executes the seven-notebook chain against a two-state slice, and field-compares the regenerated `*_smoke.json` artifacts to the versions committed at `git HEAD`. The `.ipynb` files on disk are never modified.
- **Full check** (`python reproduce_check.py --full`, ~7–16 h) reruns the real pipeline unpatched and compares the tracked full-scale JSONs plus the four tracked CSVs (hazard ratios, AFT coefficients, per-state C-index, feature importance).
- `--run-only` regenerates artifacts without comparing (used to refresh references before committing); `--compare-only`, `--only n4,n6`, and `--tol` are also available. Exit code 0 = every compared artifact matched.

The pipeline is deterministic on a single machine (fixed `random_state` / seeded RNGs; stored values are rounded), so exact comparison is the default. It requires the gitignored raw NBI + climate inputs (see **Data**) to be present locally; hosted CI cannot run it because those inputs are not distributed.

## References
- Lu, M., & Guler, S. I. (2022). Comparison of Random Survival Forest with Accelerated Failure Time-Weibull Model for Bridge Deck Deterioration. *Transportation Research Record*, 2676(8). https://doi.org/10.1177/03611981221078281
- Federal Highway Administration. *National Bridge Inventory* (1992–2025 delimited ASCII files). U.S. Department of Transportation. https://www.fhwa.dot.gov/bridge/nbi.cfm
- Thornton, M. M., et al. (2022). *Daymet: Daily Surface Weather Data on a 1-km Grid for North America, Version 4 R1.* ORNL DAAC. https://doi.org/10.3334/ORNLDAAC/2129 (retrieved via `pydaymet`)
- Pölsterl, S. (2020). scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn. *Journal of Machine Learning Research*, 21(212), 1–6.
- Davidson-Pilon, C. (2019). lifelines: survival analysis in Python. *Journal of Open Source Software*, 4(40), 1317. https://doi.org/10.21105/joss.01317
- Natural Earth. *10m Coastline* vector dataset (used for coastal distances via geopandas). https://www.naturalearthdata.com/