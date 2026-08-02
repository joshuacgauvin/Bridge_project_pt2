  # Survival Models for Bridge Deterioration Prediction


### Comparing a Random Survival Forest against Cox PH and Weibull AFT on ~800k non-culvert NBI bridges and quantifying what climate data adds.


## Motivation
Infrastructure maintenance can either be reactive, after a fault or issue occurs, or proactive, fixing small issues before the infrastructure has a large fault. The latter has often been shown to be far less costly in terms of both resources required for the proactive maintenance and the impact the maintenance has on traffic in the context of bridge and roadway repairs.

The question concerning proactive maintenance, however, is when to conduct it. Maintenance that is too frequent can be inefficient and costly, diminishing the reason for preferring a proactive maintenance schedule over a reactive one. On the other hand, infrequent maintenance can also lead to inefficiencies where bridges aren't properly maintained as often as they should be. Finding the correct maintenance schedule is also not a one-bridge-fits-all problem. Differing structures, lengths, traffic, weather, or exposure to sea salt can each affect a bridge's lifespan in different, nonlinear ways. For this reason, a tailored algorithm can be used to predict bridge deterioration.

I tested which models are best for predicting bridge deterioration at the national level using the Random Survival Forest, Cox Proportional Hazards, and Weibull AFT models.

I am also including climate data in the models to account for deterioration due to weather and exposure in addition to data on the physical bridges themselves. Ablating that environmental block improves every model tested — by +0.0140 C-Index for the Random Survival Forest and about +0.0006 for the parametric models nationally — so the climate features earn their place, though how much they buy depends heavily on the model family (see [Climate contribution](#climate-contribution-national-ablation)).

## Repository structure
| Folder | Role |
| --- | --- |
| [`Bridges_all_of_US/`](Bridges_all_of_US/) | **The main national pipeline** — data download/build (Notebooks 1–4), model training (5–6), risk map (7), ablations (8), plus the leakage ablation (`leakage_ablation_national.ipynb`), temporal validation (`temporal_validation_national.ipynb`), and bootstrap comparison (`bootstrap_model_comparison.ipynb`) |
| [`Lu+Guler_comparison/`](Lu%2BGuler_comparison/) | Self-contained Pennsylvania case study: a Lu & Guler (2022)-style deck analysis + the climate-contribution measurement |
| [`ma_bridges/`](ma_bridges/) | **Superseded** Massachusetts prototype the national pipeline was ported from — kept for lineage; not part of the current results (see its [README](ma_bridges/README.md)) |

## Data
- National Bridge Inventory (1992 - 2025): Bridge data. **Culverts (NBI item 43B code 19; 19.8% of structures) are excluded**: under NBI convention they report the deck/superstructure/substructure condition ratings the event is built from as "N" (not applicable — 99.3% of culverts), so the deterioration event is undefined for them (see the leakage audit below). The same all-"N" rule is applied structure-wide, removing a further 545 non-culverts, and one more structure is dropped for an impossible year built (recorded as 19): 1,001,296 − 198,454 − 545 − 1 = **802,296**.

- **Identifier cleaning.** States periodically renumber their NBI `STRUCTURE_NUMBER_008`, splitting one bridge's panel across two or more identifiers. FHWA's [LTBP InfoBridge](https://infobridge.fhwa.dot.gov/data) supplies a *Latest Year NBI Structure Number* for this; the ASCII distribution used here does not, so `us_record_split_chains.csv` reconstructs the mapping from the raw files — linking panels that are disjoint and adjacent in time, share a modal `YEAR_BUILT_027`, and sit within 150 m — and drops rows whose year built disagrees with their panel's modal value. This folds 17.0% of structure numbers into longer panels: 973,905 → **802,296** modelled bridges, event rate 29.3% → **31.8%**. Chains were validated against seven structural attributes the matcher never saw: members of a detected chain agree at 0.86–0.97, against 0.34–0.43 for a same-state / same-build-year null. Matching is deliberately conservative, so that share is a lower bound. Set `STITCH_RECORD_SPLITS = False` in `build_national_rsf_dataset.ipynb` to reproduce the pre-cleaning build.


- **Left-censoring.** 44.1% of events (112,514 of 254,872) are recorded at the bridge's first appearance in the panel — 86,126 of them at the 1992 start — so for those bridges the survival time is an upper bound on the true time to first poor condition. Not corrected here; the alternatives are a left-censoring-aware likelihood or dropping 44% of the events.


- Daymet v4 R1 (a **1-km** daily surface weather product), retrieved via `pydaymet`: bridges are snapped to a 0.05° grid (~5.5 km) and one daily series is pulled per occupied cell — 159,014 cells — then aggregated into 17 annual features, so bridges sharing a cell share a weather series. Bridges whose surveys carry no coordinates (79,596; **9.9%**) are placed at their county centroid and flagged. Climate coverage is **98.7%** complete overall: 0.4% missing for measured locations against 9.6% for imputed ones.


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
All three models were trained on the same 802,296-bridge NBI dataset (first transition into "poor", structural + traffic + climate + coastal features; culverts and other structures without rateable deck/superstructure/substructure components excluded; renumbered structure records folded into single panels — see Data) and scored on the identical 160,460-bridge held-out test split.

|Model | C-Index | IBS | Mean AUC (25/50/75 yr) | Fit time|
| --- | --- | --- | --- | --- |
|Weibull AFT|**0.8834**|0.0592|0.9490|3.9 min|
|Cox Proportional Hazards|0.8830|**0.0587**|**0.9503**|47 s|
|Random Survival Forest|0.7671|0.0816|0.9277|33 min|

Uncertainty on the split (paired bootstrap over the shared test set, B = 1000, `bootstrap_model_comparison.ipynb` → `us_model_bootstrap.json`): AFT [0.8820, 0.8847], Cox [0.8816, 0.8843], RSF [0.7650, 0.7693]. The parametric-vs-RSF gap is decisive (Cox−RSF +0.1159, 95% CI [+0.1141, +0.1175], p < 0.002), while Cox and AFT are practically tied (AFT−Cox +0.0004, 95% CI [+0.0002, +0.0006]; statistically nonzero, practically nothing).

The parametric models outperform the RSF at national scale. However, the RSF's hyperparameters were deliberately coarsened to fit ~800k bridges in memory (`min_samples_leaf=50`, 10% per-tree subsampling); a **cost-matched capacity check** (`ablation_state_dummies.ipynb`, Arm D: MA-specialist hyperparameters — 300 trees, leaf 5, full bootstrap — scored on the full test split) reaches pooled C = 0.7895 ± 0.0022 on 7,526-row national subsamples (5 seeds) versus 0.7671 for the full coarsened forest. Pushing the same fine-grained forest up a **capacity curve** — 50k: 0.8062, 100k: 0.8096 ± 0.0014, 200k: 0.8143 (leaf 5 with a full bootstrap is memory-bound past ~200k, the same wall that forced the coarsening) — confirms the coarsening does understate the RSF, but the recovery is front-loaded: most of it arrives by 50k (+0.0167), and the two clean doublings that follow add only +0.0034 and +0.0047 — under ~0.005 each. The gap to Cox narrows from +0.116 (full coarse) to +0.094 (cost-matched) to +0.069 at the 200k ceiling, and stalls there: taking the largest per-doubling gain observed (+0.0047) as an optimistic rate, closing the remaining 0.0687 would need ~15 further doublings — on the order of five billion bridges, ~6,000x the national inventory. So the parametric advantage at national scale is a genuine capacity limit rather than primarily a hyperparameter artifact.

The Cox model additionally yields interpretable per-feature hazard ratios, 193 of 266 covariates are significant at raw p < 0.001 and because significance is cheap at N ≈ 642k training bridges, this count is robust to multiple-testing correction (222 survive a Benjamini–Hochberg FDR at q < 0.05; 185 even under the far stricter Bonferroni family-wise bound at α = 0.05; counts derived from `us_cox_hazard_ratios.csv`). Significance is therefore not the discriminating lens: only 153 of those carry hazard ratios more than 10% away from 1.0 (and 161 more than 5%). Among the material effects: orthotropic-deck designs (NBI 43B code 8) carry ~3.7x the baseline hazard, while epoxy-coated-reinforcement decks (item 108C code 1) carry ~0.39x. A Schoenfeld-residual check (`lifelines.proportional_hazard_test`, rank transform; `us_cox_ph_check.json`) — run on a stratified 100,001-row training subsample (31,807 events), because the full-train test is quadratic in event count — flags 37 of 266 covariates at p < 0.001 and 72 at p < 0.05, construction year is by far the largest offender, i.e. era effects are genuinely time-varying so Cox is reported as a discriminative benchmark and its hazard ratios read as time-averaged effects, not as a validated proportional-hazards structural model.

**Fit times** were aquired from a single run of each model. The specifications for the machine the models were ran on are detailed below:

- CPU: AMD Ryzen 7 7700X 8-Core Processor (4.50 GHz)

- RAM: 32 GB

- OS: Windows 11

**Leakage audit.** Three feature groups carry the largest era signal and are measured era-free to avoid fingerprinting the observation window: climate joins as per-cell **1992–2025 normals** (not per-year values), and both reconstruction and **traffic** covariates (ADT, lane counts, truck share) are read at each bridge's **panel-entry row**, not the kept row. The dataset keeps one row per bridge — the first at-poor observation for failures (median survey year 1997) but the last observation for censored bridges (median 2025) — so any covariate re-measured over time leaks censoring status if read at the kept row. Per-year climate joined at that row's survey year does this; the kept-row reconstruction year is truncated at the kept year (among reconstructed bridges, 0.83% of events vs 13.5% of censored bridges show a post-2010 reconstruction under the old definition, an impossible-by-construction asymmetry; re-anchoring to each bridge's panel-entry row narrows this to 0.34% (events) vs 1.54% (censored) — roughly 4x apart rather than 16x, not fully flat, and reported honestly as a residual, smaller leak rather than a closed issue; before/after: `us_reconstruction_measurement.json`); and kept-row ADT carries the same fingerprint (measurement year `YEAR_ADT_030` median 1992 for events vs 2018 for censored; kept-row ADT median 235 vs 950; `us_traffic_leakage_measurement.json`). Measured on the shared test split by refitting all three models on the leaky build (`leakage_ablation_national.ipynb` and `us_leakage_ablation.json`), the fully leaky construction (per-year climate + kept-row reconstruction + kept-row traffic, 23 columns swapped) inflates the C-Index by **at least +0.0192 (RSF), +0.0148 (Cox), and +0.0177 (AFT)**. These are lower bounds — the anchoring is not exhaustive: operational items re-rated at each inspection (operating rating 064, inventory rating 066, scour criticality 113 — ranked 4th, 7th and 9th of 255 in RSF permutation importance) are still read at the kept row and sit in *both* ablation arms, so whatever era signal they carry cannot register here. For the operating rating, the 18.6-point gap between event and censored means (32.2 against 50.7) is roughly half measurement era: standardising both groups to a common distribution of survey years shrinks it to 9.6 points, leaving about half as condition signal and about half as calendar. Part of the drift reflects changes in NBI rating and reporting practice rather than changes in the structures. Swapping **only** the four traffic columns isolates a near-zero effect — **+0.0007 (RSF), +0.0006 (Cox), +0.0007 (AFT)** — so despite the stark measurement-era gap, kept-row traffic barely moves discrimination (ADT is one feature among 266); the climate and reconstruction constructions carry essentially all of the inflation. The Pennsylvania case study measured the same mechanism at +0.085–0.123 for single-state tree models. All numbers reported here are from the corrected, era-free build. The same audit removed the earlier "culverts carry ~0.19x the baseline hazard" claim: 99.3% of culverts report all three event-defining condition ratings as "N" (event rate 0.14% vs 31.8% for non-culverts), so that coefficient was a labeling artifact of an event culverts could barely register, not a finding, therefore culverts are excluded from the study population (`us_culvert_diagnostic.json`).

Metrics: `us_rsf_metrics.json`, `us_parametric_metrics.json`, `us_model_bootstrap.json`, `us_ablation_state_dummies.json`, `us_leakage_ablation.json`, `us_culvert_diagnostic.json`, `us_reconstruction_measurement.json`, `us_traffic_leakage_measurement.json`, `us_cox_ph_check.json`, `us_temporal_validation.json`. Hazard ratios / AFT coefficients: `us_cox_hazard_ratios.csv`, `us_aft_coefficients.csv` (regenerable from `train_national_parametric.ipynb`).

### Predicting new bridges (temporal validation)
The headline split is random, so every construction era appears on both sides. To test the stricter deployment question, can a model trained on today's bridges rank *tomorrow's*? `temporal_validation_national.ipynb` trains on the 615,890 bridges built **before 2015** (204,070 events) and scores the 32,548 bridges built **in 2015 or later** (95 events, none seen in training). An in-era control (the standard test split restricted to pre-2015 bridges) reproduces the headline C-Index for all three models (RSF 0.7669, Cox 0.8835, AFT 0.8835), confirming the setup.

On the never-seen-era bridges all three models rank well above chance, and the random survival forest outperforms the parametric models: RSF **0.8283** (95% CI [0.778, 0.877]), Cox **0.7934** ([0.737, 0.845]), AFT **0.7963** ([0.743, 0.846]) (percentile bootstrap, B = 1000). With only 95 post-2015 events those intervals overlap heavily, so the 2015 cutoff alone does not resolve the ordering.

**A 2010 cutoff (53,665 bridges, 300 events) resolves it, for all three models:**

|Model | 2015 cutoff (95 events) | 2010 cutoff (300 events) | in-era control (2010) |
| --- | --- | --- | --- |
|Random Survival Forest|0.8283 [0.778, 0.877]|**0.8539** [0.8294, 0.8781]|0.7673|
|Cox Proportional Hazards|0.7934 [0.737, 0.845]|0.7800 [0.7493, 0.8107]|0.8838|
|Weibull AFT|0.7963 [0.743, 0.846]|0.7868 [0.7570, 0.8155]|0.8835|

At 300 events the intervals **separate** — the RSF's lower bound (0.8294) sits above the AFT's upper bound (0.8155). All three models score the identical 53,665 rows, so non-overlapping marginal intervals is a conservative criterion here and a paired test would only be more decisive. The movement is symmetric: going from their own era to an unseen one, Cox and AFT each give up ~0.10 C-Index, while the RSF *gains* ~0.09 (0.7673 in-era → 0.8539 new-era). The forest ranks bridges it has never seen better than the era it trained on, at both cutoffs. The likely reason is that these are different ranking problems: the post-cutoff cohort is young and narrow in age, which strips out most of the variance in `YEAR_BUILT` — the covariate the parametric models lean on hardest and the worst offender in the Schoenfeld check — leaving the structural and environmental variation where the forest's interaction modelling pays off. Caveats: the observable horizon for new bridges is ≤ 15 years at the 2010 cutoff and ≤ 10 at 2015 (medians 9 and 6), so this validates early-life risk ranking, not long-horizon prediction, and the RSF cannot extrapolate `YEAR_BUILT` past the training range, so its new-bridge ranking rests on the non-age features.

### Climate contribution (national ablation)
`ablation_state_dummies.ipynb` (Arm E) refits **all three models with and without the environmental block** — the 17 Daymet 1992–2025 cell normals plus `dist_to_coast_km`, 18 columns — at full shipped fidelity on the 641,836-bridge training split, scored on the identical 160,460-bridge test set. The geocoding-quality flag `location_imputed` stays in *both* arms (it records data provenance, not exposure). All three "with environment" arms reproduce the headline C-Indices above exactly, confirming the ablation is measured against the shipped models.

|Model | Without environment | With environment | Delta | 95% CI |
| --- | --- | --- | --- | --- |
|Random Survival Forest|0.7531|0.7671|**+0.0140**|[+0.0135, +0.0145]|
|Cox Proportional Hazards|0.8824|0.8830|+0.0006|[+0.0005, +0.0007]|
|Weibull AFT|0.8828|0.8834|+0.0006|[+0.0005, +0.0007]|

Paired bootstrap over the shared test split, B = 1000, seed 2026; all three intervals exclude zero at p < 0.002 (`us_ablation_state_dummies.json` → `arm_E_climate_contribution`). **Environmental data measurably improves prediction for every model tested** — but the size of the improvement splits sharply by model family. The raw ratio between the forest's gain and the parametric models' is over twenty to one, but that flatters the comparison, because the two start from very different baselines and so have very different room left to improve; normalising each gain by its own remaining headroom (1 − C) gives 5.7% of the forest's headroom against 0.5% of Cox's, a ratio nearer **eleven**. Quote the headroom-normalised figure, not the raw ratio.

Climate normals describe a smooth spatial field whose effect is plausibly nonlinear and interactive (freeze-thaw cycling matters more for some deck types, coastal salt for some protection systems), which a forest represents directly through splits and a single linear term per covariate cannot. The Cox coefficients corroborate this: per-unit hazard ratios for the climate features sit essentially at 1.0 (snow days 1.0021/day, freeze-thaw 1.0003/cycle) and `dist_to_coast_km` is not significant (p = 0.10) — even though it ranks 26th of 255 features in RSF permutation importance. Two caveats: the parametric null is **specification-dependent** (explicit climate-by-structure interaction terms could plausibly recover part of what the forest finds), and no size-matched non-environmental control block was run, so this does not isolate environmental data from feature removal in general. Dropping the 51 `STATE_*` indicators costs the forest nothing at all — pooled C is 0.0021 *higher* without them — against +0.0140 for the 18 environmental columns, which is suggestive but not a matched control (different `n_train`, coarser hyperparameters, no parametric counterpart).

### Single-state check and the join-method caveat (Pennsylvania)
The [Lu & Guler comparison study](Lu%2BGuler_comparison/pa_deck_climate_study.ipynb) re-runs the same measurement on a single state's decks, over a feature-matched structural baseline on PA state-owned bridges. Its main value is the join-method demonstration, not the climate estimate:

- Adding climate normals (1992–2025 Daymet cell means) + coastal distance improves the composite first-hit-poor C-Index by **+0.0117** (95% CI +0.0072 to +0.0161, p < 0.002) and the deck-only outcome by **+0.0065** (95% CI +0.0009 to +0.0125, p = 0.018) — paired bootstrap over the shared test split, B = 1000. The deck figure is marginal; the **national ablation above carries the environmental result**, and what this single-state check adds is the join-method caveat below.
- Caveat established along the way: joining climate by *observation year* inflates these deltas to +0.085–0.123 on first-hit outcomes, because the climate year leaks the observation era. Climate normals carry only the spatial exposure signal and give the honest estimate above.
- **Comparison with Lu & Guler (2022)** — a similar analysis on NBI-derived deck spells (not a replication: public NBI panel rather than their PennDOT records, and a static ADTT-at-spell-start × years traffic clock rather than their accumulated time-varying ADTT). All three published numbers, compared honestly: their traffic-clock RSF 0.836 vs 0.870 here and calendar-clock RSF 0.591 vs 0.683 here agree qualitatively; their traffic-clock AFT 0.693 vs **0.823 here diverges** — plausibly the static-rate clock, which a linear AFT can exploit as easily as a forest. The conclusion that survives both datasets is qualitative: **the clock (cumulative traffic vs calendar time), not the model family, drives the headline number.**

### Ablations (state indicators + RSF capacity)
Including the 51 `STATE_*` dummies now leans consistently toward dropping them (median per-state C-Index delta +0.0033 *without* them; 41 of 50 states better without) — the pooled difference of 0.0021 is still only about one bootstrap standard error, but the per-state sign count is no longer ambiguous the way it was before the record-split repair. They are kept because the magnitude remains negligible and geography also enters through the continuous climate and coastal features; dropping them would now be equally defensible. The same notebook benchmarks a single-state specialist: an RSF trained only on Massachusetts bridges (same features, same test rows) scores 0.8025 within MA versus the national model's 0.6544 — pooled training wins between-state comparisons, but a dedicated state model ranks better within its own state. Arm D (described above) bounds how much of the RSF-vs-parametric gap is explained by the memory-driven hyperparameter coarsening, and Arm E (the national climate ablation above) measures what the environmental block contributes to each of the three models. All five arms write to `us_ablation_state_dummies.json`.

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