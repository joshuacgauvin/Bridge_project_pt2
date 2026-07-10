# Random Survival Forests for Bridge Deterioration Prediction


### Predicting time-to-poor of bridges listed in the National Bridge Inventory (NBI) benchmarked against Cox Survival models.


## Motivation
Infrastructure maintenance can either be reactive, after a fault or issue occurs, or proactive, fixing small issues before the infrastructure has a large fault. The latter has been often shown to be far less costly in terms of both resources required for the proactive maintenance, and the impact the maintenance has on traffic in the context of bridge and roadway repairs.
<br>
The question concerning proactive maintenance, however, is when to conduct it. Maintenance too frequent can be inefficient and costly, diminishing the reason for preferring a proactive maintenance schedule over a reactive one. On the other hand, infrequent maintenance can also lead to inefficiencies where bridges aren't properly maintained as often as they should be. Finding the correct bridge schedule is also not a one-bridge-fits-all differing structures, lengths, traffic, weather, or exposure to sea salt can each affect a bridge's lifespan in different, nonlinear ways. For this reason, a tailored algorithm can be used to predict bridge deterioration.
<br>
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