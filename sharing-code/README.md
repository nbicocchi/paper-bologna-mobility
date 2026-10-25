# Sharing and missing traffic sensors

Standalone analysis of Bologna, 2023–2024. All cohorts, features and model fits are
rebuilt directly from `../dataset/`. No other paper's code, results or intermediates
are imported. The only additional measurement source is the municipality's official
2023/2024 recording-completeness CSV, downloaded into `cache/sources/` with URLs and
SHA-256 hashes. Subsequent runs reuse these external source snapshots, not fitted results.

## Run

```bash
python -m venv sharing-code/.venv
sharing-code/.venv/bin/pip install -r sharing-code/requirements.txt
sharing-code/.venv/bin/python sharing-code/code/run.py
make -C sharing-text
```

Alternatively, `make -C sharing-text reproduce PYTHON=/absolute/path/to/python`
reruns the entire experiment pipeline before compiling the paper.

`run.py` always reruns preparation, feature construction, descriptive analysis,
all model fits, evaluation, exports and tests. It fails on errors. `results/manifest.json`
records raw input hashes, external source hashes, scripts, package versions and outputs.
`results/run-status.json` distinguishes an unfinished run from a completed one.

## Experiments

* **New location:** all counts at the test location are excluded. Bicycles use
  leave-one-location-out (both directions and colocated devices together); motor
  vehicles use five grouped 1 km spatial folds, 100 m connected groups and a 250 m
  train/test exclusion buffer. Fit 2023; test excluded locations in 2023 and 2024.
* **Known location / failed sensor:** fit all eligible locations' 2023 data; test
  the same locations in 2024, without any 2024 counts as inputs or refitting.
  Both the Context baseline and all sharing models receive two frozen historical
  covariates: log 2023 hourly mean by site/direction/day-night and by the same keys
  plus weekday (floor 0.01). Missing historical weekdays fall back to that site's
  regime profile; missing regimes fall back to its site/direction profile. These
  fallbacks also use 2023 only and their use is counted in the split audit.
  This is a pooled model with explicit local history,
  not a model forced to infer an already-known sensor's level from other places.

Both scenarios use the same 2024 observations and contemporaneous weather/sharing.
This evaluates reconstruction during sensor unavailability, not a forecast made
before 2024. Geometry uses completed rental endpoints, so long rentals can delay
availability of a departure-period chord; this is retrospective, not a real-time
latency experiment. The simulated outage lasts the full test year; real faults are not
injected into training. Counts are used only as training labels and test outcomes.

Within each learner/scenario compare `Context → Citywide → Endpoints → Geometry`.
Model settings are fixed, preprocessing uses training rows only. Uniform chord
expansion is an intentionally simple reference without site history in either scenario.
Report pooled WAPE, bias, median/macro site WAPE, local gains and paired 2,000-draw
site/block sensitivity intervals. New-location gain persistence is measured over
the two test years; it is not applicable to the one-year known-location test.

## Data conventions

Retain departures in source year, finite rental duration >=30 seconds and both
endpoints inside the analysis box. Bikes additionally require recognized type and
>=100 m displacement. Exact duplicates are removed before discarding identifiers;
identifiers are never saved. Bicycle counters retain the previous study's six
explicit exclusions, require valid paired-direction totals and complete periods,
screen >=48-hour zero runs and require >=60 usable dates in 2023. Motor devices
require >=180 usable dates and stable coordinates in 2023; every hour must have
official completeness 100%. Devices moving >100 m in 2024 are not scored as the
same position. Cohort selection never uses 2024 prediction error.

Prediction periods preserve source conventions: bike service date starting at
06:00, car calendar date; day 06–23. Years follow these period dates: the last
2023 bike night ends on 1 January 2024 at 06:00, before the first 2024 test period;
training and test hours do not overlap. Both endpoint sources assign returns to their
own timestamp. Chords are straight-line proxies, not observed routes.

Descriptive maps now use **matched valid hours and continuous 20–06 nights for
both services**, on the same 500 m grid, over cached OpenStreetMap tiles. Tiles are
downloaded only for the fixed figure viewport, without sending rental information;
URLs and hashes are recorded in `results/basemap-sources.json`. They supply
present-day cartographic context, not historical routes or prediction features. These are
not a pixel-identical reproduction of the earlier unmatched-date maps. New results
must replace, rather than be spliced into, the old paper's numerical claims.
