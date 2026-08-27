# Incremental prognostic information from procedure-wide multimodal physiology

This repository accompanies a retrospective, fixed-predictor evaluation of
whether procedure-wide multimodal physiological summaries add prognostic
information beyond age and ASA for in-hospital death or postoperative ICU stay
longer than 3 days in a selected VitalDB surgical cohort.

## Repository contents

- `reproduce_evaluation.py`: leakage-controlled case-level or patient-grouped
  cross-validation using an authorised local analytic table. The script writes
  aggregate results only.
- `aggregate_results.json`: machine-readable aggregate model performance,
  uncertainty intervals, sensitivity analyses, cohort-flow counts, and
  privacy-preserving figure source data.
- `render_figures.py`: redraws discrimination and calibration figures solely
  from `aggregate_results.json`.
- `requirements.txt`: tested Python package versions.
- `LICENSE`: MIT license for code.
- `LICENSE-DATA`: CC BY 4.0 license for the aggregate non-code material.

## Data availability and privacy boundary

VitalDB version 1.0.0 is publicly available through PhysioNet
([DOI: 10.13026/czw8-9p62](https://doi.org/10.13026/czw8-9p62)). Access and use
remain subject to the terms of the source repository. MIMIC-IV version 3.1 is
available only to credentialed users through PhysioNet under its data-use
agreement. MIMIC-IV data are not included here.

No participant-level source data, identifiers, individual outcomes, fold
assignments, individual predictions, or patient-level derivatives are included
in this repository. The repository also excludes institutional clinical data,
restricted partner data, and historical feature-selection data that cannot be
redistributed or fully reconstructed.

`aggregate_results.json` contains only aggregate counts and metrics,
interpolated ROC/precision-recall coordinates, and calibration groups of
approximately 128 cases per group. It is not a participant-level dataset.

## Fixed evaluation specification

The primary PFS specification contains five predictors:

1. age;
2. ASA physical status;
3. heart-rate-variability SDNN;
4. successive mean arterial pressure variability; and
5. a neurocardiac coupling index.

The primary comparator contains age and ASA. All models use the same 10
stratified folds. Deterministic iterative imputation, standardisation, and
unweighted logistic-regression fitting occur within each training fold.
Patient-grouped cross-validation is provided as a supportive sensitivity
analysis.

### Important caveat

This is an internally cross-validated evaluation of incremental predictive
information, not an externally validated clinical tool. The physiological
features summarise the available procedure recording, so the intended research
use is procedure-end/postoperative risk stratification rather than preoperative
prediction or real-time warning. The cohort was selected on a positive recorded
intraoperative fentanyl value, which narrows the target population and may
introduce selection bias. The historical feature-selection chain was not nested
within the current cross-validation and cannot be fully reproduced. Therefore,
no deployable final probability equation is claimed; external validation and
local recalibration are required before any clinical use.

## Reproducing the aggregate evaluation

Create an environment and install the tested dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

With an authorised local analytic CSV containing the variables named in the
script:

```bash
python reproduce_evaluation.py \
  --input /path/to/authorised_analytic_table.csv \
  --output aggregate_evaluation.json
```

For the supportive patient-grouped analysis, the input must also contain a
`subjectid` column:

```bash
python reproduce_evaluation.py \
  --input /path/to/authorised_analytic_table.csv \
  --output aggregate_grouped_evaluation.json \
  --patient-grouped
```

The input path is supplied by the user; the script contains no local absolute
paths and never writes row-level predictions.

## Redrawing figures

```bash
python render_figures.py \
  --input aggregate_results.json \
  --output rendered_figures
```

This produces PDF and PNG discrimination and calibration figures without
accessing participant-level data.

## Licenses

The Python code is released under the MIT License. The aggregate result and
figure-source material in `aggregate_results.json` is released under CC BY 4.0.
Neither license changes the access conditions of VitalDB, MIMIC-IV, or any
other source data.

## Citation

Please cite the accompanying article after publication. Citation and DOI details
will be added when available.
