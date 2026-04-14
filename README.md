# Physiological Fragility Score (PFS)

An open-source multimodal intraoperative score for adverse perioperative outcomes — development, internal validation, and cross-cohort transportability.

## What this repository contains

This is the reproducibility repository for the manuscript *"Multimodal Intraoperative Physiological Signals Match ASA Classification in Predicting Adverse Perioperative Outcomes: Development, Internal Validation, and Cross-Cohort Transportability of an Open-Source Score"*.

- **`pfs_model_coefficients.json`** — Final 5-variable PFS model (age, ASA, HRV-SDNN, MAP successive variability, neurocardiac coupling index). Includes coefficients, standard errors, odds ratios with 95% CI, feature-standardisation parameters, and apparent + bootstrap optimism-corrected performance.
- **`scripts/render_main_figures.py`** — Reproduces the four main figures from a cohort CSV and results JSONs.

## Data availability

- **VitalDB** (development cohort, n = 1 027): openly available at [vitaldb.net](https://vitaldb.net/)
- **MIMIC-IV** (transferability cohort): openly available at [physionet.org](https://physionet.org/content/mimiciv/)
- **Chinese tertiary hospital cohort** (external validation, n = 20 064): not publicly distributable due to institutional data-sharing restrictions. Summary-level data are available from the corresponding author on reasonable request.
- **Cohort CSVs are intentionally not included in this repository.** Re-extract from VitalDB using the official `vitaldb` Python package following the cohort construction rules described in the manuscript.

## Using the score

Given standardised feature values, the linear predictor is:

```
lp = intercept
   + β_age × (age − μ_age) / σ_age
   + β_asa × (asa − μ_asa) / σ_asa
   + β_sdnn × (hrv_sdnn − μ_sdnn) / σ_sdnn
   + β_mapvar × (map_successive_var − μ_mapvar) / σ_mapvar
   + β_ncc × (ncc_index − μ_ncc) / σ_ncc

p = 1 / (1 + exp(−lp))
```

All coefficients and standardisation parameters are in `pfs_model_coefficients.json`.

**Any new-setting deployment must include target-population recalibration (Platt scaling or equivalent) before clinical use.**

## Reproducibility

- Python 3.12, scikit-learn, statsmodels, matplotlib, numpy, pandas — see `requirements.txt`
- Running `scripts/render_main_figures.py` reproduces the four main manuscript figures

## License

Code released under the MIT License. Model coefficients released under CC-BY 4.0.

## Citation

Will be updated at acceptance.
