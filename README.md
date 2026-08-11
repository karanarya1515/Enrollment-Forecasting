# Clinical Trial Enrollment Forecasting

### A Two-Stage Zero-Inflated Machine Learning Platform for Quarterly Patient Enrollment Prediction

**MSc Artificial Intelligence — Capstone Project (Capstone 1)**
Karan Arya | Batch 10 | Industry host: Endpoint Clinical

---

## Overview

Forecasting how many patients a clinical trial site will enrol in an upcoming quarter is a
persistently difficult operational problem: the majority of site-quarters produce no enrolments
at all, while a small minority of high-performing sites produce very large counts. A single
regression model trained on this distribution is pulled toward zero and systematically
underestimates the sites that matter most. This project addresses that structure directly with a
**two-stage zero-inflated architecture** — an XGBoost classifier first predicts *whether* a site
will enrol at all in a given quarter, and an XGBoost Poisson regressor then predicts *how many*
patients, conditional on the site being active. The two predictions are multiplied to produce the
final forecast. On top of the point forecast, the platform quantifies uncertainty through two
independent methods (Monte Carlo simulation and native quantile regression), producing P10/P50/P90
prediction bands that are surfaced in an interactive Streamlit dashboard with drill-down from
sponsor to study to country to region. The model is trained on site-quarter records spanning
Q1 2018 – Q4 2023 and evaluated on a strictly held-out 2024 test period.

---

## ⚠️ Data availability and confidentiality

**The dataset is deliberately not included in this repository.**

This project was developed against operational clinical trial data belonging to Endpoint Clinical
and its sponsor clients. That data is commercially confidential and subject to patient-privacy
obligations, so **no raw extracts, engineered datasets, model artifacts, or logs are published
here** — these are excluded via `.gitignore`. For the same reason, the SQL queries used to build
the extracts are also withheld, as they document a proprietary internal database schema; their
purpose and output columns are instead described in prose in
[Expected data schema](#expected-data-schema) below.

**What this means for a reader:** the pipeline cannot be executed end-to-end without supplying
your own data. Everything needed to *understand and evaluate the work* is present — all
feature-engineering logic, model architecture, uncertainty quantification, evaluation code, and
the dashboard. The schema section below documents exactly what each input file must contain, so
the pipeline is fully reproducible against any equivalently-shaped dataset.

---

## Pipeline overview

The project runs as four sequential stages. Each stage writes its output to disk and the next
stage reads it, so stages can be re-run independently.

```
   SQL extracts (run separately, exported as CSV)
                 │
                 ▼
        data/raw/*.csv
                 │
                 ▼
┌────────────────────────────────────┐
│  1. build_feature_dataset.py       │   Merge 7 raw extracts onto a site-quarter
│     Feature engineering            │   spine; build 40 predictive features
└────────────────────────────────────┘
                 │  data/quarterly_feature_dataset.parquet
                 ▼
┌────────────────────────────────────┐
│  2. train_model.py                 │   Stage 1: XGBoost classifier (zero vs non-zero)
│     Two-stage model training       │   Stage 2: XGBoost Poisson regressor (count)
└────────────────────────────────────┘
                 │  models/*.json  +  data/test_predictions_2024.csv
                 ▼
┌────────────────────────────────────┐
│  3. uncertainty.py                 │   Method A: Monte Carlo (1,000 runs)
│     P10 / P50 / P90 bands          │   Method B: XGBoost quantile regression
└────────────────────────────────────┘
                 │  data/uncertainty_predictions.parquet
                 ▼
┌────────────────────────────────────┐
│  4. dashboard.py                   │   Streamlit UI: fan chart, cumulative chart,
│     Interactive visualisation      │   KPI cards, hierarchical drill-down
└────────────────────────────────────┘
```

`config.py` holds shared paths and feature/model settings for stage 1. Stages 2–4 define their own
constants locally so they can be run standalone.

---

## Setup

**Requirements:** Python 3.10 or newer. The project is platform-independent but was developed on
Windows; the logging in `train_model.py` and `uncertainty.py` explicitly handles Windows console
encoding.

```bash
# 1. Clone the repository
git clone https://github.com/karanarya1515/Enrollment-Forecasting.git
cd Enrollment-Forecasting

# 2. Create and activate a virtual environment
python -m venv venv

#    Windows (PowerShell)
venv\Scripts\Activate.ps1
#    macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Dependencies:** `pandas`, `numpy`, `pyarrow` (parquet I/O), `xgboost` (≥2.0 — required for the
`reg:quantileerror` objective used in stage 3), `scikit-learn` (evaluation metrics only),
`streamlit` and `plotly` (dashboard). No database driver is needed: the pipeline reads CSV files
from disk and never opens a database connection.

**Before running:** place the seven input CSVs in `data/raw/` using the filenames listed in
[Expected data schema](#expected-data-schema). `config.py` creates the required directories
automatically on import.

---

## Running the pipeline

Run the stages in order. Stage 2 requires stage 1's output, stage 3 requires stage 2's, and stage 4
requires stage 3's.

### Stage 1 — Build the feature dataset

```bash
python build_feature_dataset.py
```

Reads the raw CSVs, constructs a site-quarter spine, merges all feature tables onto it, and
engineers 40 features. Writes `data/quarterly_feature_dataset.parquet` (and a `.csv` backup) and
prints a dataset summary with row counts, zero-enrollment proportion, and coverage statistics.

### Stage 2 — Train the two-stage model

```bash
python train_model.py
```

Performs a time-aware split (train Q1 2018 – Q4 2023, test 2024), trains both stages using a
two-phase early-stopping procedure that prevents the test set leaking into the stopping decision,
evaluates each stage, and reports feature importance. Writes model artifacts to `models/`,
predictions to `data/test_predictions_2024.csv`, and a timestamped log to `logs/`.

### Stage 3 — Quantify uncertainty

```bash
python uncertainty.py
```

Runs 1,000 Monte Carlo simulations with calibrated Gaussian noise injected into continuous
features, trains three native quantile-regression models (α = 0.10, 0.50, 0.90), and ensembles the
two methods. Writes `data/uncertainty_predictions.parquet` and reports interval coverage.
This is the slowest stage — expect several minutes.

### Stage 4 — Launch the dashboard

```bash
streamlit run dashboard.py
```

Opens in a browser at `http://localhost:8501`. Provides the P10/P50/P90 fan chart, cumulative
enrollment chart, KPI cards, and site-level detail, filterable by sponsor, study, country, and
region.

---

## Expected data schema

The pipeline consumes seven CSV extracts placed in `data/raw/`. All are aggregated to the
**site-quarter** grain and joined on the composite key `ETLDatabase` + `SITE_ID` + `Year` +
`Quarter`. `ETLDatabase` is a per-study database identifier whose prefix also encodes the sponsor;
`SITE_ID` is a site-level surrogate key. No patient-level identifiers are used anywhere in the
pipeline — every input is a pre-aggregated count.

| File | Purpose | Columns |
|---|---|---|
| `q1_enrollments.csv` | **Target variable.** Randomisation events aggregated per site-quarter. | `ETLDatabase`, `SITE_ID`, `EnrollYear`, `EnrollQuarter`, `EnrolledCount` |
| `q2_screenings.csv` | Screening and screen-failure counts per site-quarter. | `ETLDatabase`, `SITE_ID`, `ScrnYear`, `ScrnQuarter`, `ScreenedCount`, `ScreenFailCount` |
| `q3_site_metadata.csv` | Static site attributes and enrollment caps. | `ETLDatabase`, `SITE_ID`, `SITE_NUMBER`, `COUNTRY_ID`, `FirstActivationDate`, `EnrollmentLimit`, `ScreeningLimit`, `EnrollmentOpen`, `ScreeningOpen` |
| `q5_kits.csv` | Drug-kit supply activity per site-quarter. | `ETLDatabase`, `SITE_ID`, `KitYear`, `KitQuarter`, `KitsReceived`, `UniqueKitTypes`, `KitsDispensed` |
| `q6_shipments.csv` | Shipment activity and lead times per site-quarter. | `ETLDatabase`, `SITE_ID`, `ShipYear`, `ShipQuarter`, `ShipmentsReceived`, `AvgShipmentLeadDays` |
| `q7_site_country.csv` | Site-to-country/region mapping (one row per site). | `ETLDatabase`, `SITE_ID`, `COUNTRY_ID`, `COUNTRY_NAME`, `CountryISO`, `REGION_NAME` |
| `q8_supply_spine.csv` | Site-quarters with supply activity but no patient activity — captures pre-enrollment readiness as genuine zero rows. | `ETLDatabase`, `SITE_ID`, `Year`, `Quarter` |

**Notes on the inputs**

- Only `q1_enrollments.csv` is strictly required; the remaining files are optional and the
  corresponding features degrade gracefully to zero if absent.
- `EnrollmentLimit` / `ScreeningLimit` use `9999` as a "no limit set" sentinel, which is converted
  to null on load.
- `q8_supply_spine.csv` is important to the target distribution: without it, sites that were
  supply-active but never enrolled are absent entirely, which biases the zero class.
- All extracts are filtered to `Year >= 2018` at source.

### Engineered output

`build_feature_dataset.py` produces `data/quarterly_feature_dataset.parquet`: one row per
site-quarter with identifier columns (`ETLDatabase`, `SITE_ID`, `Year`, `Quarter`, `TimeIndex`,
`Sponsor`, `COUNTRY_NAME`, `REGION_NAME`), two targets, and 40 `Feat_*` predictors:

| Feature group | Count | Examples |
|---|---|---|
| Screening | 5 | `Feat_ScreensThisQuarter`, `Feat_ScreenFailRate`, `Feat_ScreenToEnrollRate` |
| Supply chain | 7 | `Feat_KitsReceivedThisQuarter`, `Feat_AvgShipmentLeadDays`, `Feat_HasKitSupply` |
| Time-series | 12 | `Feat_EnrollLag1Q/2Q/4Q`, `Feat_EnrollMA2Q/4Q`, `Feat_EnrollTrend2Q`, `Feat_EnrollVolatility4Q`, `Feat_PrevQuarterWasZero` |
| Site maturity | 2 | `Feat_SiteMaturityCategory`, `QuartersSinceActivation` |
| Hierarchical peer | 6 | `Feat_CountryPeerEnrollAvg`, `Feat_RegionPeerEnrollAvg`, `Feat_StudyPeerEnrollAvg` |
| Engineered / interaction | 8 | `Feat_LogScreens`, `Feat_ScreenXConversion`, `Feat_ScreenXMaturity` |

Targets: `Target_QuarterlyEnrollment` (integer count) and `Target_HasEnrollment` (binary).
`TimeIndex` is a monotonic quarter counter (`(Year − 2018) × 4 + Quarter`) used for time-aware
splitting.

---

## Methodology

**Two-stage zero-inflated architecture.** A large proportion of site-quarter rows have zero
enrollment (~33% of the training window), and modelling this with a single regressor collapses
predictions toward zero. Stage 1 is a binary XGBoost classifier with `scale_pos_weight` set from
the observed class ratio. Stage 2 is an XGBoost regressor with a `count:poisson` objective, trained
**only on non-zero rows** so it learns enrollment magnitude rather than the zero/non-zero decision
that stage 1 has already made. The final prediction is the product of the two.

**Preventing leakage.** The split is strictly temporal, never random: training uses TimeIndex 1–24
(Q1 2018 – Q4 2023) and testing uses TimeIndex 25–28 (2024), asserted disjoint at runtime. Early
stopping uses a two-phase procedure — the model first fits on TimeIndex 1–20 and early-stops
against a 2023 validation slice (21–24) to find the best iteration count, then refits on the full
1–24 window with that count fixed. This prevents the test set from influencing the stopping
decision. All lag, rolling-mean, and peer-average features are computed with `.shift(1)` or
greater, so no feature contains information from the quarter being predicted.

**Outlier handling.** The stage 2 target is capped at the 99th percentile (27 patients/quarter)
before training. This stabilises the Poisson fit against a handful of very-high-volume sites, at
the acknowledged cost of underestimating those sites.

**Uncertainty quantification.** Two independent methods, then ensembled by averaging:
*Method A (Monte Carlo)* injects Gaussian noise scaled to 8% of each continuous feature's standard
deviation and runs the full two-stage pipeline 1,000 times, taking percentiles of the resulting
distribution; binary flags are excluded from noise injection since flipping them represents a
different scenario rather than measurement noise. *Method B (quantile regression)* trains three
XGBoost models on the native `reg:quantileerror` objective targeting the 10th, 50th, and 90th
percentiles directly. Monotonicity (P10 ≤ P50 ≤ P90) is enforced after ensembling.

---

## Results

Evaluated on the held-out 2024 test period (61,072 site-quarter rows, never seen during training).

**Stage 1 — Classifier (will this site enrol?)**

| Metric | Score |
|---|---|
| F1 | 0.957 |
| Precision | 0.950 |
| Recall | 0.963 |
| Accuracy | 0.942 |
| AUC-ROC | 0.982 |

**Stage 2 — Poisson regressor (how many patients?), non-zero rows**

| Metric | Capped target | Raw target |
|---|---|---|
| MAE | 0.665 | 1.107 |
| RMSE | 1.486 | 12.975 |
| R² | 0.851 | 0.150 |

The gap between the capped and raw columns is the single most important caveat in these results.
The capped figures are an apples-to-apples measure against what the model was actually trained on;
the raw figures include the uncapped outlier sites, whose errors dominate RMSE and collapse R².
Both are reported rather than only the favourable one.

**Combined** (all test rows, zeros included): MAE 0.737.

---

## Repository structure

```
Enrollment-Forecasting/
├── config.py                    # Shared paths, feature and model settings
├── build_feature_dataset.py     # Stage 1 — feature engineering
├── train_model.py               # Stage 2 — two-stage model training
├── uncertainty.py               # Stage 3 — P10/P50/P90 quantification
├── dashboard.py                 # Stage 4 — Streamlit dashboard
├── requirements.txt
├── .gitignore
└── README.md

Excluded from version control (see .gitignore):
├── data/                        # Confidential — raw extracts and engineered dataset
├── sql/                         # Excluded — proprietary source schema
├── models/                      # Trained artifacts, regenerable via train_model.py
├── outputs/                     # Generated reports
└── logs/                        # Run logs
```

---

## Known limitations

- **High-volume sites are underestimated by design.** The 99th-percentile cap on the stage 2
  training target means sites enrolling more than 27 patients per quarter are systematically
  under-forecast. This is a deliberate bias-variance trade-off, not an oversight.
- **Prediction bands are narrow at low volumes.** Median P10–P90 width is 1 patient for
  low-enrollment sites, and interval coverage degrades above roughly 10 patients per quarter.
- **Incomplete geographic coverage.** Around 14% of site-quarter rows have no country or region
  mapping. These are filled as `"Unknown"` and retained in sponsor- and study-level views, but are
  necessarily excluded from country and region drill-downs.
- **Data completeness in the final quarter.** Q4 2024 contains fewer reporting sites than earlier
  quarters, which depresses the aggregate forecast in that quarter. This is an artefact of the
  extract, not model degradation, and is annotated directly on the dashboard chart.
- **Scope.** This submission (Capstone 1) implements the enrollment forecast view. Additional
  dashboard tabs and forward-looking (rather than backtested) forecasting are planned for
  Capstone 2.

---

## Academic integrity note

All code in this repository is my own work, produced for the MSc Artificial Intelligence capstone.
The underlying data is the property of Endpoint Clinical and is not distributed here. Results
reported above were generated from that data and are reproducible by the project supervisors on
request through the appropriate internal channels.
