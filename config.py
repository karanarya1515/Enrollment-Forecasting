# =============================================================================
# config.py
# enrollment_forecasting_v2 — Central Configuration
# =============================================================================
# All paths, feature settings, and model parameters live here.
# No database connection logic — data is exported from VSCode as CSV.
# =============================================================================

from pathlib import Path

# ---------------------------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"        # CSVs exported from VSCode go here
MODEL_DIR    = PROJECT_ROOT / "models"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
LOG_DIR      = PROJECT_ROOT / "logs"
SQL_DIR      = PROJECT_ROOT / "sql" / "features"

# Final feature dataset (output of build_feature_dataset.py)
FEATURE_DATASET_PATH = DATA_DIR / "quarterly_feature_dataset.parquet"
FEATURE_DATASET_CSV  = DATA_DIR / "quarterly_feature_dataset.csv"

# Raw CSV files (exported from VSCode — one per SQL query)
RAW_CSV = {
    "enrollments":   RAW_DIR / "q1_enrollments.csv",
    "screenings":    RAW_DIR / "q2_screenings.csv",
    "site_metadata": RAW_DIR / "q3_site_metadata.csv",

    "kits":          RAW_DIR / "q5_kits.csv",
    "shipments":     RAW_DIR / "q6_shipments.csv",
    "site_country":  RAW_DIR / "q7_site_country.csv",   # ALL sponsors — site country + region
    "supply_spine":  RAW_DIR / "q8_supply_spine.csv",    # supply-active site-quarters with no patient activity
}

# Auto-create all directories on import
for _dir in [DATA_DIR, RAW_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR, SQL_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# FEATURE ENGINEERING SETTINGS
# ---------------------------------------------------------------------------
FEATURE_CONFIG = {
    # Lag windows for enrollment and screening (in quarters)
    "lag_windows":          [1, 2, 4],

    # Rolling average windows (in quarters)
    "rolling_windows":      [2, 4],

    # Minimum year to include (filter out very old data)
    "start_year":           2018,

    # Screening bins for nonlinear ordinal encoding
    "screening_bins":       [0, 1, 5, 10, 25, 50, float("inf")],
    "screening_bin_labels": ["zero", "very_low", "low", "medium", "high", "very_high"],
}

# ---------------------------------------------------------------------------
# MODEL SETTINGS
# ---------------------------------------------------------------------------
MODEL_CONFIG = {
    # Stage 1: XGBoost Classifier — predicts zero vs non-zero enrollment
    "classifier": {
        "objective":        "binary:logistic",
        "n_estimators":     500,
        "max_depth":        6,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1,   # Set dynamically at train time from class ratio
        "random_state":     42,
        "n_jobs":           -1,
    },

    # Stage 2: XGBoost Regressor — predicts enrollment count (Poisson)
    "regressor": {
        "objective":        "count:poisson",
        "n_estimators":     500,
        "max_depth":        6,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "random_state":     42,
        "n_jobs":           -1,
    },

    # Time-aware walk-forward cross-validation
    "cv_n_splits":             5,
    "test_size_quarters":      4,

    # Uncertainty quantification
    "monte_carlo_simulations": 1000,
    "quantiles":               [0.10, 0.50, 0.90],  # P10, P50, P90
}

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
LOG_CONFIG = {
    "level":    "INFO",
    "format":   "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    "datefmt":  "%Y-%m-%d %H:%M:%S",
    "log_file": LOG_DIR / "pipeline.log",
}