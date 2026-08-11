"""
uncertainty.py -- Phase 3: Uncertainty Quantification
Clinical Trial Enrollment Forecasting Platform
Karan Arya | MSc AI Batch 10 | Endpoint Clinical

Two complementary approaches to P10/P50/P90 prediction bands:

  Method A -- Monte Carlo simulation (1,000 runs with noise injection)
    Injects calibrated Gaussian noise into features, runs the two-stage
    pipeline 1,000 times per row, takes percentiles of the resulting
    distribution. Fast, model-agnostic, interpretable.

  Method B -- XGBoost Quantile Regression (native quantile objective)
    Trains three separate XGBoost regressors directly targeting the
    10th, 50th, and 90th percentiles of the count distribution.
    More principled statistically; slower to train.

Output:
  data/uncertainty_predictions.parquet
    One row per test site-quarter with columns:
      P10, P50, P90 (Monte Carlo)
      P10_q, P50_q, P90_q (Quantile regression)
      Final_Pred (from Phase 2 two-stage model)
      plus all ID and target columns

  data/uncertainty_predictions.csv  (backup)

This file is consumed directly by dashboard.py (Phase 4).
"""

import io
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).parent
DATA_DIR     = ROOT / "data"
MODEL_DIR    = ROOT / "models"
LOG_DIR      = ROOT / "logs"

PARQUET_PATH     = DATA_DIR / "quarterly_feature_dataset.parquet"
PREDICTIONS_PATH = DATA_DIR / "test_predictions_2024.csv"
META_PATH        = MODEL_DIR / "feature_columns.json"
S1_MODEL_PATH    = MODEL_DIR / "stage1_classifier.json"
S2_MODEL_PATH    = MODEL_DIR / "stage2_regressor.json"

OUT_PARQUET = DATA_DIR / "uncertainty_predictions.parquet"
OUT_CSV     = DATA_DIR / "uncertainty_predictions.csv"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging -- ASCII only, UTF-8 handlers (same pattern as train_model.py)
# ---------------------------------------------------------------------------
log_path = LOG_DIR / f"uncertainty_{time.strftime('%Y%m%d_%H%M%S')}.log"

_fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
_file_handler = logging.FileHandler(log_path, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter(_fmt))

_stdout_stream = (
    io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    if hasattr(sys.stdout, "buffer") else sys.stdout
)
_stream_handler = logging.StreamHandler(_stdout_stream)
_stream_handler.setFormatter(logging.Formatter(_fmt))

logging.basicConfig(level=logging.INFO, handlers=[_stream_handler, _file_handler])
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_S1 = "Target_HasEnrollment"
TARGET_S2 = "Target_QuarterlyEnrollment"
ID_COLS   = ["ETLDatabase", "SITE_ID", "Year", "Quarter", "TimeIndex"]

TRAIN_CUTOFF = 24
TEST_MIN     = 25
TRAIN_CORE_MAX = 20
TRAIN_VAL_MAX  = 24

# Monte Carlo settings
MC_RUNS        = 1000
MC_NOISE_SCALE = 0.08   # 8% Gaussian noise on continuous features
RANDOM_SEED    = 42

# Quantile targets
QUANTILES = [0.10, 0.50, 0.90]


# ===========================================================================
# Load models and metadata
# ===========================================================================

def load_models_and_meta():
    log.info("Loading trained models and feature metadata...")

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feat_cols  = meta["feature_columns"]
    cap_value  = meta["stage2_cap_value"]
    log.info(f"Feature columns: {len(feat_cols)}  |  Stage 2 cap: {cap_value:.0f}")

    s1_model = xgb.XGBClassifier()
    s1_model.load_model(str(S1_MODEL_PATH))

    s2_model = xgb.XGBRegressor()
    s2_model.load_model(str(S2_MODEL_PATH))

    log.info("Models loaded successfully")
    return s1_model, s2_model, feat_cols, cap_value


def load_data():
    log.info("Loading full dataset and Phase 2 predictions...")
    df   = pd.read_parquet(PARQUET_PATH)
    pred = pd.read_csv(PREDICTIONS_PATH)
    # Align SITE_ID dtype — parquet preserves str, CSV may parse as int
    pred["SITE_ID"] = pred["SITE_ID"].astype(str)
    df["SITE_ID"]   = df["SITE_ID"].astype(str)
    log.info(f"Full dataset: {df.shape[0]:,} rows  |  Phase 2 predictions: {pred.shape[0]:,} rows")
    return df, pred


# ===========================================================================
# METHOD A -- Monte Carlo Simulation
# ===========================================================================

def identify_continuous_features(feat_cols: list) -> list:
    """
    Returns the subset of feature columns that are continuous (non-binary).
    Binary features (0/1 flags) should not have noise injected --
    flipping a binary flag is not meaningful noise, it's a different scenario.
    """
    binary_patterns = [
        "Feat_Has", "Feat_PrevQuarter", "Feat_SiteMaturity",
        "Feat_ScreeningBinOrdinal",
    ]
    continuous = []
    for col in feat_cols:
        is_binary = any(col.startswith(p) for p in binary_patterns)
        if not is_binary:
            continuous.append(col)
    return continuous


def run_monte_carlo(
    test: pd.DataFrame,
    s1_model: xgb.XGBClassifier,
    s2_model: xgb.XGBRegressor,
    feat_cols: list,
    cap_value: float,
    n_runs: int = MC_RUNS,
    noise_scale: float = MC_NOISE_SCALE,
) -> pd.DataFrame:
    """
    Monte Carlo uncertainty estimation.

    For each of n_runs iterations:
      1. Inject calibrated Gaussian noise into continuous features
         (noise proportional to each feature's std -- relative perturbation)
      2. Run two-stage prediction (Stage 1 classifier -> Stage 2 regressor)
      3. Store final predictions

    After all runs, take P10/P50/P90 of the prediction distribution per row.

    Noise injection rationale:
      Real forecasting uncertainty comes from:
        - Screening rate variability (sites don't enroll at constant rates)
        - Supply chain timing uncertainty
        - Lag feature imprecision (counts fluctuate quarter to quarter)
      8% noise scale is conservative -- it reflects natural quarter-to-quarter
      coefficient of variation observed in clinical trial enrollment data.
    """
    log.info("=" * 60)
    log.info("METHOD A: Monte Carlo Simulation")
    log.info(f"  Runs: {n_runs}  |  Noise scale: {noise_scale*100:.0f}%  |  Rows: {len(test):,}")
    log.info("=" * 60)

    X_test = test[feat_cols].fillna(0).values
    continuous_idx = [
        i for i, c in enumerate(feat_cols)
        if c in identify_continuous_features(feat_cols)
    ]

    # Feature standard deviations for proportional noise scaling
    feat_stds = np.std(X_test, axis=0)
    feat_stds = np.where(feat_stds == 0, 1.0, feat_stds)  # avoid div by zero

    log.info(f"Continuous features for noise injection: {len(continuous_idx)} / {len(feat_cols)}")

    rng = np.random.RandomState(RANDOM_SEED)
    all_preds = np.zeros((len(test), n_runs), dtype=np.float32)

    t0 = time.time()
    for run in range(n_runs):
        if (run + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (run + 1) * (n_runs - run - 1)
            log.info(f"  Run {run+1:>4d}/{n_runs}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

        # Inject noise only on continuous features
        X_noisy = X_test.copy()
        noise = rng.normal(0, noise_scale, size=(len(test), len(continuous_idx)))
        for j, idx in enumerate(continuous_idx):
            X_noisy[:, idx] = np.clip(
                X_noisy[:, idx] + noise[:, j] * feat_stds[idx],
                0, None   # counts and ratios cannot go negative
            )

        # Stage 1: classifier
        s1_prob = s1_model.predict_proba(X_noisy)[:, 1]
        s1_pred = (s1_prob >= 0.5).astype(int)

        # Stage 2: regressor on predicted non-zero rows only
        s2_pred = np.zeros(len(test), dtype=np.float32)
        nz_mask = s1_pred == 1
        if nz_mask.sum() > 0:
            s2_raw = s2_model.predict(X_noisy[nz_mask])
            s2_pred[nz_mask] = s2_raw

        # Combined
        final = (s1_pred * s2_pred).clip(0)
        all_preds[:, run] = final

    elapsed_total = time.time() - t0
    log.info(f"Monte Carlo complete in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")

    # Compute percentiles
    p10 = np.percentile(all_preds, 10, axis=1)
    p50 = np.percentile(all_preds, 50, axis=1)
    p90 = np.percentile(all_preds, 90, axis=1)

    mc_df = pd.DataFrame({
        "MC_P10": np.round(p10).astype(int),
        "MC_P50": np.round(p50).astype(int),
        "MC_P90": np.round(p90).astype(int),
        "MC_Width": np.round(p90 - p10).astype(int),  # interval width = uncertainty
    }, index=test.index)

    # Evaluation: how well does MC_P50 perform vs Phase 2 Final_Pred?
    y_true = test[TARGET_S2].values
    mae_p50 = mean_absolute_error(y_true, mc_df["MC_P50"])
    log.info(f"MC P50 MAE (all rows): {mae_p50:.4f}")

    # Coverage: what % of actuals fall within [P10, P90]?
    in_band = ((y_true >= p10) & (y_true <= p90)).mean()
    log.info(f"MC P10-P90 coverage: {100*in_band:.1f}%  (ideal ~80%)")

    # Median interval width by enrollment level
    test_copy = test[[TARGET_S2]].copy()
    test_copy["width"] = mc_df["MC_Width"].values
    log.info("  Median P10-P90 width by enrollment level:")
    for label, lo, hi in [("0 pts", 0, 0), ("1-3 pts", 1, 3),
                           ("4-10 pts", 4, 10), ("11+ pts", 11, 9999)]:
        mask = (test_copy[TARGET_S2] >= lo) & (test_copy[TARGET_S2] <= hi)
        if mask.sum() == 0:
            continue
        w = test_copy.loc[mask, "width"].median()
        log.info(f"    {label:<12s}  n={mask.sum():>6,}  median_width={w:.1f}")

    return mc_df


# ===========================================================================
# METHOD B -- XGBoost Quantile Regression
# ===========================================================================

def train_quantile_models(
    df: pd.DataFrame,
    feat_cols: list,
    cap_value: float,
) -> dict:
    """
    Train three XGBoost regressors using the native quantile objective,
    one each for P10, P50, P90.

    Trains only on non-zero rows (same as Stage 2 in train_model.py).
    Uses same time-aware two-phase approach: find best n_estimators on
    core+val split, then refit on full train.

    The quantile objective minimises pinball loss:
      L(q, y, yhat) = q*(y-yhat) if y >= yhat else (1-q)*(yhat-y)
    This directly targets the desired percentile of the distribution.
    """
    log.info("=" * 60)
    log.info("METHOD B: XGBoost Quantile Regression")
    log.info("=" * 60)

    # Same splits as train_model.py
    train_core = df[df["TimeIndex"] <= TRAIN_CORE_MAX]
    train_val  = df[(df["TimeIndex"] > TRAIN_CORE_MAX) & (df["TimeIndex"] <= TRAIN_VAL_MAX)]
    train_full = df[df["TimeIndex"] <= TRAIN_CUTOFF]

    # Non-zero rows only
    core_nz = train_core[train_core[TARGET_S1] == 1].copy()
    val_nz  = train_val[train_val[TARGET_S1] == 1].copy()
    full_nz = train_full[train_full[TARGET_S1] == 1].copy()

    log.info(f"Non-zero rows -- core: {len(core_nz):,}  val: {len(val_nz):,}  full: {len(full_nz):,}")

    # Cap target (same cap as Stage 2)
    for d in [core_nz, val_nz, full_nz]:
        d["target_capped"] = d[TARGET_S2].clip(upper=cap_value)

    quantile_models = {}

    for q in QUANTILES:
        q_label = f"P{int(q*100)}"
        log.info(f"\n  Training quantile model: {q_label} (alpha={q})")

        shared_params = dict(
            objective="reg:quantileerror",
            quantile_alpha=q,
            n_estimators=800,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=10,
            reg_alpha=0.1,
            random_state=42,
            eval_metric="quantile",
            early_stopping_rounds=40,
            verbosity=0,
            n_jobs=-1,
        )

        # Phase A: find best n_estimators
        model_a = xgb.XGBRegressor(**shared_params)
        model_a.fit(
            core_nz[feat_cols].fillna(0), core_nz["target_capped"],
            eval_set=[(val_nz[feat_cols].fillna(0), val_nz["target_capped"])],
            verbose=False,
        )
        best_n = model_a.best_iteration + 1
        log.info(f"    Phase A best iteration: {best_n}")

        # Phase B: refit on full train
        final_params = {**shared_params, "n_estimators": best_n, "early_stopping_rounds": None}
        model = xgb.XGBRegressor(**final_params)
        t0 = time.time()
        model.fit(
            full_nz[feat_cols].fillna(0), full_nz["target_capped"],
            verbose=False,
        )
        log.info(f"    Phase B complete in {time.time()-t0:.1f}s")

        quantile_models[q_label] = model

    return quantile_models


def predict_quantiles(
    test: pd.DataFrame,
    s1_model: xgb.XGBClassifier,
    quantile_models: dict,
    feat_cols: list,
) -> pd.DataFrame:
    """
    Apply quantile models to test set.
    Stage 1 classifier gates which rows go to quantile regressors.
    Rows where Stage 1 predicts zero get P10=P50=P90=0.
    """
    log.info("\n--- Quantile Regression Predictions ---")

    X_test  = test[feat_cols].fillna(0)
    s1_prob = s1_model.predict_proba(X_test)[:, 1]
    s1_pred = (s1_prob >= 0.5).astype(int)
    nz_mask = s1_pred == 1

    log.info(f"Stage 1 non-zero predictions: {nz_mask.sum():,} / {len(test):,}")

    results = {}
    for q_label, model in quantile_models.items():
        preds = np.zeros(len(test), dtype=np.float32)
        if nz_mask.sum() > 0:
            preds[nz_mask] = model.predict(X_test[nz_mask])
        preds = np.clip(preds, 0, None).round()
        results[f"Q_{q_label}"] = preds.astype(int)

    q_df = pd.DataFrame(results, index=test.index)

    # Ensure monotonicity: P10 <= P50 <= P90
    q_df["Q_P10"] = np.minimum(q_df["Q_P10"], q_df["Q_P50"])
    q_df["Q_P90"] = np.maximum(q_df["Q_P50"], q_df["Q_P90"])
    q_df["Q_Width"] = q_df["Q_P90"] - q_df["Q_P10"]

    # Evaluation
    y_true = test[TARGET_S2].values
    mae_p50 = mean_absolute_error(y_true, q_df["Q_P50"])
    in_band = ((y_true >= q_df["Q_P10"].values) & (y_true <= q_df["Q_P90"].values)).mean()

    log.info(f"Quantile P50 MAE (all rows): {mae_p50:.4f}")
    log.info(f"Quantile P10-P90 coverage:   {100*in_band:.1f}%  (ideal ~80%)")

    return q_df


# ===========================================================================
# Combine and save
# ===========================================================================

def build_final_output(
    test: pd.DataFrame,
    phase2_pred: pd.DataFrame,
    mc_df: pd.DataFrame,
    q_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge Phase 2 predictions, MC bands, and quantile bands into one output.
    This is the file consumed by dashboard.py.
    """
    log.info("\n--- Building final uncertainty output ---")

    # Base: ID cols + targets from test set
    out = test[ID_COLS + [TARGET_S1, TARGET_S2]].copy()

    # Phase 2 final prediction (join on ID cols)
    p2_cols = ID_COLS + ["Final_Pred", "Stage1_Pred", "Stage2_Pred"]
    out = out.merge(
        phase2_pred[p2_cols],
        on=ID_COLS, how="left"
    )

    # Add non-ID columns from test set that the dashboard will need
    dashboard_cols = [
        "COUNTRY_NAME", "REGION_NAME", "Sponsor",
        "COUNTRY_ID", "CountryISO",
    ]
    for col in dashboard_cols:
        if col in test.columns:
            out[col] = test[col].values

    # Monte Carlo bands
    out["MC_P10"]   = mc_df["MC_P10"].values
    out["MC_P50"]   = mc_df["MC_P50"].values
    out["MC_P90"]   = mc_df["MC_P90"].values
    out["MC_Width"] = mc_df["MC_Width"].values

    # Quantile regression bands
    out["Q_P10"]   = q_df["Q_P10"].values
    out["Q_P50"]   = q_df["Q_P50"].values
    out["Q_P90"]   = q_df["Q_P90"].values
    out["Q_Width"] = q_df["Q_Width"].values

    # Ensemble P10/P50/P90: average of both methods
    # This is what the dashboard uses by default -- more robust than either alone
    out["P10"] = ((out["MC_P10"] + out["Q_P10"]) / 2).round().astype(int)
    out["P50"] = ((out["MC_P50"] + out["Q_P50"]) / 2).round().astype(int)
    out["P90"] = ((out["MC_P90"] + out["Q_P90"]) / 2).round().astype(int)
    out["P10"] = np.minimum(out["P10"], out["P50"])  # enforce monotonicity
    out["P90"] = np.maximum(out["P50"], out["P90"])

    # Risk flag: sites where P90 - P10 > 5 are high uncertainty
    out["HighUncertainty"] = (out["P90"] - out["P10"]) > 5

    log.info(f"Final output shape: {out.shape[0]:,} rows x {out.shape[1]} cols")
    log.info(f"High uncertainty sites: {out['HighUncertainty'].sum():,} "
             f"({100*out['HighUncertainty'].mean():.1f}%)")

    # Final coverage check on ensemble bands
    y_true  = out[TARGET_S2].values
    in_band = ((y_true >= out["P10"].values) & (y_true <= out["P90"].values)).mean()
    mae_p50 = mean_absolute_error(y_true, out["P50"].values)
    log.info(f"Ensemble P50 MAE      : {mae_p50:.4f}")
    log.info(f"Ensemble P10-P90 coverage: {100*in_band:.1f}%  (ideal ~80%)")

    return out


def save_output(out: pd.DataFrame):
    out.to_parquet(OUT_PARQUET, index=False)
    out.to_csv(OUT_CSV, index=False)
    log.info(f"Saved: {OUT_PARQUET}  ({out.shape[0]:,} rows x {out.shape[1]} cols)")
    log.info(f"Saved: {OUT_CSV}")


# ===========================================================================
# Summary table
# ===========================================================================

def log_summary_table(out: pd.DataFrame):
    """Log a compact per-method comparison table."""
    log.info("\n" + "=" * 60)
    log.info("UNCERTAINTY QUANTIFICATION SUMMARY")
    log.info("=" * 60)

    y_true = out[TARGET_S2].values

    rows = [
        ("Phase 2 Final",  out["Final_Pred"].values, None,              None),
        ("MC P50",         out["MC_P50"].values,     out["MC_P10"].values, out["MC_P90"].values),
        ("Quantile P50",   out["Q_P50"].values,      out["Q_P10"].values,  out["Q_P90"].values),
        ("Ensemble P50",   out["P50"].values,         out["P10"].values,    out["P90"].values),
    ]

    log.info(f"  {'Method':<18s}  {'MAE':>7s}  {'RMSE':>8s}  {'Coverage':>10s}  {'Avg Width':>10s}")
    log.info("  " + "-" * 60)

    for label, point, lo, hi in rows:
        mae  = mean_absolute_error(y_true, point)
        rmse = float(np.sqrt(mean_squared_error(y_true, point)))
        if lo is not None and hi is not None:
            coverage  = ((y_true >= lo) & (y_true <= hi)).mean()
            avg_width = float(np.mean(hi - lo))
            log.info(f"  {label:<18s}  {mae:>7.4f}  {rmse:>8.4f}  "
                     f"{100*coverage:>9.1f}%  {avg_width:>10.1f}")
        else:
            log.info(f"  {label:<18s}  {mae:>7.4f}  {rmse:>8.4f}  {'n/a':>10s}  {'n/a':>10s}")

    log.info("=" * 60)
    log.info("Note: Coverage = % of actuals falling within [P10, P90].")
    log.info("      Ideal coverage is ~80% for P10-P90 interval.")
    log.info("      Higher coverage with wider bands is a trade-off.")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    log.info("=" * 60)
    log.info("Clinical Trial Enrollment Forecasting -- Phase 3")
    log.info("Uncertainty Quantification (Monte Carlo + Quantile Regression)")
    log.info("=" * 60)
    t_start = time.time()

    # Load
    s1_model, s2_model, feat_cols, cap_value = load_models_and_meta()
    df, phase2_pred = load_data()

    # Test set only (same definition as train_model.py)
    test = df[df["TimeIndex"] >= TEST_MIN].copy()
    log.info(f"Test set: {len(test):,} rows (2024, TimeIndex {TEST_MIN}+)")

    # Method A: Monte Carlo
    mc_df = run_monte_carlo(test, s1_model, s2_model, feat_cols, cap_value)

    # Method B: Quantile regression
    quantile_models = train_quantile_models(df, feat_cols, cap_value)
    q_df = predict_quantiles(test, s1_model, quantile_models, feat_cols)

    # Combine and save
    out = build_final_output(test, phase2_pred, mc_df, q_df)
    save_output(out)

    # Summary
    log_summary_table(out)

    elapsed = time.time() - t_start
    log.info(f"\nTotal runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log.info("Output: data/uncertainty_predictions.parquet")
    log.info("        data/uncertainty_predictions.csv")
    log.info("\nNext: Phase 4 -- dashboard.py (Streamlit, 6 tabs)")


if __name__ == "__main__":
    main()
