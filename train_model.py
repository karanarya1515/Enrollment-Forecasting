"""
train_model.py -- Phase 2: Two-Stage XGBoost Training Pipeline
Clinical Trial Enrollment Forecasting Platform
Karan Arya | MSc AI Batch 10 | Endpoint Clinical

Architecture:
  Stage 1: XGBoost Classifier   -- "Will this site enroll at all?" (binary)
  Stage 2: XGBoost Poisson Reg  -- "How many patients?"           (count)
  Final:   stage1_pred * stage2_pred (zeroed where Stage 1 predicts 0)

Changes from v1:
  - ASCII-only logging (fixes Windows cp1252 UnicodeEncodeError for checkmarks/arrows)
  - Proper validation split for early stopping: TimeIndex 21-24 held out from
    train. Model fits on 1-20, stops on val loss, then refits on 1-24 with
    the found n_estimators. Prevents early stopping from peeking at test.
  - Stage 2 hyperparameter tuning:
      max_depth 6->8, lr 0.05->0.03, min_child_weight 1->10, reg_alpha=0.1
      Deeper trees handle enrollment magnitude better.
      Higher min_child_weight prevents overfitting on rare high-count sites.
  - evaluate_stage2 now reports R^2 on both raw and capped targets.
  - Error breakdown by enrollment magnitude bucket (1-3, 4-10, 11-27, 28+).
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).parent
DATA_DIR     = ROOT / "data"
MODEL_DIR    = ROOT / "models"
LOG_DIR      = ROOT / "logs"
PARQUET_PATH = DATA_DIR / "quarterly_feature_dataset.parquet"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging -- ASCII only throughout; UTF-8 file handler for the log file.
# Wrapping stdout in UTF-8 stops cp1252 crashes on Windows without removing
# any characters -- non-encodable bytes are replaced instead of raising.
# ---------------------------------------------------------------------------
log_path = LOG_DIR / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"

_fmt = "%(asctime)s  %(levelname)-8s  %(message)s"

_file_handler = logging.FileHandler(log_path, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter(_fmt))

_stdout_stream = (
    io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    if hasattr(sys.stdout, "buffer")
    else sys.stdout
)
_stream_handler = logging.StreamHandler(_stdout_stream)
_stream_handler.setFormatter(logging.Formatter(_fmt))

logging.basicConfig(level=logging.INFO, handlers=[_stream_handler, _file_handler])
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
ID_COLS   = ["ETLDatabase", "SITE_ID", "Year", "Quarter", "TimeIndex"]
TARGET_S1 = "Target_HasEnrollment"        # binary 0/1
TARGET_S2 = "Target_QuarterlyEnrollment"  # integer count

EXPECTED_FEATURES = [
    # Screening (5)
    "Feat_ScreensThisQuarter", "Feat_ScreenFailsThisQuarter",
    "Feat_ScreenFailRate", "Feat_ScreenToEnrollRate",
    "Feat_HasScreeningActivity",
    # Supply chain (7)
    "Feat_KitsReceivedThisQuarter", "Feat_UniqueKitTypes",
    "Feat_KitsDispensedThisQuarter", "Feat_HasKitSupply",
    "Feat_ShipmentsReceivedThisQuarter", "Feat_AvgShipmentLeadDays",
    "Feat_HasShipmentActivity",
    # Time-series (12)
    "Feat_EnrollLag1Q", "Feat_EnrollLag2Q", "Feat_EnrollLag4Q",
    "Feat_ScreenLag1Q", "Feat_ScreenLag2Q",
    "Feat_EnrollMA2Q", "Feat_EnrollMA4Q",
    "Feat_EnrollTrend2Q", "Feat_EnrollVolatility4Q",
    "Feat_ActiveEnrollmentQuarters", "Feat_PrevQuarterWasZero",
    "Feat_ScreenMA2Q",
    # Site (2)
    "Feat_SiteMaturityCategory", "QuartersSinceActivation",
    # Hierarchical peer (6)
    "Feat_CountryPeerEnrollAvg", "Feat_CountrySiteCount",
    "Feat_RegionPeerEnrollAvg", "Feat_SponsorPeerEnrollAvg",
    "Feat_StudyPeerEnrollAvg", "Feat_StudySiteCount",
    # Engineered / interaction (8)
    "Feat_LogScreens", "Feat_LogEnrollLag1", "Feat_SqrtEnrollLag1",
    "Feat_ScreeningBinOrdinal", "Feat_ScreenXConversion",
    "Feat_ScreenXMaturity", "Feat_LaggedEnrollXActive",
    "Feat_HasSupplyData",
]

# Time-aware split boundaries
# Train core : TimeIndex  1-20  (Q1 2018 - Q4 2022)  -- fits model weights
# Train val  : TimeIndex 21-24  (2023)                -- early stopping only
# Train full : TimeIndex  1-24  (Q1 2018 - Q4 2023)  -- final model refit
# Test       : TimeIndex 25-28  (2024)                -- never seen during training
TRAIN_CORE_MAX = 20
TRAIN_VAL_MAX  = 24
TRAIN_CUTOFF   = 24
TEST_MIN       = 25

S2_CAP_PERCENTILE = 99   # cap Stage 2 target -- handles extreme-outlier sites


# ===========================================================================
# STEP 1 -- Load dataset
# ===========================================================================

def load_dataset(path: Path) -> pd.DataFrame:
    log.info(f"Loading dataset from {path}")
    t0 = time.time()
    df = pd.read_parquet(path)
    log.info(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} cols in {time.time()-t0:.1f}s")
    return df


# ===========================================================================
# STEP 2 -- Resolve feature columns
# ===========================================================================

def resolve_feature_cols(df: pd.DataFrame) -> list:
    feat_cols = sorted([c for c in df.columns if c.startswith("Feat_")])
    if "QuartersSinceActivation" in df.columns:
        feat_cols.append("QuartersSinceActivation")

    missing = [f for f in EXPECTED_FEATURES if f not in df.columns]
    if missing:
        log.warning(f"Expected features NOT in dataset ({len(missing)}): {missing}")

    extra = [c for c in feat_cols if c not in EXPECTED_FEATURES]
    if extra:
        log.info(f"Extra feature cols found (included): {extra}")

    log.info(f"Feature columns resolved: {len(feat_cols)} features")
    return feat_cols


# ===========================================================================
# STEP 3 -- Time-aware train/test split
# ===========================================================================

def time_split(df: pd.DataFrame):
    """
    Returns four non-overlapping splits:
      train_core : TimeIndex 1-20  (fit model weights)
      train_val  : TimeIndex 21-24 (early stopping -- never used for fitting)
      train_full : TimeIndex 1-24  (final refit after best n_estimators found)
      test       : TimeIndex 25-28 (held-out 2024, completely unseen)
    """
    train_core = df[df["TimeIndex"] <= TRAIN_CORE_MAX].copy()
    train_val  = df[(df["TimeIndex"] > TRAIN_CORE_MAX) & (df["TimeIndex"] <= TRAIN_VAL_MAX)].copy()
    train_full = df[df["TimeIndex"] <= TRAIN_CUTOFF].copy()
    test       = df[df["TimeIndex"] >= TEST_MIN].copy()

    log.info(f"Train core : {len(train_core):,} rows (TimeIndex 1-{TRAIN_CORE_MAX}, Q1 2018-Q4 2022)")
    log.info(f"Train val  : {len(train_val):,} rows  (TimeIndex {TRAIN_CORE_MAX+1}-{TRAIN_VAL_MAX}, 2023 -- early stop only)")
    log.info(f"Train full : {len(train_full):,} rows (TimeIndex 1-{TRAIN_CUTOFF}, Q1 2018-Q4 2023)")
    log.info(f"Test       : {len(test):,} rows  (TimeIndex {TEST_MIN}+, 2024)")

    assert set(train_full["TimeIndex"]).isdisjoint(set(test["TimeIndex"])), \
        "FATAL: TimeIndex overlap between train and test!"

    z  = (train_full[TARGET_S1] == 0).sum()
    nz = (train_full[TARGET_S1] == 1).sum()
    log.info(f"Train full target -- zeros: {z:,} ({100*z/len(train_full):.1f}%)  "
             f"non-zeros: {nz:,} ({100*nz/len(train_full):.1f}%)")

    return train_core, train_val, train_full, test


# ===========================================================================
# STEP 4 -- Stage 1: XGBoost Classifier
# ===========================================================================

def train_stage1(
    train_core: pd.DataFrame,
    train_val: pd.DataFrame,
    train_full: pd.DataFrame,
    feat_cols: list,
) -> xgb.XGBClassifier:
    """
    Two-phase training:
      Phase A: Fit on core (1-20), early-stop against val (21-24) to find
               best n_estimators without leaking test data into stopping.
      Phase B: Refit on full train (1-24) with that fixed n_estimators.

    scale_pos_weight computed from full train (33.4% zeros).
    """
    log.info("=" * 60)
    log.info("STAGE 1: XGBoost Classifier")
    log.info("=" * 60)

    n_zeros    = (train_full[TARGET_S1] == 0).sum()
    n_nonzeros = (train_full[TARGET_S1] == 1).sum()
    spw = n_zeros / n_nonzeros
    log.info(f"scale_pos_weight = {n_zeros:,} / {n_nonzeros:,} = {spw:.4f}")

    shared_params = dict(
        objective="binary:logistic",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=spw,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=40,
        verbosity=0,
        n_jobs=-1,
    )

    # Phase A: find best iteration
    log.info("Phase A: fitting on core (1-20), validating on 2023 (21-24)...")
    model_a = xgb.XGBClassifier(**shared_params)
    model_a.fit(
        train_core[feat_cols].fillna(0), train_core[TARGET_S1],
        eval_set=[(train_val[feat_cols].fillna(0), train_val[TARGET_S1])],
        verbose=False,
    )
    best_n = model_a.best_iteration + 1
    log.info(f"Phase A best iteration: {best_n}")

    # Phase B: refit on full train with fixed n_estimators
    log.info(f"Phase B: refitting on full train (1-24) with n_estimators={best_n}...")
    final_params = {**shared_params, "n_estimators": best_n, "early_stopping_rounds": None}
    model = xgb.XGBClassifier(**final_params)
    t0 = time.time()
    model.fit(
        train_full[feat_cols].fillna(0), train_full[TARGET_S1],
        verbose=False,
    )
    log.info(f"Stage 1 complete in {time.time()-t0:.1f}s")
    return model


def evaluate_stage1(
    model: xgb.XGBClassifier,
    test: pd.DataFrame,
    feat_cols: list,
) -> tuple:
    log.info("\n--- Stage 1 Evaluation (Test Set: 2024) ---")

    X_test = test[feat_cols].fillna(0)
    y_test = test[TARGET_S1]

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred      = (y_pred_prob >= 0.5).astype(int)

    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    accuracy  = accuracy_score(y_test, y_pred)
    auc       = roc_auc_score(y_test, y_pred_prob)
    cm        = confusion_matrix(y_test, y_pred)

    log.info(f"  F1        : {f1:.4f}")
    log.info(f"  Precision : {precision:.4f}")
    log.info(f"  Recall    : {recall:.4f}")
    log.info(f"  Accuracy  : {accuracy:.4f}")
    log.info(f"  AUC-ROC   : {auc:.4f}")
    log.info(f"  Confusion Matrix:\n{cm}")
    log.info("\n" + classification_report(y_test, y_pred, target_names=["Zero", "Non-Zero"]))

    metrics = {
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4),
        "auc_roc": round(auc, 4),
        "confusion_matrix": cm.tolist(),
    }
    return metrics, y_pred, y_pred_prob


# ===========================================================================
# STEP 5 -- Stage 2: XGBoost Poisson Regressor
# ===========================================================================

def train_stage2(
    train_core: pd.DataFrame,
    train_val: pd.DataFrame,
    train_full: pd.DataFrame,
    feat_cols: list,
) -> tuple:
    """
    Same two-phase approach as Stage 1.
    Trains on non-zero rows only (enrollment magnitude, not zero/non-zero).
    Target capped at 99th percentile before training.

    Tuning changes vs v1:
      max_depth      6 -> 8   (deeper trees for enrollment magnitude)
      n_estimators 500 -> 1000
      learning_rate 0.05 -> 0.03
      min_child_weight 1 -> 10 (prevents overfitting on rare high-count sites)
      reg_alpha = 0.1          (L1 regularisation -- helps sparse count tails)
      colsample_bytree 0.8 -> 0.7
    """
    log.info("=" * 60)
    log.info("STAGE 2: XGBoost Poisson Regressor")
    log.info("=" * 60)

    # Filter non-zero rows from each split
    core_nz = train_core[train_core[TARGET_S1] == 1].copy()
    val_nz  = train_val[train_val[TARGET_S1] == 1].copy()
    full_nz = train_full[train_full[TARGET_S1] == 1].copy()

    log.info(f"Non-zero rows -- core: {len(core_nz):,}  val: {len(val_nz):,}  full: {len(full_nz):,}")

    # Cap at 99th percentile (computed on full train non-zero rows)
    cap_value = float(np.percentile(full_nz[TARGET_S2], S2_CAP_PERCENTILE))
    rows_capped = (full_nz[TARGET_S2] > cap_value).sum()
    log.info(f"99th percentile cap: {cap_value:.0f} patients/quarter  "
             f"({rows_capped:,} rows capped, {100*rows_capped/len(full_nz):.2f}%)")

    def apply_cap(df):
        d = df.copy()
        d["target_capped"] = d[TARGET_S2].clip(upper=cap_value)
        return d

    core_nz = apply_cap(core_nz)
    val_nz  = apply_cap(val_nz)
    full_nz = apply_cap(full_nz)

    log.info(
        f"Target after cap -- mean: {full_nz['target_capped'].mean():.2f}  "
        f"median: {full_nz['target_capped'].median():.2f}  "
        f"max: {full_nz['target_capped'].max():.0f}"
    )

    shared_params = dict(
        objective="count:poisson",
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="poisson-nloglik",
        early_stopping_rounds=40,
        verbosity=0,
        n_jobs=-1,
    )

    # Phase A: find best iteration
    log.info("Phase A: fitting on non-zero core rows, validating on 2023 non-zero...")
    model_a = xgb.XGBRegressor(**shared_params)
    model_a.fit(
        core_nz[feat_cols].fillna(0), core_nz["target_capped"],
        eval_set=[(val_nz[feat_cols].fillna(0), val_nz["target_capped"])],
        verbose=False,
    )
    best_n = model_a.best_iteration + 1
    log.info(f"Phase A best iteration: {best_n}")

    # Phase B: refit on full non-zero train
    log.info(f"Phase B: refitting on full non-zero train (1-24) with n_estimators={best_n}...")
    final_params = {**shared_params, "n_estimators": best_n, "early_stopping_rounds": None}
    model = xgb.XGBRegressor(**final_params)
    t0 = time.time()
    model.fit(
        full_nz[feat_cols].fillna(0), full_nz["target_capped"],
        verbose=False,
    )
    log.info(f"Stage 2 complete in {time.time()-t0:.1f}s")
    return model, cap_value


def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Poisson deviance: 2 * mean[ y*log(y/yhat) - (y - yhat) ]"""
    y_pred = np.clip(y_pred, 1e-8, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0)
    return float(2 * np.mean(log_ratio - (y_true - y_pred)))


def evaluate_stage2(
    model: xgb.XGBRegressor,
    test: pd.DataFrame,
    feat_cols: list,
    cap_value: float,
) -> tuple:
    log.info("\n--- Stage 2 Evaluation (Non-Zero Test Rows: 2024) ---")

    test_nz = test[test[TARGET_S1] == 1].copy()
    log.info(f"Non-zero test rows: {len(test_nz):,}")

    X_test = test_nz[feat_cols].fillna(0)
    y_test = test_nz[TARGET_S2].values
    y_pred = model.predict(X_test)

    # Raw evaluation (uncapped actuals) -- true real-world picture
    mae_raw  = mean_absolute_error(y_test, y_pred)
    rmse_raw = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2_raw   = r2_score(y_test, y_pred)
    pdev     = poisson_deviance(y_test, y_pred)

    # Capped evaluation -- apples-to-apples with what the model was trained on
    y_test_capped = np.clip(y_test, 0, cap_value)
    mae_cap  = mean_absolute_error(y_test_capped, y_pred)
    rmse_cap = float(np.sqrt(mean_squared_error(y_test_capped, y_pred)))
    r2_cap   = r2_score(y_test_capped, y_pred)

    log.info("  -- Raw (uncapped actuals -- true real-world) --")
    log.info(f"  MAE              : {mae_raw:.4f}")
    log.info(f"  RMSE             : {rmse_raw:.4f}  (inflated by outlier sites)")
    log.info(f"  R^2              : {r2_raw:.4f}")
    log.info(f"  Poisson Deviance : {pdev:.4f}")
    log.info("  -- Capped (apples-to-apples with training target) --")
    log.info(f"  MAE  (capped)    : {mae_cap:.4f}")
    log.info(f"  RMSE (capped)    : {rmse_cap:.4f}")
    log.info(f"  R^2  (capped)    : {r2_cap:.4f}")

    # Error breakdown by enrollment magnitude
    log.info("  -- Error breakdown by enrollment magnitude --")
    test_nz["y_pred"]  = y_pred
    test_nz["abs_err"] = np.abs(y_test - y_pred)
    for label, lo, hi in [
        ("1-3 pts",   1,    3),
        ("4-10 pts",  4,   10),
        ("11-27 pts", 11,  27),
        ("28+ pts",   28, 9999),
    ]:
        mask   = (test_nz[TARGET_S2] >= lo) & (test_nz[TARGET_S2] <= hi)
        subset = test_nz[mask]
        if len(subset) == 0:
            continue
        log.info(
            f"    {label:<12s}  n={len(subset):>6,}  "
            f"MAE={subset['abs_err'].mean():.3f}  "
            f"median_actual={subset[TARGET_S2].median():.0f}  "
            f"median_pred={subset['y_pred'].median():.1f}"
        )

    metrics = {
        "mae_raw": round(mae_raw, 4),
        "rmse_raw": round(rmse_raw, 4),
        "r2_raw": round(r2_raw, 4),
        "poisson_deviance": round(pdev, 4),
        "mae_capped": round(mae_cap, 4),
        "rmse_capped": round(rmse_cap, 4),
        "r2_capped": round(r2_cap, 4),
        "n_test_nonzero": len(test_nz),
    }

    # Return predictions aligned to full test index (zeros for Stage1=0 rows)
    s2_pred = pd.Series(0.0, index=test.index)
    s2_pred.loc[test_nz.index] = y_pred
    return metrics, s2_pred


# ===========================================================================
# STEP 6 -- Combined predictions
# ===========================================================================

def combine_predictions(
    test: pd.DataFrame,
    s1_pred: np.ndarray,
    s2_pred: pd.Series,
) -> tuple:
    """Final = stage1_pred * stage2_pred. Zero wherever Stage 1 predicts 0."""
    log.info("\n--- Combined Predictions ---")

    results = test[ID_COLS + [TARGET_S1, TARGET_S2]].copy()
    results["Stage1_Pred"] = s1_pred
    results["Stage2_Pred"] = s2_pred.values
    results["Final_Pred"]  = (s1_pred * s2_pred.values).round().clip(0).astype(int)

    y_true = results[TARGET_S2].values
    y_pred = results["Final_Pred"].values

    mae_all  = mean_absolute_error(y_true, y_pred)
    rmse_all = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2_all   = r2_score(y_true, y_pred)

    log.info(f"  Combined MAE  (all rows): {mae_all:.4f}")
    log.info(f"  Combined RMSE (all rows): {rmse_all:.4f}")
    log.info(f"  Combined R^2  (all rows): {r2_all:.4f}")

    for s1_val, label in [(0, "Stage1=0 (pred zero)"), (1, "Stage1=1 (pred non-zero)")]:
        mask   = results["Stage1_Pred"] == s1_val
        subset = results[mask]
        if len(subset) == 0:
            continue
        mae_s = mean_absolute_error(subset[TARGET_S2], subset["Final_Pred"])
        log.info(f"    {label}: n={len(subset):,}  MAE={mae_s:.4f}")

    combined_metrics = {
        "mae_all": round(mae_all, 4),
        "rmse_all": round(rmse_all, 4),
        "r2_all": round(r2_all, 4),
    }
    return results, combined_metrics


# ===========================================================================
# STEP 7 -- Feature importance
# ===========================================================================

def log_feature_importance(model, label: str, top_n: int = 20) -> list:
    importance = model.get_booster().get_score(importance_type="gain")
    imp_df = (
        pd.DataFrame.from_dict(importance, orient="index", columns=["gain"])
        .rename_axis("feature")
        .reset_index()
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )

    log.info(f"\n--- {label} -- Top {top_n} Features (by Gain) ---")
    for _, row in imp_df.head(top_n).iterrows():
        log.info(f"  {row['feature']:<40s} {row['gain']:.2f}")

    expected_leaders = [
        "Feat_PrevQuarterWasZero", "Feat_EnrollLag1Q", "Feat_EnrollMA2Q"
    ]
    top5 = imp_df["feature"].head(5).tolist()
    for leader in expected_leaders:
        if leader in top5:
            log.info(f"  [TOP 5]  {leader}")
        elif leader in imp_df["feature"].values:
            rank = imp_df[imp_df["feature"] == leader].index[0] + 1
            log.info(f"  [rank #{rank}] {leader}")

    return imp_df.head(top_n).to_dict("records")


# ===========================================================================
# STEP 8 -- Save models
# ===========================================================================

def save_models(
    stage1: xgb.XGBClassifier,
    stage2: xgb.XGBRegressor,
    feat_cols: list,
    cap_value: float,
    metrics: dict,
):
    stage1_path = MODEL_DIR / "stage1_classifier.json"
    stage2_path = MODEL_DIR / "stage2_regressor.json"
    meta_path   = MODEL_DIR / "feature_columns.json"

    stage1.save_model(str(stage1_path))
    stage2.save_model(str(stage2_path))
    log.info(f"Stage 1 model saved: {stage1_path}")
    log.info(f"Stage 2 model saved: {stage2_path}")

    meta = {
        "feature_columns": feat_cols,
        "n_features": len(feat_cols),
        "stage2_cap_percentile": S2_CAP_PERCENTILE,
        "stage2_cap_value": float(cap_value),
        "train_core_time_index": f"1-{TRAIN_CORE_MAX}",
        "train_val_time_index":  f"{TRAIN_CORE_MAX+1}-{TRAIN_VAL_MAX}",
        "train_full_time_index": f"1-{TRAIN_CUTOFF}",
        "test_time_index": f"{TEST_MIN}+",
        "train_period": "Q1 2018 - Q4 2023 (TimeIndex 1-24)",
        "test_period": "2024 (TimeIndex 25-28)",
        "target_s1": TARGET_S1,
        "target_s2": TARGET_S2,
        "metrics": metrics,
        "model_version": "v2",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Feature metadata saved: {meta_path}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    log.info("=" * 60)
    log.info("Clinical Trial Enrollment Forecasting -- Phase 2 (v2)")
    log.info("Two-Stage XGBoost Training Pipeline")
    log.info("=" * 60)
    t_start = time.time()

    # 1. Load
    df = load_dataset(PARQUET_PATH)
    assert TARGET_S1 in df.columns, f"Missing: {TARGET_S1}"
    assert TARGET_S2 in df.columns, f"Missing: {TARGET_S2}"
    assert "TimeIndex" in df.columns, "Missing: TimeIndex"
    log.info(f"TimeIndex range: {df['TimeIndex'].min()} - {df['TimeIndex'].max()}")

    # 2. Features
    feat_cols = resolve_feature_cols(df)

    # 3. Split
    train_core, train_val, train_full, test = time_split(df)

    # 4. Stage 1
    s1_model = train_stage1(train_core, train_val, train_full, feat_cols)
    s1_metrics, s1_pred, s1_pred_prob = evaluate_stage1(s1_model, test, feat_cols)
    s1_importance = log_feature_importance(s1_model, "Stage 1 Classifier")

    # 5. Stage 2
    s2_model, cap_value = train_stage2(train_core, train_val, train_full, feat_cols)
    s2_metrics, s2_pred = evaluate_stage2(s2_model, test, feat_cols, cap_value)
    s2_importance = log_feature_importance(s2_model, "Stage 2 Regressor")

    # 6. Combined
    results_df, combined_metrics = combine_predictions(test, s1_pred, s2_pred)

    pred_path = DATA_DIR / "test_predictions_2024.csv"
    results_df.to_csv(pred_path, index=False)
    log.info(f"Test predictions saved: {pred_path}  ({len(results_df):,} rows)")

    # 7+8. Save
    all_metrics = {
        "stage1": s1_metrics,
        "stage2": s2_metrics,
        "combined": combined_metrics,
        "stage1_top20_features": s1_importance,
        "stage2_top20_features": s2_importance,
    }
    save_models(s1_model, s2_model, feat_cols, cap_value, all_metrics)

    # Summary
    elapsed = time.time() - t_start
    log.info("\n" + "=" * 60)
    log.info("PHASE 2 COMPLETE")
    log.info(f"Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log.info("=" * 60)
    log.info("Stage 1 (Classifier):")
    log.info(f"  F1={s1_metrics['f1']:.4f}  AUC={s1_metrics['auc_roc']:.4f}  "
             f"Recall={s1_metrics['recall']:.4f}  Precision={s1_metrics['precision']:.4f}")
    log.info("Stage 2 (Poisson Regressor -- capped eval, non-zero rows):")
    log.info(f"  MAE={s2_metrics['mae_capped']:.4f}  RMSE={s2_metrics['rmse_capped']:.4f}  "
             f"R^2={s2_metrics['r2_capped']:.4f}  Poisson Dev={s2_metrics['poisson_deviance']:.4f}")
    log.info("Combined (all test rows):")
    log.info(f"  MAE={combined_metrics['mae_all']:.4f}  "
             f"RMSE={combined_metrics['rmse_all']:.4f}  "
             f"R^2={combined_metrics['r2_all']:.4f}")
    log.info("=" * 60)
    log.info("Saved:")
    log.info("  models/stage1_classifier.json")
    log.info("  models/stage2_regressor.json")
    log.info("  models/feature_columns.json")
    log.info("  data/test_predictions_2024.csv")
    log.info("  logs/")
    log.info("\nNext: Phase 3 -- uncertainty.py (Monte Carlo / Quantile bounds)")


if __name__ == "__main__":
    main()