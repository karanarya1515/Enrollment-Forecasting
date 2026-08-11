# =============================================================================
# build_feature_dataset.py
# enrollment_forecasting_v2 — Phase 1: Feature Dataset Builder
# =============================================================================
# NO DATABASE CONNECTION NEEDED.
# Reads 5 CSVs exported from VSCode and builds the full feature dataset.
# NOTE: Q4 (visits) dropped — the visits table is empty for the studies in scope.
#
# HOW TO RUN:
#   1. Export Q1-Q3, Q5-Q6 from VSCode and save to data/raw/
#   2. Run: python build_feature_dataset.py
# =============================================================================

import logging
import time

import numpy as np
import pandas as pd

from config import (
    RAW_CSV,
    FEATURE_DATASET_PATH,
    FEATURE_DATASET_CSV,
    FEATURE_CONFIG,
    LOG_CONFIG,
)

logging.basicConfig(
    level=LOG_CONFIG["level"],
    format=LOG_CONFIG["format"],
    datefmt=LOG_CONFIG["datefmt"],
)
logger = logging.getLogger(__name__)

JOIN_KEYS = ["ETLDatabase", "SITE_ID", "Year", "Quarter"]

# Q4 visits removed — no usable data in the visit tables
REQUIRED_CSVS = ["enrollments"]
OPTIONAL_CSVS = ["screenings", "site_metadata", "kits", "shipments"]


# ===========================================================================
# STEP 1: Load CSVs
# ===========================================================================
def load_all_csvs() -> dict:
    logger.info("Loading CSVs from data/raw/ ...")
    data = {}
    for name, path in RAW_CSV.items():
        if name == "visits":
            continue  # dropped — no data
        if not path.exists():
            if name in REQUIRED_CSVS:
                raise FileNotFoundError(
                    f"Required file missing: {path.name}\n"
                    f"Run q1_enrollments.sql in VSCode and save to data/raw/"
                )
            logger.warning(f"  ⚠️  Optional file missing: {path.name} — skipped")
            data[name] = pd.DataFrame()
        else:
            df = pd.read_csv(path, low_memory=False)
            logger.info(f"  ✅ {name}: {len(df):,} rows")
            data[name] = df
    return data


# ===========================================================================
# STEP 2: Build quarter spine
# ===========================================================================
def build_quarter_spine(
    enrollments: pd.DataFrame,
    screenings: pd.DataFrame,
    supply_spine: pd.DataFrame = None
) -> pd.DataFrame:
    logger.info("Building quarter spine...")

    # Base: enrollment events
    e = enrollments[["ETLDatabase", "SITE_ID", "EnrollYear", "EnrollQuarter"]].copy()
    e.columns = JOIN_KEYS
    frames = [e]

    # Add screening events
    if not screenings.empty:
        s = screenings[["ETLDatabase", "SITE_ID", "ScrnYear", "ScrnQuarter"]].copy()
        s.columns = JOIN_KEYS
        frames.append(s)

    # Add supply-active site-quarters (sites with kits/shipments but no patient activity)
    # These represent pre-enrollment readiness — important zero rows
    if supply_spine is not None and not supply_spine.empty:
        sup = supply_spine[["ETLDatabase", "SITE_ID", "Year", "Quarter"]].copy()
        sup.columns = JOIN_KEYS
        frames.append(sup)
        logger.info(f"  Supply spine adds {len(sup):,} additional site-quarters")

    spine = pd.concat(frames).drop_duplicates()
    spine["SITE_ID"] = spine["SITE_ID"].astype(str)
    spine["ETLDatabase"] = spine["ETLDatabase"].astype(str)
    spine = spine[spine["Year"].between(2018, 2030) & spine["Quarter"].between(1, 4)]
    spine = spine.sort_values(JOIN_KEYS).reset_index(drop=True)

    logger.info(f"Quarter spine: {len(spine):,} rows across "
                f"{spine['ETLDatabase'].nunique()} studies, "
                f"{spine['SITE_ID'].nunique()} sites")
    return spine


# ===========================================================================
# STEP 3: Merge all feature tables
# ===========================================================================
def merge_all_features(spine: pd.DataFrame, data: dict) -> pd.DataFrame:
    logger.info("Merging feature tables onto spine...")
    df = spine.copy()

    # --- Enrollment target (required) ---
    e = data["enrollments"].copy()
    e.columns = ["ETLDatabase", "SITE_ID", "Year", "Quarter", "Target_QuarterlyEnrollment"]
    e["SITE_ID"] = e["SITE_ID"].astype(str)
    df = df.merge(e, on=JOIN_KEYS, how="left")
    df["Target_QuarterlyEnrollment"] = df["Target_QuarterlyEnrollment"].fillna(0).astype(int)
    df["Target_HasEnrollment"] = (df["Target_QuarterlyEnrollment"] > 0).astype(int)
    logger.info("  ✅ Enrollment target merged")

    # --- Screenings (optional) ---
    if not data.get("screenings", pd.DataFrame()).empty:
        s = data["screenings"].copy()
        s.columns = ["ETLDatabase", "SITE_ID", "Year", "Quarter",
                     "Feat_ScreensThisQuarter", "Feat_ScreenFailsThisQuarter"]
        s["SITE_ID"] = s["SITE_ID"].astype(str)
        df = df.merge(s, on=JOIN_KEYS, how="left")
        df["Feat_ScreensThisQuarter"] = df["Feat_ScreensThisQuarter"].fillna(0).astype(int)
        df["Feat_ScreenFailsThisQuarter"] = df["Feat_ScreenFailsThisQuarter"].fillna(0).astype(int)
        logger.info("  ✅ Screenings merged")
    else:
        df["Feat_ScreensThisQuarter"] = 0
        df["Feat_ScreenFailsThisQuarter"] = 0
        logger.warning("  ⚠️  Screenings not available — set to 0")

    # --- Site metadata (optional) ---
    if not data.get("site_metadata", pd.DataFrame()).empty:
        site = data["site_metadata"].copy()
        site["SITE_ID"] = site["SITE_ID"].astype(str)

        # Fix 9999 sentinel values — means no limit set, treat as null
        for limit_col in ["EnrollmentLimit", "ScreeningLimit"]:
            if limit_col in site.columns:
                site[limit_col] = site[limit_col].replace(9999, np.nan)

        df = df.merge(site, on=["ETLDatabase", "SITE_ID"], how="left")

        # If COUNTRY_NAME empty (join bug in Q3), derive Sponsor from ETLDatabase
        if "COUNTRY_NAME" not in df.columns or df["COUNTRY_NAME"].isna().all():
            logger.warning("  COUNTRY_NAME empty — deriving Sponsor from ETLDatabase as fallback")
            df["Sponsor"] = df["ETLDatabase"].str.split("_").str[0]
        elif "COUNTRY_NAME" in df.columns and df["COUNTRY_NAME"].isna().mean() > 0.5:
            logger.warning("  COUNTRY_NAME >50% empty — also deriving Sponsor")
            df["Sponsor"] = df["ETLDatabase"].str.split("_").str[0]

        logger.info("  Site metadata merged")

    # --- Site country (Q7 — ALL sponsors via LocationAddress) ---
    if not data.get("site_country", pd.DataFrame()).empty:
        sc = data["site_country"].copy()
        sc["SITE_ID"] = sc["SITE_ID"].astype(str)

        # Drop COUNTRY_ID from df first if it exists (came from Q3 — mostly NULL)
        # Q7 has better coverage so we replace it entirely
        if "COUNTRY_ID" in df.columns:
            df = df.drop(columns=["COUNTRY_ID"])

        # Select only needed columns from Q7
        q7_cols = ["ETLDatabase", "SITE_ID"]
        for col in ["COUNTRY_ID", "COUNTRY_NAME", "CountryISO", "REGION_NAME"]:
            if col in sc.columns:
                q7_cols.append(col)
        sc = sc[q7_cols]

        # Deduplicate — one row per site (Q7 SQL should already do this but be safe)
        sc = sc.drop_duplicates(subset=["ETLDatabase", "SITE_ID"], keep="first")

        df = df.merge(sc, on=["ETLDatabase", "SITE_ID"], how="left")
        n_country = df["COUNTRY_NAME"].notna().sum() if "COUNTRY_NAME" in df.columns else 0
        n_region  = df["REGION_NAME"].notna().sum() if "REGION_NAME" in df.columns else 0
        pct = n_country / len(df) * 100
        logger.info(f"  Site country merged: {n_country:,} rows ({pct:.1f}%) have COUNTRY_NAME")
        logger.info(f"  Region coverage:     {n_region:,} rows ({n_region/len(df)*100:.1f}%) have REGION_NAME")
    
    # --- Kits (optional) ---
    if not data.get("kits", pd.DataFrame()).empty:
        k = data["kits"].copy()
        k.columns = ["ETLDatabase", "SITE_ID", "Year", "Quarter",
                     "Feat_KitsReceivedThisQuarter", "Feat_UniqueKitTypes",
                     "Feat_KitsDispensedThisQuarter"]
        k["SITE_ID"] = k["SITE_ID"].astype(str)
        df = df.merge(k, on=JOIN_KEYS, how="left")
        for c in ["Feat_KitsReceivedThisQuarter", "Feat_UniqueKitTypes",
                  "Feat_KitsDispensedThisQuarter"]:
            df[c] = df[c].fillna(0).astype(int)
        df["Feat_HasKitSupply"] = (df["Feat_KitsReceivedThisQuarter"] > 0).astype(int)
        logger.info("  ✅ Kits merged")

    # --- Shipments (optional) ---
    if not data.get("shipments", pd.DataFrame()).empty:
        sh = data["shipments"].copy()
        sh.columns = ["ETLDatabase", "SITE_ID", "Year", "Quarter",
                      "Feat_ShipmentsReceivedThisQuarter", "Feat_AvgShipmentLeadDays"]
        sh["SITE_ID"] = sh["SITE_ID"].astype(str)
        df = df.merge(sh, on=JOIN_KEYS, how="left")
        df["Feat_ShipmentsReceivedThisQuarter"] = (
            df["Feat_ShipmentsReceivedThisQuarter"].fillna(0).astype(int)
        )
        df["Feat_HasShipmentActivity"] = (
            (df["Feat_ShipmentsReceivedThisQuarter"] > 0).astype(int)
        )
        logger.info("  ✅ Shipments merged")

    logger.info(f"After all merges: {len(df):,} rows, {len(df.columns)} columns")
    return df


# ===========================================================================
# STEP 4: Base derived features
# ===========================================================================
def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing base derived features...")

    # Screen fail rate
    df["Feat_ScreenFailRate"] = np.where(
        df["Feat_ScreensThisQuarter"] > 0,
        df["Feat_ScreenFailsThisQuarter"] / df["Feat_ScreensThisQuarter"],
        0.0
    ).clip(0, 1)

    # Screen to enroll conversion rate
    df["Feat_ScreenToEnrollRate"] = np.where(
        df["Feat_ScreensThisQuarter"] > 0,
        df["Target_QuarterlyEnrollment"] / df["Feat_ScreensThisQuarter"],
        0.0
    ).clip(0, 1)

    # Site maturity in quarters since first activation
    if "FirstActivationDate" in df.columns:
        df["FirstActivationDate"] = pd.to_datetime(df["FirstActivationDate"], errors="coerce")
        quarter_month = ((df["Quarter"] - 1) * 3 + 1).astype(str)
        df["_QStart"] = pd.to_datetime(
            df["Year"].astype(str) + "-" + quarter_month + "-01", errors="coerce"
        )
        df["QuartersSinceActivation"] = (
            (df["_QStart"] - df["FirstActivationDate"]).dt.days / 91.25
        ).clip(0).fillna(0).astype(int)
        df = df.drop(columns=["_QStart"])
    else:
        df["QuartersSinceActivation"] = 0

    # Time index for walk-forward CV splitting
    df["TimeIndex"] = (df["Year"] - 2018) * 4 + df["Quarter"]

    # Site maturity bucket: 0=New(0-2Q), 1=Developing(3-5Q), 2=Mature(6Q+)
    df["Feat_SiteMaturityCategory"] = pd.cut(
        df["QuartersSinceActivation"].clip(0),
        bins=[-1, 2, 5, float("inf")],
        labels=[0, 1, 2]
    ).astype(float).fillna(0).astype(int)

    # Screening activity flag
    df["Feat_HasScreeningActivity"] = (df["Feat_ScreensThisQuarter"] > 0).astype(int)

    return df


# ===========================================================================
# STEP 5: Time-series features
# ===========================================================================
def add_timeseries_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding time-series features (lags, rolling averages, trends)...")
    df = df.sort_values(["ETLDatabase", "SITE_ID", "Year", "Quarter"]).reset_index(drop=True)
    grp = ["ETLDatabase", "SITE_ID"]

    # Enrollment lags — MOST IMPORTANT features
    for lag in FEATURE_CONFIG["lag_windows"]:  # [1, 2, 4]
        df[f"Feat_EnrollLag{lag}Q"] = (
            df.groupby(grp)["Target_QuarterlyEnrollment"]
            .shift(lag).fillna(0).astype(float)
        )

    # Screening lags (where available)
    for lag in [1, 2]:
        df[f"Feat_ScreenLag{lag}Q"] = (
            df.groupby(grp)["Feat_ScreensThisQuarter"]
            .shift(lag).fillna(0).astype(float)
        )

    # Enrollment rolling averages
    for w in FEATURE_CONFIG["rolling_windows"]:  # [2, 4]
        df[f"Feat_EnrollMA{w}Q"] = (
            df.groupby(grp)["Target_QuarterlyEnrollment"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            .fillna(0)
        )

    # Enrollment trend (momentum: last Q minus Q before that)
    df["Feat_EnrollTrend2Q"] = (
        df.groupby(grp)["Target_QuarterlyEnrollment"]
        .transform(lambda x: x.shift(1) - x.shift(2))
        .fillna(0)
    )

    # Enrollment volatility (std over last 4 quarters)
    df["Feat_EnrollVolatility4Q"] = (
        df.groupby(grp)["Target_QuarterlyEnrollment"]
        .transform(lambda x: x.shift(1).rolling(4, min_periods=2).std())
        .fillna(0)
    )

    # Cumulative active enrollment quarters (site track record)
    df["Feat_ActiveEnrollmentQuarters"] = (
        df.groupby(grp)["Target_HasEnrollment"]
        .transform(lambda x: x.shift(1).expanding().sum())
        .fillna(0).astype(int)
    )

    # Was last quarter zero? (97.7% chance of staying zero — strongest predictor)
    df["Feat_PrevQuarterWasZero"] = (
        df.groupby(grp)["Target_QuarterlyEnrollment"]
        .shift(1).fillna(0).eq(0).astype(int)
    )

    # Screening rolling average
    df["Feat_ScreenMA2Q"] = (
        df.groupby(grp)["Feat_ScreensThisQuarter"]
        .transform(lambda x: x.shift(1).rolling(2, min_periods=1).mean())
        .fillna(0)
    )

    logger.info(f"Time-series features added. Shape: {df.shape}")
    return df


# ===========================================================================
# STEP 6: Hierarchical / peer features
# ===========================================================================
def add_hierarchical_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding hierarchical/peer features...")

    # Sponsor — always available from ETLDatabase prefix
    df["Sponsor"] = df["ETLDatabase"].str.split("_").str[0]

    # Country-level peer average — use COUNTRY_NAME where it is populated
    # Falls back to COUNTRY_ID GUID, then skips if neither available
    if "COUNTRY_NAME" in df.columns and df["COUNTRY_NAME"].notna().sum() > 0:
        country_grp = "COUNTRY_NAME"
    elif "COUNTRY_ID" in df.columns and df["COUNTRY_ID"].notna().sum() > 0:
        country_grp = "COUNTRY_ID"
    else:
        country_grp = None

    if country_grp:
        df["Feat_CountryPeerEnrollAvg"] = (
            df.groupby(["ETLDatabase", country_grp, "Year", "Quarter"])
            ["Feat_EnrollLag1Q"].transform("mean").fillna(0)
        )
        df["Feat_CountrySiteCount"] = (
            df.groupby(["ETLDatabase", country_grp, "Year", "Quarter"])
            ["SITE_ID"].transform("count").fillna(0)
        )
        logger.info(f"  Country peer features built using: {country_grp}")
    else:
        df["Feat_CountryPeerEnrollAvg"] = 0.0
        df["Feat_CountrySiteCount"]     = 0
        logger.warning("  No country data — country peer features set to 0")

    # Region-level peer average — available where REGION_NAME populated
    if "REGION_NAME" in df.columns and df["REGION_NAME"].notna().sum() > 0:
        df["Feat_RegionPeerEnrollAvg"] = (
            df.groupby(["REGION_NAME", "Year", "Quarter"])
            ["Feat_EnrollLag1Q"].transform("mean").fillna(0)
        )
        logger.info("  Region peer features built using REGION_NAME")
    else:
        df["Feat_RegionPeerEnrollAvg"] = 0.0

    # Sponsor-level peer average
    df["Feat_SponsorPeerEnrollAvg"] = (
        df.groupby(["Sponsor", "Year", "Quarter"])
        ["Feat_EnrollLag1Q"].transform("mean").fillna(0)
    )
    logger.info(f"  Sponsor peer features built: {df['Sponsor'].nunique()} sponsors")

    # Study-level peer average
    df["Feat_StudyPeerEnrollAvg"] = (
        df.groupby(["ETLDatabase", "Year", "Quarter"])
        ["Feat_EnrollLag1Q"].transform("mean").fillna(0)
    )

    # Number of active sites per study-quarter
    df["Feat_StudySiteCount"] = (
        df.groupby(["ETLDatabase", "Year", "Quarter"])
        ["SITE_ID"].transform("count")
    )

    logger.info(f"Hierarchical features added. Shape: {df.shape}")
    return df


# ===========================================================================
# STEP 7: Nonlinear & interaction features
# ===========================================================================
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding engineered/interaction features...")

    # Log transforms (handle extreme outlier sites)
    df["Feat_LogScreens"] = np.log1p(df["Feat_ScreensThisQuarter"])
    df["Feat_LogEnrollLag1"] = np.log1p(df["Feat_EnrollLag1Q"])
    df["Feat_SqrtEnrollLag1"] = np.sqrt(df["Feat_EnrollLag1Q"])

    # Screening bins (ordinal encoding for tree models)
    bins = FEATURE_CONFIG["screening_bins"]
    labels = FEATURE_CONFIG["screening_bin_labels"]
    bin_map = {lbl: i for i, lbl in enumerate(labels)}
    df["Feat_ScreeningBinOrdinal"] = pd.cut(
        df["Feat_ScreensThisQuarter"],
        bins=bins, labels=labels, right=False
    ).astype(str).map(bin_map).fillna(0).astype(int)

    # Interaction: screening × conversion rate
    df["Feat_ScreenXConversion"] = (
        df["Feat_ScreensThisQuarter"] * df["Feat_ScreenToEnrollRate"].fillna(0)
    )

    # Interaction: screening × site maturity
    df["Feat_ScreenXMaturity"] = (
        df["Feat_ScreensThisQuarter"] *
        np.log1p(df["QuartersSinceActivation"].clip(0))
    )

    # Interaction: lagged enrollment × enrollment open flag
    enroll_open = df.get("EnrollmentOpen", pd.Series(1, index=df.index)).fillna(1)
    df["Feat_LaggedEnrollXActive"] = df["Feat_EnrollLag1Q"] * enroll_open

    # Supply readiness flag
    has_kits = df.get("Feat_HasKitSupply", pd.Series(0, index=df.index))
    has_ships = df.get("Feat_HasShipmentActivity", pd.Series(0, index=df.index))
    df["Feat_HasSupplyData"] = ((has_kits + has_ships) > 0).astype(int)

    logger.info(f"Engineered features added. Shape: {df.shape}")
    return df


# ===========================================================================
# STEP 8: Finalize and save
# ===========================================================================
def finalize_and_save(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Finalizing dataset...")

    # Drop internal columns not needed for ML
    drop_cols = ["FirstActivationDate"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Replace inf values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill all feature NaNs with 0
    feat_cols = [c for c in df.columns if c.startswith("Feat_")]
    df[feat_cols] = df[feat_cols].fillna(0)

    # Final sort
    df = df.sort_values(["ETLDatabase", "SITE_ID", "Year", "Quarter"]).reset_index(drop=True)

    # Summary stats
    n_total = len(df)
    n_zero = (df["Target_QuarterlyEnrollment"] == 0).sum()
    zero_pct = n_zero / n_total * 100

    logger.info("=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total rows:          {n_total:,}")
    logger.info(f"Zero enrollment:     {n_zero:,} ({zero_pct:.1f}%)")
    logger.info(f"Non-zero enrollment: {n_total - n_zero:,} ({100 - zero_pct:.1f}%)")
    logger.info(f"Studies:             {df['ETLDatabase'].nunique():,}")
    logger.info(f"Sites:               {df['SITE_ID'].nunique():,}")
    if "COUNTRY_NAME" in df.columns:
        logger.info(f"Countries:           {df['COUNTRY_NAME'].nunique():,}")
    logger.info(f"Year range:          {df['Year'].min()} – {df['Year'].max()}")
    logger.info(f"Feature columns:     {len(feat_cols)}")
    logger.info(f"Total columns:       {len(df.columns)}")
    logger.info("=" * 60)

    # Save
    FEATURE_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURE_DATASET_PATH, index=False, engine="pyarrow")
    df.to_csv(FEATURE_DATASET_CSV, index=False)
    logger.info(f"✅ Parquet saved: {FEATURE_DATASET_PATH}")
    logger.info(f"✅ CSV saved:     {FEATURE_DATASET_CSV}")
    return df


# ===========================================================================
# MAIN
# ===========================================================================
def build_feature_dataset():
    start = time.time()
    logger.info("=" * 60)
    logger.info("  PHASE 1: Building Feature Dataset from CSVs")
    logger.info("=" * 60)

    data = load_all_csvs()
    spine = build_quarter_spine(
        data["enrollments"],
        data.get("screenings", pd.DataFrame()),
        data.get("supply_spine", pd.DataFrame())
    )
    df = merge_all_features(spine, data)
    df = compute_base_features(df)
    df = add_timeseries_features(df)
    df = add_hierarchical_features(df)
    df = add_engineered_features(df)
    df = finalize_and_save(df)

    logger.info(f"\n✅ Complete in {time.time() - start:.1f}s")
    return df


if __name__ == "__main__":
    df = build_feature_dataset()

    print("\n--- FEATURE COLUMNS ---")
    for c in [col for col in df.columns if col.startswith("Feat_")]:
        print(f"  {c}")

    print("\n--- SAMPLE (non-zero enrollment) ---")
    sample = df[df["Target_QuarterlyEnrollment"] > 0].head(5)
    display_cols = ["ETLDatabase", "SITE_ID", "Year", "Quarter",
                    "Target_QuarterlyEnrollment", "Feat_EnrollLag1Q",
                    "Feat_EnrollMA2Q", "Feat_PrevQuarterWasZero"]
    print(sample[[c for c in display_cols if c in df.columns]].to_string(index=False))