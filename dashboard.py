"""
dashboard.py -- Phase 4: Enrollment Forecasting Dashboard
Clinical Trial Enrollment Forecasting Platform
Karan Arya | MSc AI Batch 10 | Endpoint Clinical

Capstone 1 scope: Tab 1 -- Enrollment Forecast only
  - P10/P50/P90 fan chart over 2024 quarters
  - Cumulative enrollment chart over 2024 quarters
  - Drill-down by Sponsor -> Study -> Country -> Region
  - KPI cards: total P50, actual enrolled, accuracy, sites
  - Actual vs predicted overlay
  - Site-level detail table

Fixes vs previous version:
  - ADDED: build_cumulative_chart() was referenced nowhere in the file --
    it did not exist. Added the function and the call for it between the
    fan chart and the Quarterly Breakdown table.
  - FIX: page_icon="favicon.ico" pointed to a file that doesn't exist.
    Changed to an emoji so there is no missing-file dependency.
  - FIX: COUNTRY_NAME / REGION_NAME nulls (14% of rows) were silently
    dropped by groupby() when drilling down by Country/Region, because
    pandas excludes NaN groups by default. Nulls are now filled with
    "Unknown" at load time and dropna=False is used in the relevant
    groupby calls, so no rows disappear silently. "Unknown" is excluded
    from the selectable dropdown options.
  - FIX: removed a redundant duplicate assignment of agg["Quarter_Label"]
    (it was set once inside aggregate_to_quarter and then overwritten
    again immediately after the call for no reason).
  - ADDED: a short annotation on the fan chart explaining the Q4 2024
    site-count drop, so it doesn't read as a model failure.

Run with:
  streamlit run dashboard.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config -- must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Enrollment Forecasting Platform",
    page_icon="📊",          # was "favicon.ico" -- that file doesn't exist
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parent
DATA_PATH = ROOT / "data" / "uncertainty_predictions.parquet"
CSV_PATH  = ROOT / "data" / "uncertainty_predictions.csv"

# ---------------------------------------------------------------------------
# Styling -- soft clinical theme with readable colours
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Soft light background */
.stApp {
    background: linear-gradient(160deg, #f0f4f8 0%, #e8eef5 50%, #f5f7fa 100%);
    color: #1e293b;
}

/* Sidebar - soft navy-blue gradient */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a5f 0%, #1a2f4a 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stDivider {
    border-color: rgba(255,255,255,0.1) !important;
}

/* Sidebar selectbox styling - fix white dropdown readability */
section[data-testid="stSidebar"] div[data-baseweb="select"] {
    background-color: rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
    fill: #94a3b8 !important;
}
section[data-testid="stSidebar"] input {
    background-color: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* Dropdown menu when opened */
div[data-baseweb="popover"] {
    border-radius: 8px !important;
    overflow: hidden !important;
}
div[data-baseweb="popover"] ul {
    background: #1e3a5f !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
}
div[data-baseweb="popover"] li {
    color: #e2e8f0 !important;
}
div[data-baseweb="popover"] li:hover {
    background: rgba(96, 165, 250, 0.2) !important;
}
div[data-baseweb="popover"] li[aria-selected="true"] {
    background: rgba(96, 165, 250, 0.3) !important;
}

/* Sidebar header */
.sidebar-header {
    padding: 0 0 24px 0;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 20px;
}
.sidebar-logo {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #60a5fa !important;
}
.sidebar-subtitle {
    font-size: 11px;
    color: #94a3b8 !important;
    margin-top: 4px;
    letter-spacing: 0.04em;
}

/* Filter section labels */
.filter-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 6px;
    margin-top: 16px;
}

/* KPI cards */
.kpi-row {
    display: flex;
    gap: 14px;
    margin-bottom: 24px;
}
.kpi-card {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
    transition: box-shadow 0.2s ease;
}
.kpi-card:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.06), 0 8px 24px rgba(0,0,0,0.05);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}
.kpi-card.blue::before  { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.kpi-card.green::before { background: linear-gradient(90deg, #10b981, #34d399); }
.kpi-card.amber::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.kpi-card.purple::before{ background: linear-gradient(90deg, #8b5cf6, #a78bfa); }

.kpi-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 8px;
}
.kpi-value {
    font-family: 'DM Mono', monospace;
    font-size: 28px;
    font-weight: 500;
    color: #1e293b;
    line-height: 1;
}
.kpi-delta {
    font-size: 11px;
    margin-top: 6px;
    color: #64748b;
}
.kpi-delta.positive { color: #059669; }
.kpi-delta.negative { color: #dc2626; }
.kpi-delta.neutral  { color: #64748b; }

/* Page title */
.page-title {
    font-size: 22px;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 4px;
    letter-spacing: -0.02em;
}
.page-subtitle {
    font-size: 13px;
    color: #64748b;
    margin-bottom: 24px;
}
.breadcrumb {
    font-size: 11px;
    color: #475569;
    font-family: 'DM Mono', monospace;
    margin-bottom: 20px;
    padding: 6px 12px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    display: inline-block;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.breadcrumb span { color: #3b82f6; }

/* Section headers */
.section-header {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 8px;
    margin-bottom: 16px;
    margin-top: 28px;
}

/* Coverage badge */
.coverage-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
}
.coverage-good  { background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; }
.coverage-warn  { background: #fffbeb; color: #d97706; border: 1px solid #fde68a; }
.coverage-poor  { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }

/* Streamlit overrides */
div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label { display: none; }

/* Main area dataframes */
.stDataFrame {
    background: #ffffff;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
}

/* Main content containers */
div[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}

/* Tab styling */
button[data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #475569 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #3b82f6 !important;
}

/* Toggle in sidebar */
section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] span {
    color: #cbd5e1 !important;
    font-size: 12px !important;
}

/* Warning/info boxes */
.info-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 3px solid #3b82f6;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 12px;
    color: #475569;
    margin-bottom: 16px;
}
.warn-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 3px solid #f59e0b;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 12px;
    color: #475569;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_parquet(DATA_PATH)
    elif CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
    else:
        st.error(f"Data file not found. Expected: {DATA_PATH}")
        st.stop()

    # FIX: fill null geographic columns so groupby doesn't silently drop
    # 14% of rows (~8,568) when drilling down by Country/Region.
    df["COUNTRY_NAME"] = df["COUNTRY_NAME"].fillna("Unknown")
    df["REGION_NAME"]  = df["REGION_NAME"].fillna("Unknown")

    return df


@st.cache_data
def aggregate_to_quarter(
    df: pd.DataFrame,
    group_cols: list,
) -> pd.DataFrame:
    """Aggregate site-level predictions up to any grouping level."""
    agg = df.groupby(group_cols, dropna=False).agg(
        P10=("P10", "sum"),
        P50=("P50", "sum"),
        P90=("P90", "sum"),
        Final_Pred=("Final_Pred", "sum"),
        Actual=("Target_QuarterlyEnrollment", "sum"),
        Sites=("SITE_ID", "nunique"),
        HighUncSites=("HighUncertainty", "sum"),
    ).reset_index()

    if "Year" in agg.columns and "Quarter" in agg.columns:
        agg["Quarter_Label"] = "Q" + agg["Quarter"].astype(str) + " " + agg["Year"].astype(str)
        agg = agg.sort_values(["Year", "Quarter"]).reset_index(drop=True)

    return agg


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
CHART_COLORS = {
    "band":    "rgba(59, 130, 246, 0.10)",
    "p50":     "#3b82f6",
    "actual":  "#10b981",
    "pred":    "#8b5cf6",
    "p10_line":"rgba(59, 130, 246, 0.35)",
    "p90_line":"rgba(59, 130, 246, 0.35)",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(255,255,255,0)",
    plot_bgcolor="rgba(255,255,255,0.6)",
    font=dict(family="DM Sans, sans-serif", color="#475569", size=12),
    xaxis=dict(
        gridcolor="#e2e8f0",
        linecolor="#cbd5e1",
        tickcolor="#94a3b8",
        showgrid=True,
    ),
    yaxis=dict(
        gridcolor="#e2e8f0",
        linecolor="#cbd5e1",
        tickcolor="#94a3b8",
        showgrid=True,
        zeroline=False,
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#e2e8f0",
        borderwidth=1,
        font=dict(size=11, color="#475569"),
    ),
    margin=dict(l=0, r=0, t=10, b=0),
    hovermode="x unified",
)


def build_fan_chart(
    agg: pd.DataFrame,
    title: str = "",
    show_actual: bool = True,
) -> go.Figure:
    """P10/P50/P90 fan chart with optional actual overlay."""
    fig = go.Figure()

    x = agg["Quarter_Label"].tolist()

    # P10-P90 shaded band
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=agg["P90"].tolist() + agg["P10"].tolist()[::-1],
        fill="toself",
        fillcolor=CHART_COLORS["band"],
        line=dict(color="rgba(0,0,0,0)"),
        name="P10-P90 Band",
        hoverinfo="skip",
        showlegend=True,
    ))

    # P10 line
    fig.add_trace(go.Scatter(
        x=x, y=agg["P10"],
        mode="lines",
        line=dict(color=CHART_COLORS["p10_line"], width=1, dash="dot"),
        name="P10",
        hovertemplate="P10: %{y:,}<extra></extra>",
    ))

    # P90 line
    fig.add_trace(go.Scatter(
        x=x, y=agg["P90"],
        mode="lines",
        line=dict(color=CHART_COLORS["p90_line"], width=1, dash="dot"),
        name="P90",
        hovertemplate="P90: %{y:,}<extra></extra>",
    ))

    # P50 (median forecast)
    fig.add_trace(go.Scatter(
        x=x, y=agg["P50"],
        mode="lines+markers",
        line=dict(color=CHART_COLORS["p50"], width=2.5),
        marker=dict(size=6, color=CHART_COLORS["p50"],
                    line=dict(color="#0d1117", width=1.5)),
        name="P50 Forecast",
        hovertemplate="P50: %{y:,}<extra></extra>",
    ))

    # Actual (if available and requested)
    if show_actual and "Actual" in agg.columns:
        fig.add_trace(go.Scatter(
            x=x, y=agg["Actual"],
            mode="lines+markers",
            line=dict(color=CHART_COLORS["actual"], width=2),
            marker=dict(size=6, symbol="diamond",
                        color=CHART_COLORS["actual"],
                        line=dict(color="#0d1117", width=1.5)),
            name="Actual",
            hovertemplate="Actual: %{y:,}<extra></extra>",
        ))

    # ADDED: annotate the Q4 2024 site-count drop so it doesn't read as a
    # model failure (27 sponsors have no Q4 data in the extract).
    if len(x) >= 4:
        q4_label = x[-1]
        q4_p50   = float(agg["P50"].iloc[-1])
        fig.add_annotation(
            x=q4_label,
            y=q4_p50,
            text="Q4: fewer sites<br>(data completeness)",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#94a3b8",
            arrowsize=0.8,
            ax=55, ay=-40,
            font=dict(size=10, color="#64748b"),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        )

    layout = {**PLOTLY_LAYOUT}
    layout["yaxis"]["title"] = "Patients Enrolled"
    layout["height"] = 360

    fig.update_layout(**layout)
    return fig


def build_cumulative_chart(agg: pd.DataFrame, show_actual: bool = True) -> go.Figure:
    """
    ADDED (was missing entirely): Cumulative enrollment over 2024.
    Shows running-total P10/P50/P90 and actual, which is what sponsors
    care about for milestone planning rather than the per-quarter figure.
    """
    fig = go.Figure()
    x = agg["Quarter_Label"].tolist()

    cum_p10 = agg["P10"].cumsum().tolist()
    cum_p50 = agg["P50"].cumsum().tolist()
    cum_p90 = agg["P90"].cumsum().tolist()
    cum_actual = agg["Actual"].cumsum().tolist() if show_actual else None

    # Shaded cumulative band
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=cum_p90 + cum_p10[::-1],
        fill="toself",
        fillcolor=CHART_COLORS["band"],
        line=dict(color="rgba(0,0,0,0)"),
        name="Cumulative P10-P90",
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=x, y=cum_p50,
        mode="lines+markers",
        line=dict(color=CHART_COLORS["p50"], width=2.5),
        marker=dict(size=6, color=CHART_COLORS["p50"],
                    line=dict(color="#0d1117", width=1.5)),
        name="Cumulative P50",
        hovertemplate="Cumulative P50: %{y:,}<extra></extra>",
    ))

    if cum_actual:
        fig.add_trace(go.Scatter(
            x=x, y=cum_actual,
            mode="lines+markers",
            line=dict(color=CHART_COLORS["actual"], width=2),
            marker=dict(size=6, symbol="diamond", color=CHART_COLORS["actual"],
                        line=dict(color="#0d1117", width=1.5)),
            name="Cumulative Actual",
            hovertemplate="Cumulative Actual: %{y:,}<extra></extra>",
        ))

    layout = {**PLOTLY_LAYOUT}
    layout["yaxis"]["title"] = "Cumulative Patients"
    layout["height"] = 300
    layout["margin"] = dict(l=0, r=0, t=6, b=0)
    fig.update_layout(**layout)
    return fig


def build_site_bar(site_agg: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of top sites by P50 total enrollment."""
    site_total = site_agg.groupby("SITE_ID").agg(
        P50=("P50", "sum"),
        Actual=("Actual", "sum"),
        Study=("ETLDatabase", "first"),
    ).reset_index().nlargest(top_n, "P50")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=site_total["SITE_ID"].astype(str),
        x=site_total["Actual"],
        orientation="h",
        name="Actual",
        marker_color=CHART_COLORS["actual"],
        opacity=0.75,
    ))
    fig.add_trace(go.Bar(
        y=site_total["SITE_ID"].astype(str),
        x=site_total["P50"],
        orientation="h",
        name="P50 Forecast",
        marker_color=CHART_COLORS["p50"],
        opacity=0.8,
    ))

    layout = {**PLOTLY_LAYOUT}
    layout["barmode"] = "overlay"
    layout["height"] = max(300, top_n * 22)
    layout["xaxis"]["title"] = "Patients"
    layout["yaxis"]["title"] = ""
    layout["margin"] = dict(l=60, r=0, t=10, b=0)
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def coverage_html(coverage_pct: float) -> str:
    if coverage_pct >= 80:
        cls = "coverage-good"
    elif coverage_pct >= 65:
        cls = "coverage-warn"
    else:
        cls = "coverage-poor"
    return f'<span class="coverage-badge {cls}">{coverage_pct:.1f}% coverage</span>'


def compute_coverage(df_sub: pd.DataFrame) -> float:
    y = df_sub["Target_QuarterlyEnrollment"]
    return float(((y >= df_sub["P10"]) & (y <= df_sub["P90"])).mean() * 100)


def compute_accuracy(actual: float, pred: float) -> float:
    if actual == 0:
        return 0.0
    return max(0.0, (1 - abs(actual - pred) / actual) * 100)


def delta_html(actual: int, pred: int) -> str:
    if actual == 0:
        return '<span class="kpi-delta neutral">no actuals to compare</span>'
    diff = actual - pred
    pct  = diff / actual * 100
    sign = "+" if diff >= 0 else ""
    cls  = "positive" if diff >= 0 else "negative"
    return f'<span class="kpi-delta {cls}">{sign}{diff:,} vs forecast ({sign}{pct:.1f}%)</span>'


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(df: pd.DataFrame) -> dict:
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <div class="sidebar-logo">Endpoint Clinical</div>
            <div class="sidebar-subtitle">Enrollment Forecasting Platform</div>
        </div>
        """, unsafe_allow_html=True)

        # --- Level selector ---
        st.markdown('<div class="filter-label">View Level</div>', unsafe_allow_html=True)
        level = st.selectbox(
            "level", ["Sponsor", "Study", "Country", "Region"],
            label_visibility="collapsed",
        )

        filters = {"level": level}

        # --- Sponsor filter ---
        st.markdown('<div class="filter-label">Sponsor</div>', unsafe_allow_html=True)
        sponsors = sorted(df["Sponsor"].dropna().unique().tolist())
        selected_sponsor = st.selectbox(
            "sponsor", ["All Sponsors"] + sponsors,
            label_visibility="collapsed",
        )
        filters["sponsor"] = selected_sponsor

        # --- Study filter (cascades from sponsor) ---
        if selected_sponsor != "All Sponsors":
            sponsor_df = df[df["Sponsor"] == selected_sponsor]
        else:
            sponsor_df = df

        st.markdown('<div class="filter-label">Study</div>', unsafe_allow_html=True)
        studies = sorted(sponsor_df["ETLDatabase"].unique().tolist())
        selected_study = st.selectbox(
            "study", ["All Studies"] + studies,
            label_visibility="collapsed",
        )
        filters["study"] = selected_study

        # --- Country filter ---
        # FIX: exclude the "Unknown" fill value from the pickable list --
        # it's a placeholder for missing geo data, not a real country.
        st.markdown('<div class="filter-label">Country</div>', unsafe_allow_html=True)
        countries = sorted([c for c in df["COUNTRY_NAME"].unique() if c != "Unknown"])
        selected_country = st.selectbox(
            "country", ["All Countries"] + countries,
            label_visibility="collapsed",
        )
        filters["country"] = selected_country

        # --- Region filter ---
        st.markdown('<div class="filter-label">Region</div>', unsafe_allow_html=True)
        regions = sorted([r for r in df["REGION_NAME"].unique() if r != "Unknown"])
        selected_region = st.selectbox(
            "region", ["All Regions"] + regions,
            label_visibility="collapsed",
        )
        filters["region"] = selected_region

        st.divider()

        # --- Show actual toggle ---
        st.markdown('<div class="filter-label">Chart Options</div>', unsafe_allow_html=True)
        show_actual = st.toggle("Show actual enrollment", value=True)
        filters["show_actual"] = show_actual

        st.divider()

        # --- Data info ---
        st.markdown(f"""
        <div style="font-size:11px; color:#8b949e; line-height:1.8;">
            <div><b style="color:#e6edf3;">Dataset</b></div>
            <div>Period: 2024 (Q1-Q4)</div>
            <div>Studies: {df['ETLDatabase'].nunique():,}</div>
            <div>Sponsors: {df['Sponsor'].nunique():,}</div>
            <div>Sites: {df['SITE_ID'].nunique():,}</div>
            <div style="margin-top:8px;"><b style="color:#e6edf3;">Model</b></div>
            <div>Two-stage XGBoost</div>
            <div>Stage 1: Classifier (F1=0.957)</div>
            <div>Stage 2: Poisson Reg (R²=0.851)</div>
            <div>Uncertainty: MC + Quantile</div>
        </div>
        """, unsafe_allow_html=True)

    return filters


# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------
def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    sub = df.copy()
    if filters["sponsor"] != "All Sponsors":
        sub = sub[sub["Sponsor"] == filters["sponsor"]]
    if filters["study"] != "All Studies":
        sub = sub[sub["ETLDatabase"] == filters["study"]]
    if filters["country"] != "All Countries":
        sub = sub[sub["COUNTRY_NAME"] == filters["country"]]
    if filters["region"] != "All Regions":
        sub = sub[sub["REGION_NAME"] == filters["region"]]
    return sub


def build_breadcrumb(filters: dict) -> str:
    parts = []
    if filters["sponsor"] != "All Sponsors":
        parts.append(f'<span>{filters["sponsor"]}</span>')
    if filters["study"] != "All Studies":
        parts.append(f'<span>{filters["study"]}</span>')
    if filters["country"] != "All Countries":
        parts.append(f'<span>{filters["country"]}</span>')
    if filters["region"] != "All Regions":
        parts.append(f'<span>{filters["region"]}</span>')
    if not parts:
        return '<div class="breadcrumb">All Sponsors / All Studies / All Countries</div>'
    return '<div class="breadcrumb">' + ' &rsaquo; '.join(parts) + '</div>'


# ---------------------------------------------------------------------------
# Main Tab 1 render
# ---------------------------------------------------------------------------
def render_tab1(df: pd.DataFrame, filtered: pd.DataFrame, filters: dict):

    # Title + breadcrumb
    st.markdown('<div class="page-title">Enrollment Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">P10 / P50 / P90 quarterly forecast with actual enrollment overlay</div>', unsafe_allow_html=True)
    st.markdown(build_breadcrumb(filters), unsafe_allow_html=True)

    if len(filtered) == 0:
        st.warning("No data matches the current filters.")
        return

    # -------------------------------------------------------
    # Aggregate to quarter level for charts
    # -------------------------------------------------------
    agg = aggregate_to_quarter(filtered, ["Year", "Quarter"])
    # NOTE: removed the redundant duplicate reassignment of
    # agg["Quarter_Label"] that was here before -- it's already set
    # inside aggregate_to_quarter().

    total_actual = int(agg["Actual"].sum())
    total_p50    = int(agg["P50"].sum())
    total_p10    = int(agg["P10"].sum())
    total_p90    = int(agg["P90"].sum())
    total_sites  = int(filtered["SITE_ID"].nunique())
    coverage     = compute_coverage(filtered)
    accuracy     = compute_accuracy(total_actual, total_p50)
    hu_sites     = int(filtered[filtered["HighUncertainty"] == True]["SITE_ID"].nunique())

    # -------------------------------------------------------
    # KPI cards
    # -------------------------------------------------------
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card blue">
            <div class="kpi-label">P50 Forecast (2024)</div>
            <div class="kpi-value">{total_p50:,}</div>
            <div class="kpi-delta neutral">P10: {total_p10:,} &mdash; P90: {total_p90:,}</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-label">Actual Enrolled (2024)</div>
            <div class="kpi-value">{total_actual:,}</div>
            {delta_html(total_actual, total_p50)}
        </div>
        <div class="kpi-card amber">
            <div class="kpi-label">Forecast Accuracy</div>
            <div class="kpi-value">{accuracy:.1f}%</div>
            <div class="kpi-delta neutral">{coverage_html(coverage)}</div>
        </div>
        <div class="kpi-card purple">
            <div class="kpi-label">Active Sites</div>
            <div class="kpi-value">{total_sites:,}</div>
            <div class="kpi-delta {'negative' if hu_sites > 0 else 'neutral'}">{hu_sites} high-uncertainty sites</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------
    # Fan chart
    # -------------------------------------------------------
    st.markdown('<div class="section-header">Quarterly Enrollment Forecast</div>', unsafe_allow_html=True)

    # Note about null-geography rows if a country/region filter is active
    if filters["country"] != "All Countries" or filters["region"] != "All Regions":
        unknown_count = (filtered["COUNTRY_NAME"] == "Unknown").sum()
        if unknown_count > 0:
            st.markdown(
                f'<div class="info-box">{unknown_count:,} site-quarter rows have no geographic '
                f'mapping and are excluded from this country/region view. '
                f'Sponsor and study-level views include all sites.</div>',
                unsafe_allow_html=True,
            )

    fan_fig = build_fan_chart(agg, show_actual=filters["show_actual"])
    st.plotly_chart(fan_fig, use_container_width=True, config={"displayModeBar": False})

    # -------------------------------------------------------
    # Cumulative chart (ADDED -- this was missing entirely before)
    # -------------------------------------------------------
    st.markdown('<div class="section-header">Cumulative Enrollment (2024)</div>', unsafe_allow_html=True)
    cum_fig = build_cumulative_chart(agg, show_actual=filters["show_actual"])
    st.plotly_chart(cum_fig, use_container_width=True, config={"displayModeBar": False})

    # -------------------------------------------------------
    # Quarterly breakdown table
    # -------------------------------------------------------
    st.markdown('<div class="section-header">Quarterly Breakdown</div>', unsafe_allow_html=True)

    display_agg = agg[["Quarter_Label", "P10", "P50", "P90",
                        "Actual", "Final_Pred", "Sites"]].copy()
    display_agg.columns = ["Quarter", "P10", "P50 (Median)", "P90",
                           "Actual", "Point Pred", "Sites"]

    if filters["show_actual"]:
        display_agg["Accuracy"] = display_agg.apply(
            lambda r: f"{compute_accuracy(r['Actual'], r['P50 (Median)']):.1f}%", axis=1
        )
        display_agg["vs Forecast"] = display_agg.apply(
            lambda r: f"{r['Actual'] - r['P50 (Median)']:+,}", axis=1
        )

    st.dataframe(
        display_agg,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Quarter":      st.column_config.TextColumn("Quarter"),
            "P10":          st.column_config.NumberColumn("P10", format="%d"),
            "P50 (Median)": st.column_config.NumberColumn("P50", format="%d"),
            "P90":          st.column_config.NumberColumn("P90", format="%d"),
            "Actual":       st.column_config.NumberColumn("Actual", format="%d"),
            "Point Pred":   st.column_config.NumberColumn("Point Pred", format="%d"),
            "Sites":        st.column_config.NumberColumn("Sites"),
        }
    )

    # -------------------------------------------------------
    # Drill-down: sub-level view
    # -------------------------------------------------------
    level = filters["level"]
    level_col_map = {
        "Sponsor":  "Sponsor",
        "Study":    "ETLDatabase",
        "Country":  "COUNTRY_NAME",
        "Region":   "REGION_NAME",
    }
    drill_col = level_col_map[level]

    st.markdown(f'<div class="section-header">Breakdown by {level}</div>', unsafe_allow_html=True)

    # FIX: dropna=False so Country/Region drill-downs don't silently
    # drop the rows filled as "Unknown" from missing geo data.
    drill_agg = filtered.groupby(drill_col, dropna=False).agg(
        P10=("P10", "sum"),
        P50=("P50", "sum"),
        P90=("P90", "sum"),
        Actual=("Target_QuarterlyEnrollment", "sum"),
        Sites=("SITE_ID", "nunique"),
        HighUnc=("HighUncertainty", "sum"),
    ).reset_index()

    drill_agg = drill_agg.sort_values("P50", ascending=False).head(30)
    drill_agg["Accuracy"] = drill_agg.apply(
        lambda r: f"{compute_accuracy(r['Actual'], r['P50']):.1f}%" if r["Actual"] > 0 else "—",
        axis=1,
    )
    drill_agg["Band"] = drill_agg.apply(
        lambda r: f"{r['P10']:,} – {r['P90']:,}", axis=1
    )

    display_drill = drill_agg[[drill_col, "P50", "Band", "Actual",
                                "Accuracy", "Sites", "HighUnc"]].copy()
    display_drill.columns = [level, "P50", "P10–P90 Band",
                              "Actual", "Accuracy", "Sites", "High Unc Sites"]

    st.dataframe(
        display_drill,
        use_container_width=True,
        hide_index=True,
        column_config={
            level:              st.column_config.TextColumn(level),
            "P50":              st.column_config.NumberColumn("P50 Forecast", format="%d"),
            "P10–P90 Band":     st.column_config.TextColumn("P10–P90 Band"),
            "Actual":           st.column_config.NumberColumn("Actual", format="%d"),
            "Accuracy":         st.column_config.TextColumn("Accuracy"),
            "Sites":            st.column_config.NumberColumn("Sites"),
            "High Unc Sites":   st.column_config.NumberColumn("High Unc Sites"),
        }
    )

    # -------------------------------------------------------
    # Site-level detail (collapsed by default)
    # -------------------------------------------------------
    with st.expander("Site-level detail", expanded=False):
        site_agg = aggregate_to_quarter(filtered, ["SITE_ID", "ETLDatabase",
                                                    "Sponsor", "Year", "Quarter"])

        # Top sites bar chart
        st.markdown('<div class="section-header" style="margin-top:8px;">Top Sites by P50 Forecast</div>',
                    unsafe_allow_html=True)
        bar_fig = build_site_bar(site_agg, top_n=20)
        st.plotly_chart(bar_fig, use_container_width=True,
                        config={"displayModeBar": False})

        # Site table
        site_total = filtered.groupby(["SITE_ID", "ETLDatabase", "Sponsor"]).agg(
            P10=("P10", "sum"),
            P50=("P50", "sum"),
            P90=("P90", "sum"),
            Actual=("Target_QuarterlyEnrollment", "sum"),
            HighUnc=("HighUncertainty", "any"),
        ).reset_index().sort_values("P50", ascending=False)

        site_total["Accuracy"] = site_total.apply(
            lambda r: f"{compute_accuracy(r['Actual'], r['P50']):.1f}%" if r["Actual"] > 0 else "—",
            axis=1,
        )

        st.dataframe(
            site_total.rename(columns={
                "SITE_ID": "Site ID", "ETLDatabase": "Study",
                "HighUnc": "High Uncertainty",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # -------------------------------------------------------
    # Limitation callout
    # -------------------------------------------------------
    st.markdown("""
    <div class="warn-box" style="margin-top: 24px;">
        <b>Model limitations:</b>
        Sites enrolling 28+ patients/quarter are underestimated by design
        (training target capped at 99th percentile = 27 pts/quarter).
        P10-P90 bands are narrow for low-enrollment sites (median width = 1 pt)
        and coverage degrades above 10 pts/quarter. See uncertainty tab for full analysis.
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------
def main():
    df = load_data()

    filters  = render_sidebar(df)
    filtered = apply_filters(df, filters)

    # Single tab for Capstone 1 -- placeholder structure for Capstone 2
    tab1, = st.tabs(["Enrollment Forecast"])

    with tab1:
        render_tab1(df, filtered, filters)


if __name__ == "__main__":
    main()