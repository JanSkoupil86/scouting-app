# pages/5_Player_Screening.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Utilities
# -----------------------------
def _first_existing_df_from_session(keys: list[str]) -> pd.DataFrame | None:
    for k in keys:
        v = st.session_state.get(k, None)
        if isinstance(v, pd.DataFrame) and not v.empty:
            return v
    return None


def _numeric_metric_candidates(df: pd.DataFrame, exclude_cols: set[str]) -> list[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    metrics = [c for c in numeric_cols if c not in exclude_cols]
    return sorted(metrics)


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# -----------------------------
# Player Screening Feature
# -----------------------------
def render_player_screening(
    df: pd.DataFrame,
    *,
    league_col: str,
    position_col: str,
    age_col: str,
    minutes_col: str,
    player_col: str,
    team_col: str,
    season_col: str | None = None,  # optional
    exclude_cols: set[str] | None = None,
    default_metrics: list[str] | None = None,
):
    st.markdown(
        "Screen players by requiring percentile ranges for selected metrics.\n\n"
        "**Percentiles are computed within the selected cohort** "
        "(League(s) + Position + Age filter + Minutes filter)."
    )

    # ---- required columns check
    required = [league_col, position_col, age_col, minutes_col]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        st.error(f"Missing required columns in dataset: {missing_required}")
        st.stop()

    # ---- exclude cols default
    if exclude_cols is None:
        exclude_cols = {league_col, position_col, age_col, minutes_col}
        if player_col in df.columns:
            exclude_cols.add(player_col)
        if team_col in df.columns:
            exclude_cols.add(team_col)
        if season_col and season_col in df.columns:
            exclude_cols.add(season_col)

    # ---- metric candidates (numeric only)
    metric_candidates = _numeric_metric_candidates(df, exclude_cols)
    if not metric_candidates:
        st.warning("No numeric metrics available for screening (after exclusions).")
        st.stop()

    # ---- default metrics
    if default_metrics is None:
        preferred = [
            "xG per 90",
            "xA per 90",
            "Shots per 90",
            "Key passes per 90",
            "Progressive passes per 90",
            "Progressive runs per 90",
        ]
        default_metrics = [m for m in preferred if m in metric_candidates][:6]
        if not default_metrics:
            default_metrics = metric_candidates[:6]

    # =========================
    # Filters
    # =========================
    st.subheader("Filters")

    leagues = sorted([x for x in df[league_col].dropna().unique().tolist()])
    positions = sorted([x for x in df[position_col].dropna().unique().tolist()])

    c1, c2, c3 = st.columns([2, 2, 2])

    with c1:
        sel_leagues = st.multiselect(
            "League(s)",
            options=leagues,
            default=leagues[:1] if leagues else [],
            key="ps_leagues",
        )

    with c2:
        sel_position = st.selectbox(
            "Position",
            options=positions,
            index=0 if positions else 0,
            key="ps_position",
        )

    # Age slider bounds (robust)
    age_series = pd.to_numeric(df[age_col], errors="coerce")
    if age_series.notna().any():
        age_min = int(np.nanmin(age_series.values))
        age_max = int(np.nanmax(age_series.values))
    else:
        age_min, age_max = 0, 60

    with c3:
        age_range = st.slider(
            "Age range",
            min_value=age_min,
            max_value=age_max,
            value=(age_min, age_max),
            step=1,
            key="ps_age_range",
        )

    # Minutes slider bounds (robust)
    min_series = pd.to_numeric(df[minutes_col], errors="coerce")
    if min_series.notna().any():
        min_min = int(np.nanmin(min_series.values))
        min_max = int(np.nanmax(min_series.values))
    else:
        min_min, min_max = 0, 0

    min_minutes_default = 400 if min_max >= 400 else max(0, min_max)
    min_minutes = st.slider(
        "Minimum Minutes",
        min_value=min_min,
        max_value=min_max,
        value=min_minutes_default,
        step=10 if min_max >= 10 else 1,
        key="ps_min_minutes",
    )

    # ---- apply base cohort filters
    cohort = df.copy()

    if sel_leagues:
        cohort = cohort[cohort[league_col].isin(sel_leagues)]

    if sel_position in positions:
        cohort = cohort[cohort[position_col] == sel_position]

    cohort[age_col] = pd.to_numeric(cohort[age_col], errors="coerce")
    cohort[minutes_col] = pd.to_numeric(cohort[minutes_col], errors="coerce")

    cohort = cohort[
        cohort[age_col].between(age_range[0], age_range[1], inclusive="both")
        & (cohort[minutes_col] >= float(min_minutes))
    ].copy()

    st.caption(
        f"Cohort size: {len(cohort):,} | "
        f"Leagues: {len(sel_leagues)} | "
        f"Position: {sel_position} | "
        f"Age: {age_range[0]}–{age_range[1]} | "
        f"Minutes ≥ {min_minutes}"
    )

    if cohort.empty:
        st.warning("No players in cohort after filters. Adjust filters and try again.")
        st.stop()

    # =========================
    # Metrics & thresholds
    # =========================
    st.subheader("Metrics & Screening thresholds")

    sel_metrics = st.multiselect(
        "Select metrics for screening",
        options=metric_candidates,
        default=[m for m in default_metrics if m in metric_candidates],
        key="ps_metrics",
    )

    if not sel_metrics:
        st.info("Select at least one metric to screen players.")
        st.stop()

    st.markdown("**Percentile ranges (applied to each metric):**")

    # Persist thresholds per metric across reruns
    if "ps_thresholds" not in st.session_state:
        st.session_state.ps_thresholds = {}

    DEFAULT_LOW, DEFAULT_HIGH = 30, 100
    thresholds: dict[str, tuple[int, int]] = {}

    for metric in sel_metrics:
        if metric not in st.session_state.ps_thresholds:
            st.session_state.ps_thresholds[metric] = (DEFAULT_LOW, DEFAULT_HIGH)

        low0, high0 = st.session_state.ps_thresholds[metric]

        low, high = st.slider(
            metric,
            min_value=0,
            max_value=100,
            value=(int(low0), int(high0)),
            step=1,
            key=f"ps_thr__{metric}",  # stable per metric
        )

        st.session_state.ps_thresholds[metric] = (low, high)
        thresholds[metric] = (low, high)

    # =========================
    # Percentile computation (within cohort)
    # =========================
    pct_df = pd.DataFrame(index=cohort.index)

    for metric in sel_metrics:
        s = pd.to_numeric(cohort[metric], errors="coerce")
        pct = s.rank(pct=True, method="average") * 100.0
        pct_df[f"{metric} pctl"] = pct

    # =========================
    # Apply AND screening across metrics
    # =========================
    mask = pd.Series(True, index=cohort.index)

    for metric in sel_metrics:
        low, high = thresholds[metric]
        pcol = f"{metric} pctl"
        mask &= pct_df[pcol].between(low, high, inclusive="both") & pct_df[pcol].notna()

    screened = cohort.loc[mask].copy()
    screened = screened.join(pct_df, how="left")

    if screened.empty:
        st.warning("No players met all thresholds. Widen percentile ranges or change metrics.")
        st.stop()

    # =========================
    # Output table
    # =========================
    base_cols = []
    for c in [player_col, team_col, league_col, position_col, age_col, minutes_col]:
        if c in screened.columns:
            base_cols.append(c)

    raw_metric_cols = sel_metrics
    pctl_cols = [f"{m} pctl" for m in sel_metrics]

    screened["Avg pctl"] = screened[pctl_cols].mean(axis=1)
    display_cols = base_cols + ["Avg pctl"] + raw_metric_cols + pctl_cols

    screened = screened.sort_values("Avg pctl", ascending=False)

    st.dataframe(
        screened[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    csv = screened[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download screened players (CSV)",
        data=csv,
        file_name="player_screening.csv",
        mime="text/csv",
        key="ps_download_csv",
    )


# -----------------------------
# Page entrypoint (CRITICAL)
# -----------------------------
st.set_page_config(page_title="Player Screening", page_icon="🔎", layout="wide")
st.title("Player Screening")

SESSION_DF_KEYS = ["df", "data", "players_df", "dataset", "master_df", "merged_df"]
df_shared = _first_existing_df_from_session(SESSION_DF_KEYS)

if df_shared is None:
    st.info("No dataset loaded. Go to **Home** and upload/select your data first.")
    st.stop()

# Auto-detect Wyscout-ish columns
league_col = _pick_col(df_shared, ["League", "Competition", "league"])
position_col = _pick_col(df_shared, ["Specific Position", "Main Position", "Position"])
age_col = _pick_col(df_shared, ["Age"])
minutes_col = _pick_col(df_shared, ["Minutes", "Minutes played", "Min", "minutes_played"])
player_col = _pick_col(df_shared, ["Player", "Name"]) or "Player"
team_col = _pick_col(df_shared, ["Squad", "Team", "Club"]) or "Team"
season_col = _pick_col(df_shared, ["Season"])

missing = []
if league_col is None:
    missing.append("League")
if position_col is None:
    missing.append("Position")
if age_col is None:
    missing.append("Age")
if minutes_col is None:
    missing.append("Minutes")

if missing:
    st.error(f"Cannot find required columns: {missing}")
    st.write("Available columns:", list(df_shared.columns))
    st.stop()

render_player_screening(
    df_shared,
    league_col=league_col,
    position_col=position_col,
    age_col=age_col,
    minutes_col=minutes_col,
    player_col=player_col,
    team_col=team_col,
    season_col=season_col,
)
