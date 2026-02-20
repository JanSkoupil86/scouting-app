import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Player Screening Feature
# -----------------------------
def render_player_screening(
    df: pd.DataFrame,
    *,
    league_col: str = "League",
    position_col: str = "Specific Position",
    age_col: str = "Age",
    minutes_col: str = "Minutes",
    player_col: str = "Player",
    team_col: str = "Squad",
    season_col: str | None = "Season",  # optional
    exclude_cols: set[str] | None = None,
    default_metrics: list[str] | None = None,
):
    st.header("Player Screening")

    # Safety checks
    required = [league_col, position_col, age_col, minutes_col]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        st.error(f"Missing required columns: {missing_required}")
        return

    if exclude_cols is None:
        exclude_cols = {player_col, team_col, league_col, position_col, age_col, minutes_col}
        if season_col and season_col in df.columns:
            exclude_cols.add(season_col)

    # Identify candidate metric columns (numeric only)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    metric_candidates = [c for c in numeric_cols if c not in exclude_cols]
    metric_candidates = sorted(metric_candidates)

    if not metric_candidates:
        st.warning("No numeric metric columns found for screening.")
        return

    # Defaults
    if default_metrics is None:
        # choose a few common-looking ones if present, else first 5
        preferred = ["xG p90", "xA p90", "Shots p90", "Key Passes p90", "Progressive Passes p90"]
        default_metrics = [m for m in preferred if m in metric_candidates][:5]
        if not default_metrics:
            default_metrics = metric_candidates[:5]

    # -----------------------------
    # Filters
    # -----------------------------
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
            "Specific Position",
            options=positions,
            index=0 if positions else None,
            key="ps_position",
        )

    # Age filter (range)
    age_min = int(np.nanmin(df[age_col].values)) if df[age_col].notna().any() else 0
    age_max = int(np.nanmax(df[age_col].values)) if df[age_col].notna().any() else 60

    with c3:
        age_range = st.slider(
            "Age range",
            min_value=age_min,
            max_value=age_max,
            value=(age_min, age_max),
            key="ps_age_range",
        )

    # Minutes filter
    min_min = int(np.nanmin(df[minutes_col].values)) if df[minutes_col].notna().any() else 0
    min_max = int(np.nanmax(df[minutes_col].values)) if df[minutes_col].notna().any() else 0

    min_minutes = st.slider(
        "Minimum Minutes",
        min_value=min_min,
        max_value=min_max,
        value=min(400, min_max) if min_max > 0 else 0,
        step=10,
        key="ps_min_minutes",
    )

    # Apply base cohort filters
    cohort = df.copy()

    if sel_leagues:
        cohort = cohort[cohort[league_col].isin(sel_leagues)]

    if sel_position and sel_position in positions:
        cohort = cohort[cohort[position_col] == sel_position]

    cohort = cohort[
        (cohort[age_col].between(age_range[0], age_range[1], inclusive="both")) &
        (cohort[minutes_col] >= min_minutes)
    ].copy()

    st.caption(
        f"Cohort size: {len(cohort):,} | "
        f"Leagues: {len(sel_leagues)} | Position: {sel_position} | "
        f"Age: {age_range[0]}–{age_range[1]} | Minutes ≥ {min_minutes}"
    )

    if cohort.empty:
        st.warning("No players in cohort after filters. Adjust filters and try again.")
        return

    # -----------------------------
    # Metrics & thresholds
    # -----------------------------
    st.subheader("Metrics & Screening thresholds")

    sel_metrics = st.multiselect(
        "Select metrics for screening",
        options=metric_candidates,
        default=[m for m in default_metrics if m in metric_candidates],
        key="ps_metrics",
    )

    if not sel_metrics:
        st.info("Select at least one metric to screen players.")
        return

    st.markdown("**Percentile ranges (applied to each metric):**")

    # Percentile thresholds per metric (stable keys)
    # Store in session_state so removing/re-adding a metric doesn’t wipe its last threshold.
    if "ps_thresholds" not in st.session_state:
        st.session_state.ps_thresholds = {}

    # Default threshold
    DEFAULT_LOW, DEFAULT_HIGH = 30, 100

    thresholds = {}
    for metric in sel_metrics:
        if metric not in st.session_state.ps_thresholds:
            st.session_state.ps_thresholds[metric] = (DEFAULT_LOW, DEFAULT_HIGH)

        low0, high0 = st.session_state.ps_thresholds[metric]

        low, high = st.slider(
            label=f"{metric}",
            min_value=0,
            max_value=100,
            value=(int(low0), int(high0)),
            step=1,
            key=f"ps_thr_{metric}",  # stable per metric
        )
        st.session_state.ps_thresholds[metric] = (low, high)
        thresholds[metric] = (low, high)

    # -----------------------------
    # Percentile computation (within cohort)
    # -----------------------------
    # Using rank(pct=True) is fast, vectorized, and stable.
    pct_df = pd.DataFrame(index=cohort.index)

    for metric in sel_metrics:
        # Coerce numeric safety
        s = pd.to_numeric(cohort[metric], errors="coerce")

        # Percentile rank; higher value = higher percentile.
        # If you need "lower is better" metrics, invert here (see note below).
        pct = s.rank(pct=True, method="average") * 100.0
        pct_df[f"{metric} pctl"] = pct

    # -----------------------------
    # Apply AND screening across metrics
    # -----------------------------
    mask = pd.Series(True, index=cohort.index)

    for metric in sel_metrics:
        low, high = thresholds[metric]
        pcol = f"{metric} pctl"

        # Conservative: NaN fails screening
        mask &= pct_df[pcol].between(low, high, inclusive="both") & pct_df[pcol].notna()

    screened = cohort.loc[mask].copy()
    screened = screened.join(pct_df, how="left")

    # -----------------------------
    # Output table
    # -----------------------------
    if screened.empty:
        st.warning("No players met all thresholds. Widen percentile ranges or change metrics.")
        return

    # Build display columns
    base_cols = [c for c in [player_col, team_col, league_col, position_col, age_col, minutes_col] if c in screened.columns]
    raw_metric_cols = sel_metrics
    pctl_cols = [f"{m} pctl" for m in sel_metrics]

    display_cols = base_cols + raw_metric_cols + pctl_cols

    # Helpful sort: average percentile across selected metrics
    screened["Avg pctl"] = screened[pctl_cols].mean(axis=1)
    display_cols = base_cols + ["Avg pctl"] + raw_metric_cols + pctl_cols

    screened = screened.sort_values("Avg pctl", ascending=False)

    st.dataframe(
        screened[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    # Optional: download
    csv = screened[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download screened players (CSV)",
        data=csv,
        file_name="player_screening.csv",
        mime="text/csv",
        key="ps_download_csv",
    )

    # Note for directionality support
    st.caption(
        "Note: Percentiles currently assume **higher is better**. "
        "If you have metrics where **lower is better** (e.g., Fouls, Errors), invert them before ranking."
    )
