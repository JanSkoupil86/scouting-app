# pages/5_Player_Screening.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


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


def _default_direction_for_metric(metric_name: str) -> str:
    """
    Heuristic: return 'lower' for metrics that are typically "bad when high".
    Extend this list as your schema grows.
    """
    m = metric_name.strip().lower()
    lower_tokens = [
        "foul",
        "yellow",
        "red",
        "card",
        "conceded",
        "against",
        "shots against",
        "xg against",
        "error",
        "mistake",
        "lost",
        "turnover",
        "dispossess",
        "offsides",
        "offside",
        "penalty conceded",
        "goals against",
    ]
    return "lower" if any(tok in m for tok in lower_tokens) else "higher"


def _make_player_key(row: pd.Series, player_col: str, team_col: str, league_col: str, position_col: str) -> str:
    p = str(row.get(player_col, "")).strip()
    t = str(row.get(team_col, "")).strip()
    l = str(row.get(league_col, "")).strip()
    pos = str(row.get(position_col, "")).strip()
    return f"{p} — {t} ({l}, {pos})"


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
        "**Screening filters**: League(s) + Position(s) + Age + Minutes.\n\n"
        "**Percentile cohort** can be computed either:\n"
        "- within the **selected positions** (more role-specific), or\n"
        "- within the **whole league cohort** (position-agnostic within chosen leagues).\n"
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

    pct_mode = st.radio(
        "Percentile reference cohort",
        options=["Percentiles vs Selected Positions", "Percentiles vs Whole League"],
        index=0,
        horizontal=True,
        key="ps_pct_mode",
    )

    c1, c2, c3 = st.columns([2, 2, 2])

    with c1:
        sel_leagues = st.multiselect(
            "League(s)",
            options=leagues,
            default=leagues[:1] if leagues else [],
            key="ps_leagues",
        )

    with c2:
        sel_positions = st.multiselect(
            "Position(s)",
            options=positions,
            default=[positions[0]] if positions else [],
            key="ps_positions",
        )

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

    # ---- base cohort: League + Age + Minutes
    base = df.copy()
    if sel_leagues:
        base = base[base[league_col].isin(sel_leagues)]

    base[age_col] = pd.to_numeric(base[age_col], errors="coerce")
    base[minutes_col] = pd.to_numeric(base[minutes_col], errors="coerce")

    base = base[
        base[age_col].between(age_range[0], age_range[1], inclusive="both")
        & (base[minutes_col] >= float(min_minutes))
    ].copy()

    if base.empty:
        st.warning("No players in cohort after League/Age/Minutes filters. Adjust filters and try again.")
        st.stop()

    # ---- screening pool: add Positions
    screened_pool = base.copy()
    if sel_positions:
        screened_pool = screened_pool[screened_pool[position_col].isin(sel_positions)].copy()

    if screened_pool.empty:
        st.warning("No players in cohort after adding Position filter(s). Adjust positions/filters.")
        st.stop()

    # ---- percentile reference cohort
    if pct_mode == "Percentiles vs Selected Positions":
        pct_ref = screened_pool
        pct_ref_label = "Selected Positions"
    else:
        pct_ref = base
        pct_ref_label = "Whole League (selected leagues, age, minutes)"

    st.caption(
        f"Screening pool size: {len(screened_pool):,} | "
        f"Percentiles computed vs: {pct_ref_label} | "
        f"Leagues: {len(sel_leagues)} | "
        f"Positions: {', '.join(sel_positions) if sel_positions else 'All'} | "
        f"Age: {age_range[0]}–{age_range[1]} | "
        f"Minutes ≥ {min_minutes}"
    )

    # =========================
    # Metrics, directions, thresholds
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

    # ---- Direction control
    st.markdown("**Metric direction (higher/lower is better):**")

    if "ps_directions" not in st.session_state:
        st.session_state.ps_directions = {}

    for m in sel_metrics:
        if m not in st.session_state.ps_directions:
            st.session_state.ps_directions[m] = _default_direction_for_metric(m)

    with st.expander("Set metric directions", expanded=False):
        cols = st.columns(2)
        for i, m in enumerate(sel_metrics):
            col = cols[i % 2]
            with col:
                choice = st.radio(
                    label=m,
                    options=["Higher is better", "Lower is better"],
                    index=0 if st.session_state.ps_directions[m] == "higher" else 1,
                    key=f"ps_dir__{m}",
                    horizontal=True,
                )
            st.session_state.ps_directions[m] = "higher" if choice == "Higher is better" else "lower"

    directions = {m: st.session_state.ps_directions[m] for m in sel_metrics}

    # ---- Threshold sliders
    st.markdown("**Percentile ranges (applied to each metric):**")

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
            key=f"ps_thr__{metric}",
        )

        st.session_state.ps_thresholds[metric] = (low, high)
        thresholds[metric] = (low, high)

    # =========================
    # Percentiles computed on pct_ref, then applied to screened_pool
    # =========================
    pct_cols = {}
    for metric in sel_metrics:
        s = pd.to_numeric(pct_ref[metric], errors="coerce")
        if directions.get(metric, "higher") == "lower":
            s = -s  # invert so "lower is better" maps to higher percentile
        pct_cols[metric] = s.rank(pct=True, method="average") * 100.0

    pct_ref_pct_df = pd.DataFrame(
        {f"{m} pctl": pct_cols[m] for m in sel_metrics},
        index=pct_ref.index,
    )

    screened_pct = screened_pool.join(pct_ref_pct_df, how="left")

    # =========================
    # Apply AND screening across metrics
    # =========================
    mask = pd.Series(True, index=screened_pct.index)
    for metric in sel_metrics:
        low, high = thresholds[metric]
        pcol = f"{metric} pctl"
        mask &= screened_pct[pcol].between(low, high, inclusive="both") & screened_pct[pcol].notna()

    result = screened_pct.loc[mask].copy()

    if result.empty:
        st.warning("No players met all thresholds. Widen percentile ranges or change metrics.")
        st.stop()

    # =========================
    # Output table + Heatmap styling (UPDATED formatting)
    # =========================
    base_cols = [c for c in [player_col, team_col, league_col, position_col, age_col, minutes_col] if c in result.columns]
    pctl_cols = [f"{m} pctl" for m in sel_metrics]

    result["Avg pctl"] = result[pctl_cols].mean(axis=1)
    display_cols = base_cols + ["Avg pctl"] + sel_metrics + pctl_cols
    result = result.sort_values("Avg pctl", ascending=False)

    st.subheader("Screened players (heatmap percentiles)")

    disp = result[display_cols].copy()

    # --- Formatting rules ---
    if age_col in disp.columns:
        disp[age_col] = pd.to_numeric(disp[age_col], errors="coerce").round(0)
    if minutes_col in disp.columns:
        disp[minutes_col] = pd.to_numeric(disp[minutes_col], errors="coerce").round(0)

    for c in pctl_cols + ["Avg pctl"]:
        if c in disp.columns:
            disp[c] = pd.to_numeric(disp[c], errors="coerce").round(0)

    for c in sel_metrics:
        if c in disp.columns:
            disp[c] = pd.to_numeric(disp[c], errors="coerce").round(2)

    # --- Styling ---
    styler = disp.style
    heat_cols = [c for c in (["Avg pctl"] + pctl_cols) if c in disp.columns]
    if heat_cols:
        styler = styler.background_gradient(subset=heat_cols, axis=None, cmap="RdYlGn")

    format_dict = {}
    if age_col in disp.columns:
        format_dict[age_col] = "{:.0f}"
    if minutes_col in disp.columns:
        format_dict[minutes_col] = "{:.0f}"
    for c in sel_metrics:
        if c in disp.columns:
            format_dict[c] = "{:.2f}"
    for c in heat_cols:
        format_dict[c] = "{:.0f}"

    styler = styler.format(format_dict)

    st.dataframe(styler, use_container_width=True, hide_index=True)

    csv = disp.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download screened players (CSV)",
        data=csv,
        file_name="player_screening.csv",
        mime="text/csv",
        key="ps_download_csv",
    )

    # =========================
    # Radar comparison (stronger contrast + lower position)
    # =========================
    st.subheader("Radar comparison (screened players)")

    radar_df = result.copy()
    radar_df["_player_key"] = radar_df.apply(
        lambda r: _make_player_key(r, player_col, team_col, league_col, position_col), axis=1
    )

    player_options = radar_df["_player_key"].tolist()

    cA, cB = st.columns([2, 1])
    with cA:
        selected_players = st.multiselect(
            "Select up to 5 players to compare",
            options=player_options,
            default=player_options[:2] if len(player_options) >= 2 else player_options[:1],
            key="ps_radar_players",
        )
    with cB:
        max_players = st.number_input(
            "Max players",
            min_value=1,
            max_value=5,
            value=5,
            step=1,
            key="ps_radar_max_players",
        )

    if len(selected_players) > int(max_players):
        st.warning(f"Please select at most {int(max_players)} players.")
        selected_players = selected_players[: int(max_players)]

    if not selected_players:
        st.info("Select at least one player to display the radar.")
        return

    radar_metric_options = sel_metrics[:]
    radar_metrics = st.multiselect(
        "Radar metrics",
        options=radar_metric_options,
        default=radar_metric_options[: min(10, len(radar_metric_options))],
        key="ps_radar_metrics",
    )

    if not radar_metrics:
        st.info("Select at least one radar metric.")
        return

    radar_pctl_cols = [f"{m} pctl" for m in radar_metrics]

    # Strong contrasting palette
    color_palette = px.colors.qualitative.Bold

    fig = go.Figure()

    sub = radar_df[radar_df["_player_key"].isin(selected_players)].copy()
    for c in radar_pctl_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    categories = radar_metrics[:]
    categories_closed = categories + [categories[0]]

    for i, (_, row) in enumerate(sub.iterrows()):
        values = [row.get(f"{m} pctl", np.nan) for m in radar_metrics]
        values_closed = values + [values[0]]

        color = color_palette[i % len(color_palette)]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                mode="lines",
                fill="toself",
                fillcolor=color,
                line=dict(color=color, width=3),
                opacity=0.35,
                name=row["_player_key"],
                hovertemplate="%{theta}<br>%{r:.0f} pctl<extra></extra>",
            )
        )

    fig.update_layout(
        height=750,
        margin=dict(l=90, r=90, t=110, b=120),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.10,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
        polar=dict(
            bgcolor="white",
           radialaxis=dict(
    visible=True,
    range=[0, 100],
    tickmode="array",
    tickvals=[0, 20, 40, 60, 80, 100],
    ticks="",
    showline=True,                 # ✅ draw the outer ring boundary
    linewidth=2,                   # ✅ make boundary visible
    linecolor="rgba(0,0,0,0.35)",  # ✅ stronger than grid
    gridcolor="rgba(0,0,0,0.18)",
    tickfont=dict(size=11),
),
            angularaxis=dict(
                rotation=90,
                direction="clockwise",
                showline=False,
                gridcolor="rgba(0,0,0,0.15)",
                tickfont=dict(size=13),
            ),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


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
