# pages/7_Player_vs_League_Benchmark.py
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


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _numeric_metric_candidates(df: pd.DataFrame, exclude_cols: set[str]) -> list[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    metrics = [c for c in numeric_cols if c not in exclude_cols]
    return sorted(metrics)


def _default_direction_for_metric(metric_name: str) -> str:
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


def _player_key(row: pd.Series, player_col: str, team_col: str, league_col: str, position_col: str) -> str:
    p = str(row.get(player_col, "")).strip()
    t = str(row.get(team_col, "")).strip()
    l = str(row.get(league_col, "")).strip()
    pos = str(row.get(position_col, "")).strip()
    return f"{p} — {t} ({l}, {pos})"


def _zscore(x: float, mean: float, std: float) -> float:
    if std is None or not np.isfinite(std) or std == 0:
        return np.nan
    return (x - mean) / std


# -----------------------------
# Page
# -----------------------------
def render_benchmark_page(
    df: pd.DataFrame,
    *,
    league_col: str,
    position_col: str,
    age_col: str,
    minutes_col: str,
    player_col: str,
    team_col: str,
    matches_col: str | None = None,
):
    st.title("Player vs League Benchmark")
    st.markdown(
        "Benchmark a player against their league (and optionally position-filtered cohort).\n\n"
        "Outputs:\n"
        "- Table: player value vs cohort median/mean, delta, z-score, percentile\n"
        "- Radar: player percentiles across selected metrics\n"
        "- Distribution: selected metric distribution with player marker\n"
    )

    # Ensure numeric parsing for core columns
    df = df.copy()
    df[age_col] = _safe_num(df[age_col])
    df[minutes_col] = _safe_num(df[minutes_col])
    if matches_col and matches_col in df.columns:
        df[matches_col] = _safe_num(df[matches_col])

    # -----------------------------
    # Filters (choose player first, then benchmark cohort)
    # -----------------------------
    st.subheader("Select player")

    leagues = sorted([x for x in df[league_col].dropna().unique().tolist()])
    positions = sorted([x for x in df[position_col].dropna().unique().tolist()])

    # Quick filter to limit player dropdown size
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        filter_leagues = st.multiselect(
            "League(s) (to search player list)",
            options=leagues,
            default=leagues[:1] if leagues else [],
            key="pb_filter_leagues",
        )
    with c2:
        filter_positions = st.multiselect(
            "Position(s) (to search player list)",
            options=positions,
            default=[],
            key="pb_filter_positions",
        )
    with c3:
        player_search = st.text_input("Search player (optional)", value="", key="pb_player_search")

    player_pool = df.copy()
    if filter_leagues:
        player_pool = player_pool[player_pool[league_col].isin(filter_leagues)]
    if filter_positions:
        player_pool = player_pool[player_pool[position_col].isin(filter_positions)]
    if player_search.strip():
        q = player_search.strip().lower()
        # search across player + team
        player_pool = player_pool[
            player_pool[player_col].astype(str).str.lower().str.contains(q, na=False)
            | player_pool[team_col].astype(str).str.lower().str.contains(q, na=False)
        ]

    if player_pool.empty:
        st.warning("No players match the current search filters. Broaden league/position/search.")
        st.stop()

    player_pool = player_pool.copy()
    player_pool["_player_key"] = player_pool.apply(
        lambda r: _player_key(r, player_col, team_col, league_col, position_col), axis=1
    )

    base_key = st.selectbox(
        "Target player",
        options=player_pool["_player_key"].tolist(),
        index=0,
        key="pb_base_player",
    )

    base_rows = player_pool[player_pool["_player_key"] == base_key]
    if base_rows.empty:
        st.error("Could not find selected player.")
        st.stop()

    # If duplicates exist (same label), take first
    base = base_rows.iloc[0].copy()

    st.caption(
        f"Selected: **{base[player_col]}** | {base.get(team_col,'')} | "
        f"{base.get(league_col,'')} | {base.get(position_col,'')} | "
        f"Age: {int(base[age_col]) if pd.notna(base[age_col]) else '—'} | "
        f"Minutes: {int(base[minutes_col]) if pd.notna(base[minutes_col]) else '—'}"
    )

    # -----------------------------
    # Define benchmark cohort
    # -----------------------------
    st.subheader("Benchmark cohort")

    mode = st.radio(
        "Benchmark mode",
        options=["Whole League", "League + Selected Positions"],
        index=0,
        horizontal=True,
        key="pb_mode",
    )

    # Default cohort league = player's league
    default_league = [base[league_col]] if pd.notna(base[league_col]) else (leagues[:1] if leagues else [])
    bench_leagues = st.multiselect(
        "League(s) for benchmark cohort",
        options=leagues,
        default=default_league,
        key="pb_bench_leagues",
    )

    bench_positions = []
    if mode == "League + Selected Positions":
        default_pos = [base[position_col]] if pd.notna(base[position_col]) else []
        bench_positions = st.multiselect(
            "Position(s) for benchmark cohort",
            options=positions,
            default=default_pos,
            key="pb_bench_positions",
        )

    # Age / minutes cohort filters
    age_all = df[age_col]
    if age_all.notna().any():
        age_min = int(np.nanmin(age_all.values))
        age_max = int(np.nanmax(age_all.values))
    else:
        age_min, age_max = 0, 60

    minutes_all = df[minutes_col]
    if minutes_all.notna().any():
        min_min = int(np.nanmin(minutes_all.values))
        min_max = int(np.nanmax(minutes_all.values))
    else:
        min_min, min_max = 0, 0

    cA, cB = st.columns([2, 2])
    with cA:
        age_range = st.slider(
            "Age range (cohort)",
            min_value=age_min,
            max_value=age_max,
            value=(age_min, age_max),
            step=1,
            key="pb_age_range",
        )
    with cB:
        min_minutes_default = 400 if min_max >= 400 else max(0, min_max)
        min_minutes = st.slider(
            "Minimum Minutes (cohort)",
            min_value=min_min,
            max_value=min_max,
            value=min_minutes_default,
            step=10 if min_max >= 10 else 1,
            key="pb_min_minutes",
        )

    cohort = df.copy()
    if bench_leagues:
        cohort = cohort[cohort[league_col].isin(bench_leagues)]
    if mode == "League + Selected Positions" and bench_positions:
        cohort = cohort[cohort[position_col].isin(bench_positions)]

    cohort = cohort[
        cohort[age_col].between(age_range[0], age_range[1], inclusive="both")
        & (cohort[minutes_col] >= float(min_minutes))
    ].copy()

    if cohort.empty or len(cohort) < 10:
        st.warning("Benchmark cohort is too small. Broaden leagues/positions or relax filters.")
        st.stop()

    st.caption(
        f"Benchmark cohort size: {len(cohort):,} | "
        f"Mode: {mode} | "
        f"Leagues: {len(bench_leagues)} | "
        f"Positions: {', '.join(bench_positions) if bench_positions else ('All' if mode=='Whole League' else 'None selected')} | "
        f"Age: {age_range[0]}–{age_range[1]} | "
        f"Minutes ≥ {min_minutes}"
    )

    # -----------------------------
    # Metrics selection + direction
    # -----------------------------
    st.subheader("Metrics")

    exclude = {league_col, position_col, age_col, minutes_col, player_col, team_col}
    if matches_col:
        exclude.add(matches_col)

    metric_candidates = _numeric_metric_candidates(df, exclude)
    if not metric_candidates:
        st.error("No numeric metrics available.")
        st.stop()

    preferred_defaults = [
        "xG per 90",
        "xA per 90",
        "Shots per 90",
        "Key passes per 90",
        "Progressive passes per 90",
        "Progressive runs per 90",
        "Touches in box per 90",
        "Dribbles per 90",
        "Successful defensive actions per 90",
        "Defensive duels won, %",
    ]
    default_metrics = [m for m in preferred_defaults if m in metric_candidates]
    if not default_metrics:
        default_metrics = metric_candidates[:8]

    sel_metrics = st.multiselect(
        "Select benchmark metrics",
        options=metric_candidates,
        default=default_metrics[: min(10, len(default_metrics))],
        key="pb_metrics",
    )
    if not sel_metrics:
        st.info("Select at least one metric.")
        st.stop()

    # Directions (session-persistent)
    if "pb_directions" not in st.session_state:
        st.session_state.pb_directions = {}
    for m in sel_metrics:
        if m not in st.session_state.pb_directions:
            st.session_state.pb_directions[m] = _default_direction_for_metric(m)

    with st.expander("Metric direction (higher/lower is better)", expanded=False):
        cols = st.columns(2)
        for i, m in enumerate(sel_metrics):
            with cols[i % 2]:
                choice = st.radio(
                    m,
                    options=["Higher is better", "Lower is better"],
                    index=0 if st.session_state.pb_directions[m] == "higher" else 1,
                    key=f"pb_dir__{m}",
                    horizontal=True,
                )
                st.session_state.pb_directions[m] = "higher" if choice == "Higher is better" else "lower"

    directions = {m: st.session_state.pb_directions[m] for m in sel_metrics}

    # -----------------------------
    # Compute benchmark table
    # -----------------------------
    # Build a direction-adjusted cohort series per metric for percentile calc
    rows = []
    for m in sel_metrics:
        cohort_s = _safe_num(cohort[m])
        player_val = _safe_num(pd.Series([base.get(m, np.nan)])).iloc[0]

        # direction adjustment for percentile and z interpretation
        adj = -1.0 if directions.get(m, "higher") == "lower" else 1.0
        cohort_adj = cohort_s * adj
        player_adj = player_val * adj

        # basic stats on raw (not adjusted) for reporting
        cohort_mean = float(np.nanmean(cohort_s.values)) if cohort_s.notna().any() else np.nan
        cohort_median = float(np.nanmedian(cohort_s.values)) if cohort_s.notna().any() else np.nan
        cohort_std = float(np.nanstd(cohort_s.values, ddof=0)) if cohort_s.notna().any() else np.nan

        # z-score computed on raw scale (but flip sign so higher-z always "better")
        z_raw = _zscore(float(player_val), cohort_mean, cohort_std)
        z_better = z_raw * (1.0 if directions.get(m, "higher") == "higher" else -1.0)

        # percentile computed on adjusted
        if cohort_adj.notna().sum() >= 3 and np.isfinite(player_adj):
            # rank percentile: proportion <= player
            pctl = float(pd.Series(cohort_adj).rank(pct=True).loc[cohort_adj.index].mean())  # placeholder safety
            # compute percentile properly: rank player among cohort
            # Using numpy: percentile = % of cohort <= player
            pctl = float(np.mean(cohort_adj.values <= player_adj) * 100.0)
        else:
            pctl = np.nan

        delta_vs_median = float(player_val - cohort_median) if np.isfinite(player_val) and np.isfinite(cohort_median) else np.nan
        delta_vs_mean = float(player_val - cohort_mean) if np.isfinite(player_val) and np.isfinite(cohort_mean) else np.nan

        rows.append(
            {
                "Metric": m,
                "Direction": "Higher" if directions.get(m, "higher") == "higher" else "Lower",
                "Player": player_val,
                "Cohort median": cohort_median,
                "Cohort mean": cohort_mean,
                "Δ vs median": delta_vs_median,
                "Δ vs mean": delta_vs_mean,
                "Z (better+)": z_better,
                "Percentile (better+)": pctl,
            }
        )

    bench = pd.DataFrame(rows)

    # rounding / formatting
    for c in ["Player", "Cohort median", "Cohort mean", "Δ vs median", "Δ vs mean", "Z (better+)", "Percentile (better+)"]:
        bench[c] = pd.to_numeric(bench[c], errors="coerce")

    bench["Percentile (better+)"] = bench["Percentile (better+)"].round(0)
    bench["Z (better+)"] = bench["Z (better+)"].round(2)

    # keep player/cohort values at 2dp
    for c in ["Player", "Cohort median", "Cohort mean", "Δ vs median", "Δ vs mean"]:
        bench[c] = bench[c].round(2)

    st.subheader("Benchmark table")
    st.dataframe(bench, use_container_width=True, hide_index=True)

    csv = bench.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download benchmark table (CSV)",
        data=csv,
        file_name="player_benchmark.csv",
        mime="text/csv",
        key="pb_download_csv",
    )

    # -----------------------------
    # Radar: player percentiles
    # -----------------------------
    st.subheader("Radar (percentiles)")

    radar_metrics = st.multiselect(
        "Radar metrics",
        options=sel_metrics,
        default=sel_metrics[: min(10, len(sel_metrics))],
        key="pb_radar_metrics",
    )
    if not radar_metrics:
        st.info("Select at least one radar metric.")
        return

    # Compute percentiles for radar metrics (better+)
    radar_vals = []
    for m in radar_metrics:
        cohort_s = _safe_num(cohort[m])
        player_val = _safe_num(pd.Series([base.get(m, np.nan)])).iloc[0]
        adj = -1.0 if directions.get(m, "higher") == "lower" else 1.0
        cohort_adj = cohort_s * adj
        player_adj = player_val * adj
        if cohort_adj.notna().sum() >= 3 and np.isfinite(player_adj):
            pctl = float(np.mean(cohort_adj.values <= player_adj) * 100.0)
        else:
            pctl = np.nan
        radar_vals.append(pctl)

    # Close loop
    categories = radar_metrics[:]
    categories_closed = categories + [categories[0]]
    values_closed = radar_vals + [radar_vals[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            mode="lines",
            fill="toself",
            fillcolor=px.colors.qualitative.Bold[0],
            line=dict(color=px.colors.qualitative.Bold[0], width=3),
            opacity=0.35,
            name="Player percentile",
            hovertemplate="%{theta}<br>%{r:.0f} pctl<extra></extra>",
        )
    )

    fig.update_layout(
        height=760,
        margin=dict(l=90, r=90, t=120, b=120),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="left", x=0),
        polar=dict(
            bgcolor="white",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickmode="array",
                tickvals=[0, 20, 40, 60, 80, 100],
                ticks="",
                showline=True,
                linewidth=2,
                linecolor="rgba(0,0,0,0.40)",
                gridcolor="rgba(0,0,0,0.18)",
                tickfont=dict(size=11),
            ),
            angularaxis=dict(
                rotation=90,
                direction="clockwise",
                showline=True,
                linewidth=1,
                linecolor="rgba(0,0,0,0.18)",
                gridcolor="rgba(0,0,0,0.15)",
                tickfont=dict(size=13),
            ),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Distribution plot for one metric
    # -----------------------------
    st.subheader("Metric distribution (cohort)")

    dist_metric = st.selectbox(
        "Select metric to view distribution",
        options=sel_metrics,
        index=0,
        key="pb_dist_metric",
    )

    cohort_s = _safe_num(cohort[dist_metric]).dropna()
    player_val = _safe_num(pd.Series([base.get(dist_metric, np.nan)])).iloc[0]

    if cohort_s.empty or not np.isfinite(player_val):
        st.info("Distribution cannot be displayed for this metric (missing data).")
        return

    # Histogram + player marker
    hist = go.Figure()
    hist.add_trace(go.Histogram(x=cohort_s.values, nbinsx=30, opacity=0.75, name="Cohort"))
    hist.add_vline(x=float(player_val), line_width=3, line_dash="solid", annotation_text="Player", annotation_position="top")

    hist.update_layout(
        height=420,
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
        xaxis_title=dist_metric,
        yaxis_title="Count",
    )

    st.plotly_chart(hist, use_container_width=True)


# -----------------------------
# Page entrypoint
# -----------------------------
st.set_page_config(page_title="Benchmark", page_icon="🔄", layout="wide")

SESSION_DF_KEYS = ["df", "data", "players_df", "dataset", "master_df", "merged_df"]
df_shared = _first_existing_df_from_session(SESSION_DF_KEYS)

if df_shared is None:
    st.info("No dataset loaded. Go to **Home** and upload/select your data first.")
    st.stop()

league_col = _pick_col(df_shared, ["League", "Competition", "league"])
position_col = _pick_col(df_shared, ["Specific Position", "Main Position", "Position"])
age_col = _pick_col(df_shared, ["Age"])
minutes_col = _pick_col(df_shared, ["Minutes", "Minutes played", "Min", "minutes_played"])
matches_col = _pick_col(df_shared, ["Matches played", "Matches", "Appearances"])
player_col = _pick_col(df_shared, ["Player", "Name"]) or "Player"
team_col = _pick_col(df_shared, ["Squad", "Team", "Club"]) or "Team"

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

render_benchmark_page(
    df_shared,
    league_col=league_col,
    position_col=position_col,
    age_col=age_col,
    minutes_col=minutes_col,
    player_col=player_col,
    team_col=team_col,
    matches_col=matches_col,
)
