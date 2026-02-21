# pages/7_Player_vs_League_Benchmark.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


# -----------------------------
# KPI Templates (Option 1: hard-coded)
# - You can expand this dict as you upload KPI screens for other positions.
# -----------------------------
KPI_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "LW": {
        "Dribbling – ATT 1v1": ["Dribbles per 90", "Successful dribbles, %", "Offensive duels won, %"],
        "Speed": ["Progressive runs per 90", "Accelerations per 90", "Touches in box per 90"],
        "Positioning - inverting": ["Touches in box per 90", "xG per 90", "Key passes per 90"],
        "Crossing": ["Crosses per 90", "Accurate crosses, %", "Deep completed crosses per 90"],
        "Defending in mid/low block": ["Successful defensive actions per 90", "Defensive duels per 90", "Defensive duels won, %"],
    }
}


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


def _template_positions_from_data(df: pd.DataFrame, position_col: str) -> list[str]:
    """
    We keep template keys short like 'LW', but your dataset may have 'LW', 'LWF', 'Left winger', etc.
    For now: use template keys as-is and let user select.
    """
    return sorted(list(KPI_TEMPLATES.keys()))


def _coerce_template_to_available_metrics(
    template: dict[str, list[str]],
    available_metrics: set[str],
) -> dict[str, list[str]]:
    fixed = {}
    for kpi, metrics in template.items():
        fixed[kpi] = [m for m in metrics if m in available_metrics]
    return fixed


def _union_metrics(kpi_map: dict[str, list[str]]) -> list[str]:
    out = []
    seen = set()
    for _, ms in kpi_map.items():
        for m in ms:
            if m not in seen:
                seen.add(m)
                out.append(m)
    return out


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
        "Benchmark a player against a cohort. Use **manual metric selection** or a **KPI template** (per position) "
        "that you can override in the UI.\n\n"
        "Outputs:\n"
        "- Table: player value vs cohort median/mean, delta, z-score, percentile\n"
        "- Radar: player vs cohort baseline (Percentiles or Z-scores) with raw + median in hover\n"
    )

    # Ensure numeric parsing for core columns
    df = df.copy()
    df[age_col] = _safe_num(df[age_col])
    df[minutes_col] = _safe_num(df[minutes_col])
    if matches_col and matches_col in df.columns:
        df[matches_col] = _safe_num(df[matches_col])

    # -----------------------------
    # Select player
    # -----------------------------
    st.subheader("Select player")

    leagues = sorted([x for x in df[league_col].dropna().unique().tolist()])
    positions = sorted([x for x in df[position_col].dropna().unique().tolist()])

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

    base = base_rows.iloc[0].copy()

    st.caption(
        f"Selected: **{base[player_col]}** | {base.get(team_col,'')} | "
        f"{base.get(league_col,'')} | {base.get(position_col,'')} | "
        f"Age: {int(base[age_col]) if pd.notna(base[age_col]) else '—'} | "
        f"Minutes: {int(base[minutes_col]) if pd.notna(base[minutes_col]) else '—'}"
    )

    # -----------------------------
    # Benchmark cohort
    # -----------------------------
    st.subheader("Benchmark cohort")

    mode = st.radio(
        "Benchmark mode",
        options=["Whole League", "League + Selected Positions"],
        index=0,
        horizontal=True,
        key="pb_mode",
    )

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

    age_all = df[age_col]
    age_min = int(np.nanmin(age_all.values)) if age_all.notna().any() else 0
    age_max = int(np.nanmax(age_all.values)) if age_all.notna().any() else 60

    minutes_all = df[minutes_col]
    min_min = int(np.nanmin(minutes_all.values)) if minutes_all.notna().any() else 0
    min_max = int(np.nanmax(minutes_all.values)) if minutes_all.notna().any() else 0

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
    # Metrics universe
    # -----------------------------
    exclude = {league_col, position_col, age_col, minutes_col, player_col, team_col}
    if matches_col:
        exclude.add(matches_col)

    metric_candidates = _numeric_metric_candidates(df, exclude)
    if not metric_candidates:
        st.error("No numeric metrics available.")
        st.stop()
    metric_set = set(metric_candidates)

    # -----------------------------
    # KPI templates vs manual selection
    # -----------------------------
    st.subheader("Role KPIs and metrics")

    metric_mode = st.radio(
        "Metric selection mode",
        options=["Use KPI template", "Manual metrics"],
        index=0,
        horizontal=True,
        key="pb_metric_mode",
    )

    # Session override storage: position_key -> kpi list (as list of dicts for stable ordering)
    if "pb_kpi_overrides" not in st.session_state:
        st.session_state.pb_kpi_overrides = {}

    # Default manual metrics
    preferred_defaults = [
        "xG per 90",
        "xA per 90",
        "Shots per 90",
        "Key passes per 90",
        "Progressive passes per 90",
        "Progressive runs per 90",
        "Touches in box per 90",
        "Dribbles per 90",
    ]
    default_manual = [m for m in preferred_defaults if m in metric_set]
    if not default_manual:
        default_manual = metric_candidates[:8]

    # Determine a default template position: try to match base position if key exists
    template_positions = _template_positions_from_data(df, position_col)
    base_pos_str = str(base.get(position_col, "")).strip()
    default_template_pos = base_pos_str if base_pos_str in template_positions else (template_positions[0] if template_positions else "LW")

    selected_template_pos = None
    kpi_map_effective: dict[str, list[str]] = {}
    sel_metrics: list[str] = []

    if metric_mode == "Manual metrics":
        sel_metrics = st.multiselect(
            "Select benchmark metrics (table)",
            options=metric_candidates,
            default=default_manual[: min(12, len(default_manual))],
            key="pb_metrics_manual",
        )
        if not sel_metrics:
            st.info("Select at least one metric.")
            st.stop()

        # For radar: start from same, but can select anything later
        st.info("Radar metrics can be selected independently below (from all dataset metrics).")

    else:
        if not template_positions:
            st.warning("No KPI templates available yet. Switch to Manual metrics.")
            st.stop()

        selected_template_pos = st.selectbox(
            "Choose KPI template (position)",
            options=template_positions,
            index=template_positions.index(default_template_pos) if default_template_pos in template_positions else 0,
            key="pb_template_pos",
        )

        # Load base template and remove non-existent metrics
        base_template = KPI_TEMPLATES.get(selected_template_pos, {})
        base_template = _coerce_template_to_available_metrics(base_template, metric_set)

        # Initialize override from template (once per position)
        if selected_template_pos not in st.session_state.pb_kpi_overrides:
            # keep order stable
            st.session_state.pb_kpi_overrides[selected_template_pos] = [
                {"kpi": kpi_name, "metrics": metrics[:]} for kpi_name, metrics in base_template.items()
            ]

        # Controls
        cX, cY, cZ = st.columns([1.2, 1.2, 1.2])
        with cX:
            apply_to_table = st.checkbox("Apply KPIs to benchmark table", value=True, key="pb_apply_kpi_table")
        with cY:
            apply_to_radar_default = st.checkbox("Apply KPIs as default radar metrics", value=True, key="pb_apply_kpi_radar")
        with cZ:
            reset = st.button("Reset KPIs to template defaults", use_container_width=True, key="pb_kpi_reset")

        if reset:
            st.session_state.pb_kpi_overrides[selected_template_pos] = [
                {"kpi": kpi_name, "metrics": metrics[:]} for kpi_name, metrics in base_template.items()
            ]

        st.markdown("Edit KPI names and assign metrics (2–3 recommended per KPI).")

        overrides = st.session_state.pb_kpi_overrides[selected_template_pos]

        # Render KPI blocks
        for i, block in enumerate(overrides):
            with st.expander(f"KPI {i+1}: {block['kpi']}", expanded=True):
                new_name = st.text_input(
                    "KPI name",
                    value=block["kpi"],
                    key=f"pb_kpi_name__{selected_template_pos}__{i}",
                )
                block["kpi"] = new_name.strip() if new_name.strip() else block["kpi"]

                chosen = st.multiselect(
                    "Assign metrics",
                    options=metric_candidates,
                    default=[m for m in block.get("metrics", []) if m in metric_set],
                    key=f"pb_kpi_metrics__{selected_template_pos}__{i}",
                )
                block["metrics"] = chosen

        # Persist changes back
        st.session_state.pb_kpi_overrides[selected_template_pos] = overrides

        # Build effective KPI map + union metrics
        kpi_map_effective = {b["kpi"]: b.get("metrics", []) for b in overrides if b.get("kpi")}
        template_union = _union_metrics(kpi_map_effective)

        if apply_to_table:
            sel_metrics = template_union[:]
            if not sel_metrics:
                st.warning("No KPI metrics selected. Assign metrics to KPIs or switch to Manual metrics.")
                st.stop()
        else:
            # allow manual pick even in KPI mode
            sel_metrics = st.multiselect(
                "Select benchmark metrics (table) — manual override",
                options=metric_candidates,
                default=template_union[: min(12, len(template_union))] if template_union else default_manual,
                key="pb_metrics_kpi_manual_override",
            )
            if not sel_metrics:
                st.info("Select at least one metric.")
                st.stop()

        # Optional: show what KPIs resolve to
        with st.expander("Show KPI → metrics mapping", expanded=False):
            for kpi, ms in kpi_map_effective.items():
                st.write(f"**{kpi}**: {', '.join(ms) if ms else '—'}")

        # Store default radar set for later use
        if apply_to_radar_default:
            st.session_state["pb_radar_default_from_kpi"] = template_union[:]
        else:
            st.session_state["pb_radar_default_from_kpi"] = None

    # -----------------------------
    # Directions store (table + radar)
    # -----------------------------
    if "pb_directions" not in st.session_state:
        st.session_state.pb_directions = {}

    for m in sel_metrics:
        if m not in st.session_state.pb_directions:
            st.session_state.pb_directions[m] = _default_direction_for_metric(m)

    with st.expander("Metric direction (higher/lower is better) — for table metrics", expanded=False):
        cols = st.columns(2)
        for i, m in enumerate(sel_metrics):
            with cols[i % 2]:
                choice = st.radio(
                    m,
                    options=["Higher is better", "Lower is better"],
                    index=0 if st.session_state.pb_directions[m] == "higher" else 1,
                    key=f"pb_dir__table__{m}",
                    horizontal=True,
                )
                st.session_state.pb_directions[m] = "higher" if choice == "Higher is better" else "lower"

    directions_table = {m: st.session_state.pb_directions[m] for m in sel_metrics}

    # -----------------------------
    # Compute benchmark table
    # -----------------------------
    rows = []
    for m in sel_metrics:
        cohort_s = _safe_num(cohort[m])
        player_val = _safe_num(pd.Series([base.get(m, np.nan)])).iloc[0]

        adj = -1.0 if directions_table.get(m, "higher") == "lower" else 1.0
        cohort_adj = cohort_s * adj
        player_adj = player_val * adj

        cohort_mean = float(np.nanmean(cohort_s.values)) if cohort_s.notna().any() else np.nan
        cohort_median = float(np.nanmedian(cohort_s.values)) if cohort_s.notna().any() else np.nan
        cohort_std = float(np.nanstd(cohort_s.values, ddof=0)) if cohort_s.notna().any() else np.nan

        z_raw = _zscore(float(player_val), cohort_mean, cohort_std)
        z_better = z_raw * (1.0 if directions_table.get(m, "higher") == "higher" else -1.0)

        if cohort_adj.notna().sum() >= 3 and np.isfinite(player_adj):
            pctl = float(np.mean(cohort_adj.values <= player_adj) * 100.0)
        else:
            pctl = np.nan

        delta_vs_median = float(player_val - cohort_median) if np.isfinite(player_val) and np.isfinite(cohort_median) else np.nan
        delta_vs_mean = float(player_val - cohort_mean) if np.isfinite(player_val) and np.isfinite(cohort_mean) else np.nan

        rows.append(
            {
                "Metric": m,
                "Direction": "Higher" if directions_table.get(m, "higher") == "higher" else "Lower",
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
    for c in ["Player", "Cohort median", "Cohort mean", "Δ vs median", "Δ vs mean", "Z (better+)", "Percentile (better+)"]:
        bench[c] = pd.to_numeric(bench[c], errors="coerce")

    bench["Percentile (better+)"] = bench["Percentile (better+)"].round(0)
    bench["Z (better+)"] = bench["Z (better+)"].round(2)
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
    # Radar: direct comparison (Percentiles or Z) — metrics from ALL dataset metrics
    # -----------------------------
    st.subheader("Radar (direct comparison)")

    radar_scale = st.radio(
        "Radar scale",
        options=["Percentiles", "Z-scores"],
        index=0,
        horizontal=True,
        key="pb_radar_scale",
    )
    show_baseline = st.checkbox("Show cohort baseline", value=True, key="pb_radar_baseline")

    # Default radar metrics:
    # - if KPI mode and user enabled "apply KPIs as default radar metrics" => use union
    # - else use table metrics as default
    radar_default = st.session_state.get("pb_radar_default_from_kpi", None)
    if not radar_default:
        radar_default = sel_metrics[:]
    radar_default = [m for m in radar_default if m in metric_set][:12]

    radar_metrics = st.multiselect(
        "Radar metrics (any numeric metric)",
        options=metric_candidates,
        default=radar_default,
        key="pb_radar_metrics",
    )
    if not radar_metrics:
        st.info("Select at least one radar metric.")
        return

    # Ensure direction settings exist for radar-only metrics
    for m in radar_metrics:
        if m not in st.session_state.pb_directions:
            st.session_state.pb_directions[m] = _default_direction_for_metric(m)
    radar_directions = {m: st.session_state.pb_directions[m] for m in radar_metrics}

    radar_rows = []
    for m in radar_metrics:
        cohort_s_raw = _safe_num(cohort[m]).dropna()
        player_raw = _safe_num(pd.Series([base.get(m, np.nan)])).iloc[0]

        if cohort_s_raw.empty or not np.isfinite(player_raw):
            cohort_median = np.nan
            cohort_mean = np.nan
            cohort_std = np.nan
        else:
            cohort_median = float(np.nanmedian(cohort_s_raw.values))
            cohort_mean = float(np.nanmean(cohort_s_raw.values))
            cohort_std = float(np.nanstd(cohort_s_raw.values, ddof=0))

        adj = -1.0 if radar_directions.get(m, "higher") == "lower" else 1.0
        cohort_adj = cohort_s_raw * adj
        player_adj = player_raw * adj

        if cohort_adj.notna().sum() >= 3 and np.isfinite(player_adj):
            pctl = float(np.mean(cohort_adj.values <= player_adj) * 100.0)
        else:
            pctl = np.nan

        z_raw = _zscore(float(player_raw), cohort_mean, cohort_std)
        z_better = z_raw * (1.0 if radar_directions.get(m, "higher") == "higher" else -1.0)

        radar_rows.append(
            dict(
                Metric=m,
                Player_raw=float(player_raw) if np.isfinite(player_raw) else np.nan,
                Cohort_median=float(cohort_median) if np.isfinite(cohort_median) else np.nan,
                Percentile=float(pctl) if np.isfinite(pctl) else np.nan,
                Z=float(z_better) if np.isfinite(z_better) else np.nan,
            )
        )

    radar_df = pd.DataFrame(radar_rows)

    if radar_scale == "Percentiles":
        player_r = radar_df["Percentile"].tolist()
        baseline_r = [50.0] * len(radar_metrics)
        r_range = [0, 100]
        hover_r_label = "Percentile"
    else:
        player_r = radar_df["Z"].tolist()
        baseline_r = [0.0] * len(radar_metrics)
        max_abs = np.nanmax(np.abs(radar_df["Z"].values)) if radar_df["Z"].notna().any() else 2.0
        max_abs = float(np.clip(max_abs, 1.5, 4.0))
        r_range = [-max_abs, max_abs]
        hover_r_label = "Z-score"

    categories = radar_metrics[:]
    categories_closed = categories + [categories[0]]
    player_closed = player_r + [player_r[0]]
    baseline_closed = baseline_r + [baseline_r[0]]

    hover_text = []
    for _, r in radar_df.iterrows():
        p_raw = f"{r['Player_raw']:.2f}" if np.isfinite(r["Player_raw"]) else "—"
        c_med = f"{r['Cohort_median']:.2f}" if np.isfinite(r["Cohort_median"]) else "—"
        pctl_txt = f"{r['Percentile']:.0f}" if np.isfinite(r["Percentile"]) else "—"
        z_txt = f"{r['Z']:.2f}" if np.isfinite(r["Z"]) else "—"
        hover_text.append(
            f"Player: {p_raw}<br>"
            f"Cohort median: {c_med}<br>"
            f"Percentile: {pctl_txt}<br>"
            f"Z-score: {z_txt}"
        )
    hover_text_closed = hover_text + [hover_text[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=player_closed,
            theta=categories_closed,
            mode="lines",
            fill="toself",
            fillcolor=px.colors.qualitative.Bold[0],
            line=dict(color=px.colors.qualitative.Bold[0], width=3),
            opacity=0.35,
            name="Player",
            text=hover_text_closed,
            hovertemplate="%{theta}<br>" + hover_r_label + ": %{r}<br>%{text}<extra></extra>",
        )
    )

    if show_baseline:
        fig.add_trace(
            go.Scatterpolar(
                r=baseline_closed,
                theta=categories_closed,
                mode="lines",
                fill="none",
                line=dict(color=px.colors.qualitative.Bold[1], width=3, dash="dash"),
                name="Cohort baseline",
                hovertemplate="%{theta}<br>Baseline: %{r}<extra></extra>",
            )
        )

    fig.update_layout(
        height=780,
        margin=dict(l=90, r=90, t=120, b=120),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="left", x=0),
        polar=dict(
            bgcolor="white",
            radialaxis=dict(
                visible=True,
                range=r_range,
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

    with st.expander("Show numeric values (player vs cohort median)", expanded=False):
        t = radar_df[["Metric", "Player_raw", "Cohort_median", "Percentile", "Z"]].copy()
        t["Player_raw"] = pd.to_numeric(t["Player_raw"], errors="coerce").round(2)
        t["Cohort_median"] = pd.to_numeric(t["Cohort_median"], errors="coerce").round(2)
        t["Percentile"] = pd.to_numeric(t["Percentile"], errors="coerce").round(0)
        t["Z"] = pd.to_numeric(t["Z"], errors="coerce").round(2)
        st.dataframe(t, use_container_width=True, hide_index=True)


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
