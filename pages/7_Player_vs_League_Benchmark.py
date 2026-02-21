# pages/7_Player_vs_League_Benchmark.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


# -----------------------------
# KPI Templates (Option 1: hard-coded)
# -----------------------------
KPI_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "LW": {
        "Dribbling – ATT 1v1": ["Dribbles per 90", "Successful dribbles, %", "Offensive duels won, %"],
        "Speed": ["Progressive runs per 90", "Accelerations per 90", "Touches in box per 90"],
        "Positioning - inverting": ["Touches in box per 90", "xG per 90", "Key passes per 90"],
        "Crossing": ["Crosses per 90", "Accurate crosses, %", "Deep completed crosses per 90"],
        "Defending in mid/low block": [
            "Successful defensive actions per 90",
            "Defensive duels per 90",
            "Defensive duels won, %",
        ],
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


def _build_metric_to_group(kpi_map: dict[str, list[str]] | None) -> dict[str, str] | None:
    if not isinstance(kpi_map, dict) or not kpi_map:
        return None
    metric_to_group: dict[str, str] = {}
    for kpi, ms in kpi_map.items():
        for m in ms:
            if m and m not in metric_to_group:
                metric_to_group[m] = kpi
    return metric_to_group


def _order_metrics_by_kpi(radar_metrics: list[str], metric_to_group: dict[str, str] | None) -> list[str]:
    """
    Reorder metrics so KPI groups are contiguous, preserving:
    - KPI order by first appearance in current selection
    - metric order within KPI as in current selection
    """
    if not metric_to_group:
        return radar_metrics

    kpi_order = []
    seen = set()
    for m in radar_metrics:
        g = metric_to_group.get(m, "Other")
        if g not in seen:
            seen.add(g)
            kpi_order.append(g)

    ordered: list[str] = []
    for g in kpi_order:
        ordered.extend([m for m in radar_metrics if metric_to_group.get(m, "Other") == g])
    return ordered


def _z_to_0_100(z: float, clip: float = 3.0) -> float:
    if not np.isfinite(z):
        return np.nan
    zc = float(np.clip(z, -clip, clip))
    return (zc + clip) / (2.0 * clip) * 100.0


def _make_contrast_palette(n: int) -> list[str]:
    base = (
        px.colors.qualitative.Bold
        + px.colors.qualitative.Prism
        + px.colors.qualitative.Safe
        + px.colors.qualitative.Vivid
    )
    if n <= len(base):
        return base[:n]
    return [base[i % len(base)] for i in range(n)]


def _format_value_chip(v: float, scale: str) -> str:
    if not np.isfinite(v):
        return "—"
    if scale == "Percentiles":
        return f"{v:.0f}"
    return f"{v:.2f}"


# -----------------------------
# Wedge radar (KPI-colored) with clean KPI legend + contiguous KPI ordering
# -----------------------------
def make_wedge_radar(
    radar_df: pd.DataFrame,
    radar_metrics: list[str],
    *,
    scale: str,
    show_baseline: bool,
    metric_to_group: dict[str, str] | None,
    dark_bg: bool = True,
) -> go.Figure:
    """
    Wedge radar via Barpolar:
      - each metric = wedge
      - wedge color = KPI group
      - value = percentile (0..100) or z mapped to 0..100
      - baseline ring: 50th percentile (or z=0 mapped to 50)
      - clean KPI legend using dummy traces (1 per KPI)
    """
    n = len(radar_metrics)
    step = 360.0 / float(n) if n else 0.0

    # radial values in 0..100
    if scale == "Percentiles":
        r_vals = radar_df["Percentile"].to_numpy(dtype=float)
        chip_vals = radar_df["Percentile"].to_numpy(dtype=float)
        baseline_r = 50.0
    else:
        z_vals = radar_df["Z"].to_numpy(dtype=float)
        r_vals = np.array([_z_to_0_100(z) for z in z_vals], dtype=float)
        chip_vals = z_vals
        baseline_r = 50.0  # z=0 -> 50

    # group per metric
    groups = []
    if metric_to_group:
        for m in radar_metrics:
            groups.append(metric_to_group.get(m, "Other"))
    else:
        groups = ["Other"] * n

    # group ordering (first appearance in ordered metrics)
    group_order = []
    seen = set()
    for g in groups:
        if g not in seen:
            seen.add(g)
            group_order.append(g)

    palette = _make_contrast_palette(len(group_order))
    group_color = {g: palette[i] for i, g in enumerate(group_order)}
    colors = [group_color[g] for g in groups]

    # angles: start at top and go clockwise
    theta = np.array([90.0 - i * step for i in range(n)], dtype=float)
    width = np.array([step] * n, dtype=float)

    fig = go.Figure()

    # main wedges
    fig.add_trace(
        go.Barpolar(
            r=r_vals,
            theta=theta,
            width=width,
            marker=dict(color=colors, line=dict(width=1, color="rgba(0,0,0,0.55)")),
            opacity=0.92,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=[
                (
                    f"<b>{m}</b><br>"
                    f"KPI: {groups[i]}<br>"
                    f"Player: {_format_value_chip(float(radar_df.loc[i,'Player_raw']), 'Z-scores')}"
                    f"<br>Cohort median: {_format_value_chip(float(radar_df.loc[i,'Cohort_median']), 'Z-scores')}"
                    f"<br>Percentile: {_format_value_chip(float(radar_df.loc[i,'Percentile']), 'Percentiles')}"
                    f"<br>Z (better+): {_format_value_chip(float(radar_df.loc[i,'Z']), 'Z-scores')}"
                )
                for i, m in enumerate(radar_metrics)
            ],
            showlegend=False,
        )
    )

    # baseline ring
    if show_baseline:
        ring_theta = np.linspace(0, 360, 361)
        ring_r = np.full_like(ring_theta, baseline_r, dtype=float)
        fig.add_trace(
            go.Scatterpolar(
                r=ring_r,
                theta=ring_theta,
                mode="lines",
                line=dict(width=3, dash="dash", color="rgba(120,210,255,0.85)"),
                opacity=0.85,
                name="Cohort baseline",
                hoverinfo="skip",
                showlegend=True,
            )
        )

    # numeric chips near outer edge
    chip_theta = theta
    chip_r = np.clip(r_vals + 8.0, 0, 100)
    chip_text = [_format_value_chip(chip_vals[i], scale) for i in range(n)]
    fig.add_trace(
        go.Scatterpolar(
            r=chip_r,
            theta=chip_theta,
            mode="text",
            text=chip_text,
            textfont=dict(size=12, color="white" if dark_bg else "black"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # KPI legend entries (dummy traces)
    for g in group_order:
        fig.add_trace(
            go.Scatterpolar(
                r=[None],
                theta=[None],
                mode="markers",
                marker=dict(size=10, color=group_color[g]),
                name=g,
                showlegend=True,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        height=820,
        margin=dict(l=80, r=80, t=80, b=120),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.14,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color="rgba(255,255,255,0.85)" if dark_bg else "rgba(0,0,0,0.80)"),
        ),
        polar=dict(
            bgcolor="rgba(0,0,0,0)" if dark_bg else "white",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickmode="array",
                tickvals=[0, 25, 50, 75, 100],
                tickfont=dict(size=11, color="rgba(255,255,255,0.80)" if dark_bg else "rgba(0,0,0,0.70)"),
                gridcolor="rgba(255,255,255,0.12)" if dark_bg else "rgba(0,0,0,0.12)",
                linecolor="rgba(255,255,255,0.25)" if dark_bg else "rgba(0,0,0,0.25)",
                showline=True,
            ),
            angularaxis=dict(
                tickmode="array",
                tickvals=theta.tolist(),
                ticktext=radar_metrics,
                tickfont=dict(size=12, color="rgba(255,255,255,0.90)" if dark_bg else "rgba(0,0,0,0.85)"),
                gridcolor="rgba(255,255,255,0.10)" if dark_bg else "rgba(0,0,0,0.10)",
                linecolor="rgba(255,255,255,0.12)" if dark_bg else "rgba(0,0,0,0.12)",
                rotation=90,
                direction="clockwise",
            ),
        ),
        paper_bgcolor="rgb(18, 8, 38)" if dark_bg else "white",
        plot_bgcolor="rgb(18, 8, 38)" if dark_bg else "white",
    )

    return fig


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

    if "pb_kpi_overrides" not in st.session_state:
        st.session_state.pb_kpi_overrides = {}

    st.session_state["pb_active_kpi_map"] = None
    st.session_state["pb_kpi_union"] = None

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
    default_manual = [m for m in preferred_defaults if m in metric_set] or metric_candidates[:8]

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
    else:
        template_positions = sorted(list(KPI_TEMPLATES.keys()))
        if not template_positions:
            st.warning("No KPI templates available yet. Switch to Manual metrics.")
            st.stop()

        base_pos_str = str(base.get(position_col, "")).strip()
        default_template_pos = base_pos_str if base_pos_str in template_positions else template_positions[0]

        selected_template_pos = st.selectbox(
            "Choose KPI template (position)",
            options=template_positions,
            index=template_positions.index(default_template_pos) if default_template_pos in template_positions else 0,
            key="pb_template_pos",
        )

        base_template = KPI_TEMPLATES.get(selected_template_pos, {})
        base_template = _coerce_template_to_available_metrics(base_template, metric_set)

        if selected_template_pos not in st.session_state.pb_kpi_overrides:
            st.session_state.pb_kpi_overrides[selected_template_pos] = [
                {"kpi": kpi_name, "metrics": metrics[:]} for kpi_name, metrics in base_template.items()
            ]

        reset = st.button("Reset KPIs to template defaults", use_container_width=True, key="pb_kpi_reset")
        if reset:
            st.session_state.pb_kpi_overrides[selected_template_pos] = [
                {"kpi": kpi_name, "metrics": metrics[:]} for kpi_name, metrics in base_template.items()
            ]

        overrides = st.session_state.pb_kpi_overrides[selected_template_pos]

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

        st.session_state.pb_kpi_overrides[selected_template_pos] = overrides

        kpi_map_effective = {b["kpi"]: b.get("metrics", []) for b in overrides if b.get("kpi")}
        template_union = _union_metrics(kpi_map_effective)

        st.session_state["pb_active_kpi_map"] = kpi_map_effective
        st.session_state["pb_kpi_union"] = template_union[:]

        sel_metrics = template_union[:] if template_union else default_manual

    # -----------------------------
    # Directions store
    # -----------------------------
    if "pb_directions" not in st.session_state:
        st.session_state.pb_directions = {}

    for m in sel_metrics:
        if m not in st.session_state.pb_directions:
            st.session_state.pb_directions[m] = _default_direction_for_metric(m)

    # -----------------------------
    # Benchmark table (simple)
    # -----------------------------
    rows = []
    directions_table = {m: st.session_state.pb_directions[m] for m in sel_metrics}
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

        rows.append(
            {
                "Metric": m,
                "Player": player_val,
                "Cohort median": cohort_median,
                "Percentile (better+)": pctl,
                "Z (better+)": z_better,
            }
        )

    bench = pd.DataFrame(rows)
    st.subheader("Benchmark table")
    st.dataframe(
        bench.assign(
            **{
                "Player": pd.to_numeric(bench["Player"], errors="coerce").round(2),
                "Cohort median": pd.to_numeric(bench["Cohort median"], errors="coerce").round(2),
                "Percentile (better+)": pd.to_numeric(bench["Percentile (better+)"], errors="coerce").round(0),
                "Z (better+)": pd.to_numeric(bench["Z (better+)"], errors="coerce").round(2),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------
    # Radar
    # -----------------------------
    st.subheader("Radar (direct comparison)")

    cTop1, cTop2, cTop3 = st.columns([1.3, 1.1, 1.3])
    with cTop1:
        radar_scale = st.radio(
            "Radar scale",
            ["Percentiles", "Z-scores"],
            index=0,
            horizontal=True,
            key="pb_radar_scale",
        )
    with cTop2:
        radar_style = st.radio(
            "Radar style",
            ["Polygon", "Wedge (KPI colored)"],
            index=1,
            horizontal=True,
            key="pb_radar_style",
        )
    with cTop3:
        show_baseline = st.checkbox("Show cohort baseline", value=True, key="pb_radar_baseline")

    cBtn1, _ = st.columns([1.2, 3.0])
    with cBtn1:
        apply_btn = st.button("Apply Role KPIs → Radar", use_container_width=True, key="pb_apply_kpi_to_radar")

    if apply_btn:
        kpi_union = st.session_state.get("pb_kpi_union", None)
        if kpi_union:
            st.session_state["pb_radar_metrics"] = [m for m in kpi_union if m in metric_set][:14]
            st.rerun()
        else:
            st.warning("No KPI template metrics available to apply. Use KPI mode and assign metrics first.")

    radar_default = st.session_state.get("pb_radar_metrics", None)
    if not radar_default:
        radar_default = st.session_state.get("pb_kpi_union", None) or sel_metrics[:]
    radar_default = [m for m in radar_default if m in metric_set][:14]

    radar_metrics = st.multiselect(
        "Radar metrics",
        options=metric_candidates,
        default=radar_default,
        key="pb_radar_metrics",
    )

    if not radar_metrics:
        auto = st.session_state.get("pb_kpi_union", None)
        if auto:
            st.session_state["pb_radar_metrics"] = [m for m in auto if m in metric_set][:14]
            st.rerun()
        st.info("Select at least one radar metric.")
        return

    # KPI group map for wedge radar + ordering
    active_kpi_map = st.session_state.get("pb_active_kpi_map", None)
    metric_to_group = _build_metric_to_group(active_kpi_map)

    # ✅ Make KPI blocks contiguous (applies to wedge and polygon for consistent ordering)
    radar_metrics = _order_metrics_by_kpi(radar_metrics, metric_to_group)

    # Ensure directions for radar metrics
    for m in radar_metrics:
        if m not in st.session_state.pb_directions:
            st.session_state.pb_directions[m] = _default_direction_for_metric(m)
    radar_directions = {m: st.session_state.pb_directions[m] for m in radar_metrics}

    # Build radar_df
    radar_rows = []
    for m in radar_metrics:
        cohort_s_raw = _safe_num(cohort[m]).dropna()
        player_raw = _safe_num(pd.Series([base.get(m, np.nan)])).iloc[0]

        cohort_median = float(np.nanmedian(cohort_s_raw.values)) if (not cohort_s_raw.empty) else np.nan
        cohort_mean = float(np.nanmean(cohort_s_raw.values)) if (not cohort_s_raw.empty) else np.nan
        cohort_std = float(np.nanstd(cohort_s_raw.values, ddof=0)) if (not cohort_s_raw.empty) else np.nan

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

    if radar_style == "Wedge (KPI colored)":
        fig = make_wedge_radar(
            radar_df=radar_df,
            radar_metrics=radar_metrics,
            scale=radar_scale,
            show_baseline=show_baseline,
            metric_to_group=metric_to_group,
            dark_bg=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Polygon radar (fallback)
        if radar_scale == "Percentiles":
            player_r = radar_df["Percentile"].tolist()
            baseline_r = [50.0] * len(radar_metrics)
            r_range = [0, 100]
        else:
            player_r = radar_df["Z"].tolist()
            baseline_r = [0.0] * len(radar_metrics)
            max_abs = np.nanmax(np.abs(radar_df["Z"].values)) if radar_df["Z"].notna().any() else 2.0
            max_abs = float(np.clip(max_abs, 1.5, 4.0))
            r_range = [-max_abs, max_abs]

        categories = radar_metrics[:]
        categories_closed = categories + [categories[0]]
        player_closed = player_r + [player_r[0]]
        baseline_closed = baseline_r + [baseline_r[0]]

        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=player_closed,
                theta=categories_closed,
                mode="lines",
                fill="toself",
                opacity=0.35,
                name="Player",
                hovertemplate="%{theta}<br>%{r}<extra></extra>",
            )
        )
        if show_baseline:
            fig.add_trace(
                go.Scatterpolar(
                    r=baseline_closed,
                    theta=categories_closed,
                    mode="lines",
                    fill="none",
                    line=dict(width=3, dash="dash"),
                    name="Cohort baseline",
                    hoverinfo="skip",
                )
            )

        fig.update_layout(
            height=780,
            margin=dict(l=90, r=90, t=120, b=120),
            polar=dict(
                radialaxis=dict(visible=True, range=r_range),
                angularaxis=dict(rotation=90, direction="clockwise"),
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True)


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
