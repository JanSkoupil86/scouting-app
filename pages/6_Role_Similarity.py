# pages/6_Role_Similarity.py
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


def _make_player_key(row: pd.Series, player_col: str, team_col: str, league_col: str, position_col: str) -> str:
    p = str(row.get(player_col, "")).strip()
    t = str(row.get(team_col, "")).strip()
    l = str(row.get(league_col, "")).strip()
    pos = str(row.get(position_col, "")).strip()
    return f"{p} — {t} ({l}, {pos})"


def _safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _cosine_similarity_matrix(X: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = X @ b
    X_norm = np.linalg.norm(X, axis=1)
    b_norm = np.linalg.norm(b)
    denom = X_norm * (b_norm if b_norm != 0 else np.nan)
    sim = dot / denom
    sim = np.where(np.isfinite(sim), sim, np.nan)
    return sim


def _top_k_metric_explain(base_pctl: pd.Series, cand_pctl: pd.Series, k: int = 3) -> tuple[str, str]:
    delta = (cand_pctl - base_pctl).abs().dropna()
    if delta.empty:
        return ("", "")
    best = delta.nsmallest(min(k, len(delta))).index.tolist()
    worst = delta.nlargest(min(k, len(delta))).index.tolist()
    return ", ".join(best), ", ".join(worst)


# -----------------------------
# Page: Role Similarity Finder
# -----------------------------
def render_role_similarity_page(
    df: pd.DataFrame,
    *,
    league_col: str,
    position_col: str,
    age_col: str,
    minutes_col: str,
    player_col: str,
    team_col: str,
):
    st.title("Role Similarity Finder")
    st.markdown(
        "Find players **similar to a target player** within selected leagues and **one or more positions**.\n\n"
        "Similarity uses **direction-aware percentile profiles** and **cosine similarity** (profile-shape matching)."
    )

    # -----------------------------
    # Build metric candidates
    # -----------------------------
    exclude_cols = {league_col, position_col, age_col, minutes_col, player_col, team_col}
    metric_candidates = _numeric_metric_candidates(df, exclude_cols)
    if not metric_candidates:
        st.error("No numeric metrics found to compute similarity.")
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
    ]
    default_metrics = [m for m in preferred_defaults if m in metric_candidates]
    if not default_metrics:
        default_metrics = metric_candidates[:8]

    # -----------------------------
    # Cohort filters
    # -----------------------------
    st.subheader("Cohort filters")

    leagues = sorted([x for x in df[league_col].dropna().unique().tolist()])
    positions = sorted([x for x in df[position_col].dropna().unique().tolist()])

    c1, c2, c3 = st.columns([2, 2, 2])

    with c1:
        sel_leagues = st.multiselect(
            "League(s)",
            options=leagues,
            default=leagues[:1] if leagues else [],
            key="rs_leagues",
        )

    with c2:
        sel_positions = st.multiselect(
            "Position(s)",
            options=positions,
            default=[positions[0]] if positions else [],
            key="rs_positions",
        )

    # Age bounds (for cohort)
    age_series_all = _safe_to_numeric(df[age_col])
    if age_series_all.notna().any():
        age_min = int(np.nanmin(age_series_all.values))
        age_max = int(np.nanmax(age_series_all.values))
    else:
        age_min, age_max = 0, 60

    with c3:
        age_range = st.slider(
            "Age range (cohort)",
            min_value=age_min,
            max_value=age_max,
            value=(age_min, age_max),
            step=1,
            key="rs_age_range",
        )

    # Minutes bounds
    min_series_all = _safe_to_numeric(df[minutes_col])
    if min_series_all.notna().any():
        min_min = int(np.nanmin(min_series_all.values))
        min_max = int(np.nanmax(min_series_all.values))
    else:
        min_min, min_max = 0, 0

    min_minutes_default = 400 if min_max >= 400 else max(0, min_max)
    min_minutes = st.slider(
        "Minimum Minutes",
        min_value=min_min,
        max_value=min_max,
        value=min_minutes_default,
        step=10 if min_max >= 10 else 1,
        key="rs_min_minutes",
    )

    # Apply cohort filters
    cohort = df.copy()
    if sel_leagues:
        cohort = cohort[cohort[league_col].isin(sel_leagues)]
    if sel_positions:
        cohort = cohort[cohort[position_col].isin(sel_positions)]

    cohort[age_col] = _safe_to_numeric(cohort[age_col])
    cohort[minutes_col] = _safe_to_numeric(cohort[minutes_col])

    cohort = cohort[
        cohort[age_col].between(age_range[0], age_range[1], inclusive="both")
        & (cohort[minutes_col] >= float(min_minutes))
    ].copy()

    st.caption(
        f"Cohort size: {len(cohort):,} | "
        f"Leagues: {len(sel_leagues)} | "
        f"Positions: {', '.join(sel_positions) if sel_positions else 'All'} | "
        f"Age: {age_range[0]}–{age_range[1]} | "
        f"Minutes ≥ {min_minutes}"
    )

    if cohort.empty or len(cohort) < 5:
        st.warning("Cohort too small. Broaden leagues/positions/filters.")
        st.stop()

    # -----------------------------
    # Target player selection (from cohort)
    # -----------------------------
    st.subheader("Target player")

    cohort = cohort.copy()
    cohort["_player_key"] = cohort.apply(
        lambda r: _make_player_key(r, player_col, team_col, league_col, position_col), axis=1
    )

    player_options = cohort["_player_key"].tolist()
    base_key = st.selectbox(
        "Select the player to match against the cohort",
        options=player_options,
        index=0,
        key="rs_base_player",
    )

    base_idx = cohort.index[cohort["_player_key"] == base_key]
    if len(base_idx) == 0:
        st.error("Could not locate the selected player in the cohort.")
        st.stop()
    base_idx = base_idx[0]

    # -----------------------------
    # Metrics, directions, weights
    # -----------------------------
    st.subheader("Feature space (metrics)")

    sel_metrics = st.multiselect(
        "Select metrics used for similarity",
        options=metric_candidates,
        default=[m for m in default_metrics if m in metric_candidates],
        key="rs_metrics",
    )
    if not sel_metrics or len(sel_metrics) < 3:
        st.info("Select at least 3 metrics to compute stable similarity.")
        st.stop()

    # Directions
    st.markdown("**Metric direction (higher/lower is better):**")
    if "rs_directions" not in st.session_state:
        st.session_state.rs_directions = {}

    for m in sel_metrics:
        if m not in st.session_state.rs_directions:
            st.session_state.rs_directions[m] = _default_direction_for_metric(m)

    with st.expander("Set metric directions", expanded=False):
        cols = st.columns(2)
        for i, m in enumerate(sel_metrics):
            col = cols[i % 2]
            with col:
                choice = st.radio(
                    label=m,
                    options=["Higher is better", "Lower is better"],
                    index=0 if st.session_state.rs_directions[m] == "higher" else 1,
                    key=f"rs_dir__{m}",
                    horizontal=True,
                )
            st.session_state.rs_directions[m] = "higher" if choice == "Higher is better" else "lower"

    directions = {m: st.session_state.rs_directions[m] for m in sel_metrics}

    # Weights
    st.markdown("**Metric weights (optional):**")
    use_weights = st.checkbox("Enable custom weights", value=False, key="rs_use_weights")

    weights = np.ones(len(sel_metrics), dtype=float)
    if use_weights:
        with st.expander("Set weights (higher = more important)", expanded=False):
            w_cols = st.columns(2)
            for i, m in enumerate(sel_metrics):
                with w_cols[i % 2]:
                    w = st.slider(m, 0.0, 5.0, 1.0, 0.1, key=f"rs_w__{m}")
                    weights[i] = float(w)
        if np.all(weights == 0):
            st.warning("All weights are 0. Resetting to equal weights.")
            weights = np.ones(len(sel_metrics), dtype=float)

    # Similarity controls + NEW results age filter
    cA, cB, cC, cD = st.columns([1.2, 1.0, 1.0, 1.4])
    with cA:
        min_coverage = st.slider(
            "Min metric coverage",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.05,
            key="rs_min_coverage",
            help="Players missing too many metric values are excluded before similarity.",
        )
    with cB:
        top_n = st.slider("Top N results", 5, 100, 25, 1, key="rs_top_n")
    with cC:
        exclude_same_team = st.checkbox("Exclude same team", value=False, key="rs_excl_team")

    with cD:
        # ✅ Results-only age filter (does NOT affect similarity computation)
        apply_result_age = st.checkbox("Filter results by age (e.g. U25)", value=False, key="rs_apply_result_age")
        result_age_max = st.slider(
            "Max age in results",
            min_value=age_min,
            max_value=age_max,
            value=min(25, age_max),
            step=1,
            key="rs_result_age_max",
            disabled=not apply_result_age,
        )

    # -----------------------------
    # Build direction-aware percentile matrix (0..100)
    # -----------------------------
    pctl = pd.DataFrame(index=cohort.index)
    for m in sel_metrics:
        s = _safe_to_numeric(cohort[m])
        if directions.get(m, "higher") == "lower":
            s = -s
        pctl[m] = s.rank(pct=True, method="average") * 100.0

    # Coverage filtering
    coverage = pctl.notna().mean(axis=1)
    keep = coverage >= float(min_coverage)
    pctl = pctl.loc[keep].copy()

    if base_idx not in pctl.index:
        st.warning(
            "The selected base player is missing too many selected metrics under the current coverage threshold. "
            "Lower the coverage requirement or adjust metrics."
        )
        st.stop()

    # Median imputation
    med = pctl.median(axis=0, skipna=True)
    pctl = pctl.fillna(med)

    # Weighted vectors in 0..1
    X = (pctl[sel_metrics].to_numpy(dtype=float) / 100.0)
    w = weights.astype(float).copy()
    if np.sum(w) > 0:
        w = w / np.sum(w)
    Xw = X * w
    b = (pctl.loc[base_idx, sel_metrics].to_numpy(dtype=float) / 100.0) * w

    # Cosine similarity
    sim = _cosine_similarity_matrix(Xw, b)
    sim_series = pd.Series(sim, index=pctl.index, name="cos_sim")

    # Exclude base
    sim_series = sim_series.drop(index=base_idx, errors="ignore")

    # Optional: exclude same team
    if exclude_same_team and team_col in cohort.columns:
        base_team = str(cohort.loc[base_idx, team_col])
        same_team_idx = cohort.index[cohort[team_col].astype(str) == base_team]
        sim_series = sim_series.drop(index=same_team_idx, errors="ignore")

    sim_series = sim_series.dropna().sort_values(ascending=False)

    if sim_series.empty:
        st.warning("No comparable players after filtering. Broaden filters or relax coverage.")
        st.stop()

    # -----------------------------
    # Build results table (Top N first, then apply results age filter)
    # -----------------------------
    top_idx = sim_series.head(int(top_n)).index

    out = cohort.loc[top_idx, [player_col, team_col, league_col, position_col, age_col, minutes_col]].copy()
    out[age_col] = _safe_to_numeric(out[age_col])
    out[minutes_col] = _safe_to_numeric(out[minutes_col])

    out["Similarity"] = (sim_series.loc[top_idx].values * 100.0).clip(0, 100)

    # Explainability
    base_pctl = pctl.loc[base_idx, sel_metrics]
    best_list, worst_list = [], []
    for idx in top_idx:
        cand_pctl = pctl.loc[idx, sel_metrics]
        best, worst = _top_k_metric_explain(base_pctl, cand_pctl, k=3)
        best_list.append(best)
        worst_list.append(worst)

    out["Best matches"] = best_list
    out["Biggest diffs"] = worst_list
    out["_player_key"] = cohort.loc[top_idx, "_player_key"].values

    # ✅ Apply results-only age filter (U25 etc.)
    if apply_result_age:
        out = out[out[age_col].notna() & (out[age_col] <= float(result_age_max))].copy()

    if out.empty:
        st.warning("No results left after applying the results age filter. Increase the max age or disable it.")
        st.stop()

    # Formatting
    out["Similarity"] = pd.to_numeric(out["Similarity"], errors="coerce").round(1)
    out[age_col] = pd.to_numeric(out[age_col], errors="coerce").round(0)
    out[minutes_col] = pd.to_numeric(out[minutes_col], errors="coerce").round(0)

    st.subheader("Most similar players")
    st.dataframe(
        out[
            [
                "_player_key",
                "Similarity",
                player_col,
                team_col,
                league_col,
                position_col,
                age_col,
                minutes_col,
                "Best matches",
                "Biggest diffs",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    csv = out.drop(columns=["_player_key"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download similarity results (CSV)",
        data=csv,
        file_name="role_similarity.csv",
        mime="text/csv",
        key="rs_download_csv",
    )

    # -----------------------------
    # Radar comparison (base vs selected similar players)
    # -----------------------------
    st.subheader("Radar comparison")

    radar_metrics = st.multiselect(
        "Radar metrics",
        options=sel_metrics,
        default=sel_metrics[: min(10, len(sel_metrics))],
        key="rs_radar_metrics",
    )
    if not radar_metrics:
        st.info("Select at least one radar metric.")
        return

    candidates_for_radar = out["_player_key"].tolist()
    default_radar = candidates_for_radar[:2] if len(candidates_for_radar) >= 2 else candidates_for_radar[:1]

    chosen_similar = st.multiselect(
        "Select similar players to overlay (up to 4)",
        options=candidates_for_radar,
        default=default_radar,
        key="rs_radar_players",
    )[:4]

    # Build radar pctl from pctl (already imputed)
    radar_pctl = pctl[radar_metrics].copy()

    plot_rows = []
    base_label = f"BASE: {cohort.loc[base_idx, '_player_key']}"
    plot_rows.append((base_idx, base_label))

    key_to_idx = {cohort.loc[i, "_player_key"]: i for i in cohort.index}
    for k in chosen_similar:
        idx = key_to_idx.get(k)
        if idx is not None and idx in radar_pctl.index:
            plot_rows.append((idx, k))

    if len(plot_rows) < 2:
        st.info("Select at least one similar player to compare with the base player.")
        return

    color_palette = px.colors.qualitative.Bold
    fig = go.Figure()

    categories = radar_metrics[:]
    categories_closed = categories + [categories[0]]

    for i, (idx, label) in enumerate(plot_rows):
        vals = [float(radar_pctl.loc[idx, m]) for m in radar_metrics]
        vals_closed = vals + [vals[0]]

        color = color_palette[i % len(color_palette)]
        fig.add_trace(
            go.Scatterpolar(
                r=vals_closed,
                theta=categories_closed,
                mode="lines",
                fill="toself",
                fillcolor=color,
                line=dict(color=color, width=3),
                opacity=0.35,
                name=label,
                hovertemplate="%{theta}<br>%{r:.0f} pctl<extra></extra>",
            )
        )

    fig.update_layout(
        height=780,
        margin=dict(l=90, r=90, t=120, b=120),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
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
# Page entrypoint
# -----------------------------
st.set_page_config(page_title="Role Similarity", page_icon="🔎", layout="wide")

SESSION_DF_KEYS = ["df", "data", "players_df", "dataset", "master_df", "merged_df"]
df_shared = _first_existing_df_from_session(SESSION_DF_KEYS)

if df_shared is None:
    st.info("No dataset loaded. Go to **Home** and upload/select your data first.")
    st.stop()

league_col = _pick_col(df_shared, ["League", "Competition", "league"])
position_col = _pick_col(df_shared, ["Specific Position", "Main Position", "Position"])
age_col = _pick_col(df_shared, ["Age"])
minutes_col = _pick_col(df_shared, ["Minutes", "Minutes played", "Min", "minutes_played"])
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

render_role_similarity_page(
    df_shared,
    league_col=league_col,
    position_col=position_col,
    age_col=age_col,
    minutes_col=minutes_col,
    player_col=player_col,
    team_col=team_col,
)
