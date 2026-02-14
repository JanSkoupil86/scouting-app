# pages/3_Compare.py
# Compare page (full): multi-player + cross-season mode, League→Team-dependent filters,
# Main Position-based position filter, PROFILES presets + manual metrics,
# Raw table (2 decimals), Percentile table (filled highlight), Radar (filled) + profile name,
# Z-score stripplot (distribution dots + selected diamonds),
# Optional exports (CSV/HTML; PNG/PDF if Kaleido works).

import re
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PROFILES (recommended in src/profiles.py). Fallback if not present.
try:
    from src.profiles import PROFILES
except Exception:
    PROFILES = {}

st.set_page_config(page_title="Compare", layout="wide")
st.title("Compare")

# -----------------------------
# Require uploaded data
# -----------------------------
if "data" not in st.session_state:
    st.warning("No data loaded yet. Go to **Home** and upload a CSV.")
    st.stop()

df = st.session_state["data"]
if df is None or not isinstance(df, pd.DataFrame) or df.empty:
    st.warning("Dataset is empty/invalid. Re-upload your CSV on Home.")
    st.stop()

# -----------------------------
# Column mapping (your conventions)
# -----------------------------
LEAGUE_COL = "Competition" if "Competition" in df.columns else ("League" if "League" in df.columns else None)
TEAM_COL = "Team within selected timeframe" if "Team within selected timeframe" in df.columns else ("Team" if "Team" in df.columns else None)
POS_COL = "Main Position" if "Main Position" in df.columns else ("Position" if "Position" in df.columns else None)

required = ["Player", "Minutes played"]
missing = [c for c in required if c not in df.columns]
if LEAGUE_COL is None:
    missing.append("Competition/League")
if TEAM_COL is None:
    missing.append("Team within selected timeframe/Team")
if POS_COL is None:
    missing.append("Main Position/Position")

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Clean text cols
for c in ["Player", LEAGUE_COL, TEAM_COL, POS_COL]:
    df[c] = df[c].astype(str).str.strip()

# -----------------------------
# Helpers
# -----------------------------
def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def unique_sorted(series: pd.Series) -> list[str]:
    s = series.dropna().astype(str).str.strip()
    s = s[(s != "") & (s.str.lower() != "nan")]
    return sorted(s.unique().tolist())

def extract_season_from_text(text: str) -> str | None:
    """
    Extract season token from Competition/League text.
    Supports:
      - 2024-25 / 2024/25 -> 24/25
      - 24/25 -> 24/25
      - 2024 / 2025 (spring-fall) -> 2024
    """
    if text is None:
        return None
    t = str(text)

    m = re.search(r"\b(20\d{2})\s*[-/]\s*(\d{2})\b", t)
    if m:
        y1 = int(m.group(1))
        y2 = int(m.group(2))
        return f"{str(y1)[2:]}/{y2:02d}"

    m = re.search(r"\b(\d{2})\s*/\s*(\d{2})\b", t)
    if m:
        return f"{int(m.group(1)):02d}/{int(m.group(2)):02d}"

    m = re.search(r"\b(20\d{2})\b", t)
    if m:
        return m.group(1)

    return None

def position_group(main_pos: str) -> str:
    if not main_pos:
        return "Other"
    p = str(main_pos).upper()

    if "GK" in p:
        return "Goalkeeper"
    if any(x in p for x in ["CB", "LCB", "RCB"]):
        return "Center Back"
    if any(x in p for x in ["LB", "LWB"]):
        return "Left Back/WB"
    if any(x in p for x in ["RB", "RWB"]):
        return "Right Back/WB"
    if "DMF" in p:
        return "Defensive Midfield"
    if any(x in p for x in ["CMF", "LCMF", "RCMF"]):
        return "Center Midfield"
    if "AMF" in p:
        return "Attacking Midfield"
    if any(x in p for x in ["LWF", "LW"]):
        return "Left Wing"
    if any(x in p for x in ["RWF", "RW"]):
        return "Right Wing"
    if any(x in p for x in ["CF", "ST", "SS"]):
        return "Center Forward"
    return "Other"

def is_lower_better(metric_name: str) -> bool:
    """Heuristic: metrics where lower values are better (invert percentiles/z)."""
    m = (metric_name or "").lower()
    bad_tokens = [
        "conceded", "against", "foul", "yellow", "red", "error",
        "shots against", "xg against", "cards", "penalties conceded",
    ]
    return any(t in m for t in bad_tokens)

def safe_metric_list(df_: pd.DataFrame) -> list[str]:
    """Detect numeric-ish columns suitable for comparisons."""
    exclude = {LEAGUE_COL, TEAM_COL, POS_COL, "Season", "Position Group", "Player"}
    candidates = [c for c in df_.columns if c not in exclude]
    out = []
    for c in candidates:
        s = to_num(df_[c])
        if s.notna().sum() >= max(20, int(0.05 * len(df_))):
            out.append(c)
    return sorted(out)

def compute_percentiles(cohort: pd.DataFrame, players: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """
    Returns a dataframe: rows = metrics, cols = selected player labels, values = percentile (0-100).
    Percentiles computed within cohort distribution. Inverts if lower is better.
    """
    result = {}
    for m in metrics:
        s = to_num(cohort[m])
        if s.notna().sum() < 10:
            continue

        # percentile for each value = rank position within cohort
        ranks = s.rank(pct=True) * 100.0

        # map value->percentile via interpolation on sorted pairs
        # (rank(pct) already aligns to original index; we just lookup for players' values via CDF-like approach)
        # Use numpy percentiles for stability:
        cohort_vals = s.dropna().values
        cohort_vals_sorted = np.sort(cohort_vals)

        def pct_of_value(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return np.nan
            # percentage of cohort <= v
            return float(np.searchsorted(cohort_vals_sorted, v, side="right")) / len(cohort_vals_sorted) * 100.0

        col = {}
        for label, row in players.iterrows():
            v = to_num(pd.Series([row[m]])).iloc[0]
            p = pct_of_value(v)
            if is_lower_better(m) and np.isfinite(p):
                p = 100.0 - p
            col[label] = p

        result[m] = col

    pct_df = pd.DataFrame(result).T  # rows=metrics
    return pct_df

def compute_zscores(cohort: pd.DataFrame, players: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, dict, dict]:
    """
    Returns:
      z_df: rows=metrics, cols=player labels, z-scores
      mu, sd: dict per metric
    Inverts sign if lower is better (so right = better).
    """
    mu, sd = {}, {}
    z_out = {}
    for m in metrics:
        s = to_num(cohort[m])
        if s.notna().sum() < 10:
            continue
        mu[m] = float(s.mean(skipna=True))
        sd_m = float(s.std(skipna=True))
        sd[m] = sd_m if sd_m > 1e-9 else np.nan

        rowz = {}
        for label, prow in players.iterrows():
            v = to_num(pd.Series([prow[m]])).iloc[0]
            if not np.isfinite(sd[m]) or pd.isna(v):
                rowz[label] = np.nan
            else:
                z = (float(v) - mu[m]) / sd[m]
                if is_lower_better(m) and np.isfinite(z):
                    z = -z
                rowz[label] = z
        z_out[m] = rowz

    z_df = pd.DataFrame(z_out).T
    return z_df, mu, sd

def zscore_stripplot(
    cohort_df: pd.DataFrame,
    players_df_labeled: pd.DataFrame,
    metrics: list[str],
    title: str,
    color_map: dict[str, str],
) -> go.Figure:
    """
    Grey dots = cohort z distribution; diamonds = selected players (labeled index).
    Right = better (we invert lower-better metrics).
    """
    usable = []
    for m in metrics:
        if m in cohort_df.columns:
            s = to_num(cohort_df[m])
            if s.notna().sum() >= 10:
                usable.append(m)

    if not usable:
        fig = go.Figure()
        fig.update_layout(title="No usable metrics for z-score plot.")
        return fig

    # Precompute cohort z distributions
    z_cohort = {}
    for m in usable:
        s = to_num(cohort_df[m])
        mu = float(s.mean(skipna=True))
        sd = float(s.std(skipna=True))
        if sd <= 1e-9:
            z = s * np.nan
        else:
            z = (s - mu) / sd
            if is_lower_better(m):
                z = -z
        z_cohort[m] = z.dropna().values

    fig = make_subplots(
        rows=len(usable),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=usable,
    )

    for i, m in enumerate(usable, start=1):
        zvals = z_cohort[m]
        if len(zvals) == 0:
            continue

        # cohort dots
        fig.add_trace(
            go.Scatter(
                x=zvals,
                y=np.zeros(len(zvals)),
                mode="markers",
                marker=dict(size=6, color="rgba(140,140,140,0.45)"),
                hovertemplate=f"{m}<br>z=%{{x:.2f}}<extra></extra>",
                showlegend=False,
            ),
            row=i,
            col=1,
        )

        # mean line at 0
        fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="rgba(80,80,80,0.65)", row=i, col=1)

        # player diamonds
        for label, row in players_df_labeled.iterrows():
            v = to_num(pd.Series([row.get(m, np.nan)])).iloc[0]
            # compute z using cohort mean/sd (again) for that metric
            s = to_num(cohort_df[m])
            mu = float(s.mean(skipna=True))
            sd = float(s.std(skipna=True))
            if sd <= 1e-9 or pd.isna(v):
                continue
            z = (float(v) - mu) / sd
            if is_lower_better(m):
                z = -z

            fig.add_trace(
                go.Scatter(
                    x=[z],
                    y=[0],
                    mode="markers+text",
                    marker=dict(size=10, symbol="diamond", color=color_map.get(label, "#1f77b4")),
                    text=[label],
                    textposition="top center",
                    hovertemplate=f"{label}<br>{m}: {float(v):.2f}<br>z={float(z):.2f}<extra></extra>",
                    name=label,
                    showlegend=(i == 1),
                ),
                row=i,
                col=1,
            )

        fig.update_yaxes(visible=False, row=i, col=1)

    fig.update_xaxes(title_text="Z-score (0 = cohort average)", row=len(usable), col=1)
    fig.update_layout(
        title=title,
        height=max(420, 120 * len(usable)),
        margin=dict(l=20, r=20, t=70, b=40),
    )
    fig.add_annotation(x=-2.5, y=1.0, xref="x", yref="paper", text="Worse", showarrow=False, opacity=0.55)
    fig.add_annotation(x=2.5, y=1.0, xref="x", yref="paper", text="Better", showarrow=False, opacity=0.55)
    return fig

def radar_percentiles(pct_df: pd.DataFrame, title: str, color_map: dict[str, str]) -> go.Figure:
    """
    pct_df: rows=metrics, cols=players (labels), values 0..100
    """
    metrics = pct_df.index.tolist()
    fig = go.Figure()

    for label in pct_df.columns:
        vals = pct_df[label].values.astype(float)
        # close the polygon
        theta = metrics + [metrics[0]]
        r = list(vals) + [vals[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=r,
                theta=theta,
                fill="toself",
                name=label,
                line=dict(color=color_map.get(label, None)),
                opacity=0.35,
            )
        )

    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
        ),
        legend=dict(orientation="v"),
        height=520,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig

# -----------------------------
# Build Season + Position Group
# -----------------------------
df2 = df.copy()
if "Season" not in df2.columns:
    df2["Season"] = df2[LEAGUE_COL].apply(extract_season_from_text)

df2["Season"] = df2["Season"].astype(str).str.strip()
df2.loc[df2["Season"].str.lower().isin(["nan", "none", ""]), "Season"] = np.nan

df2["Position Group"] = df2[POS_COL].apply(position_group)

# Numeric candidate metrics
numeric_cols = safe_metric_list(df2)
if not numeric_cols:
    st.error("No numeric metrics detected.")
    st.stop()

# -----------------------------
# Sidebar: Cohort Filters (League -> Team dependent)
# -----------------------------
with st.sidebar:
    st.header("Player Filters")

    # Seasons (multi)
    seasons_all = unique_sorted(df2["Season"])
    seasons_selected = st.multiselect(
        "Season(s)",
        options=seasons_all,
        default=seasons_all[:1] if seasons_all else [],
        help="Select one or multiple seasons. 24/25 + 25/26 and/or 2024 + 2025.",
    )

    df_f = df2.copy()
    if seasons_selected:
        df_f = df_f[df_f["Season"].astype(str).isin([str(s) for s in seasons_selected])]

    # Leagues (multi)
    leagues_all = unique_sorted(df_f[LEAGUE_COL])
    leagues_selected = st.multiselect(
        "League(s)",
        options=leagues_all,
        default=leagues_all[:1] if leagues_all else [],
        help="Leagues available in selected seasons.",
    )
    if leagues_selected:
        df_f = df_f[df_f[LEAGUE_COL].astype(str).isin([str(l) for l in leagues_selected])]

    # Minutes
    df_f["Minutes played"] = to_num(df_f["Minutes played"]).fillna(0)
    max_mins = int(df_f["Minutes played"].max()) if not df_f.empty else 0
    min_minutes = st.slider("Minimum minutes", 0, max(200, max_mins), min(600, max_mins) if max_mins else 0, step=50)
    df_f = df_f[df_f["Minutes played"] >= min_minutes]

    # Team (dependent on league + season)
    teams_all = unique_sorted(df_f[TEAM_COL])
    team_selected = st.selectbox("Team", ["All"] + teams_all, index=0)
    if team_selected != "All":
        df_f = df_f[df_f[TEAM_COL].astype(str) == str(team_selected)]

    # Position (use Main Position)
    pos_all = unique_sorted(df_f[POS_COL])
    pos_selected = st.selectbox("Main Position", ["All"] + pos_all, index=0)
    if pos_selected != "All":
        df_f = df_f[df_f[POS_COL].astype(str) == str(pos_selected)]

    # Position Group benchmark control
    pos_group_all = unique_sorted(df_f["Position Group"])
    pos_group_selected = st.selectbox("Position Group (benchmark)", ["All"] + pos_group_all, index=0)
    if pos_group_selected != "All":
        df_f = df_f[df_f["Position Group"].astype(str) == str(pos_group_selected)]

# This df_f is the cohort basis (also used for percentiles/z-scores benchmark)
if df_f.empty:
    st.info("No players match the cohort filters. Adjust Season/League/Minutes.")
    st.stop()

league_label = " / ".join(leagues_selected) if leagues_selected else "All leagues"
season_label = " / ".join(seasons_selected) if seasons_selected else "All seasons"
bench_label = f"{pos_group_selected if pos_group_selected != 'All' else 'All positions'} | {league_label} | {season_label}"

# -----------------------------
# Compare Mode: different players vs same player across seasons
# -----------------------------
st.markdown("### Selection")

mode = st.radio(
    "Compare mode",
    ["Compare players", "Same player across seasons"],
    horizontal=True,
)

# Choose number of comparison slots (2–5)
n_players = st.number_input("Number of players to compare", min_value=2, max_value=5, value=2, step=1)

# Colors
with st.expander("🎨 Customize player colors", expanded=False):
    default_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    color_picks = []
    for i in range(int(n_players)):
        c = st.color_picker(f"Player {i+1} color", value=default_palette[i % len(default_palette)])
        color_picks.append(c)

# Build selection rows
players_available = unique_sorted(df_f["Player"])

selected_rows_labeled = pd.DataFrame()
labels = []

if mode == "Compare players":
    # multi-select players (2..5) from cohort
    selected_players = st.multiselect(
        "Players to compare",
        options=players_available,
        default=players_available[: int(n_players)] if len(players_available) >= int(n_players) else players_available,
        help="Pick players from the filtered cohort.",
    )
    selected_players = selected_players[: int(n_players)]

    if len(selected_players) < 2:
        st.info("Select at least 2 players.")
        st.stop()

    # Pick *one row per player* (if player appears multiple times: choose best minutes within cohort)
    tmp = df_f.copy()
    tmp["_minutes"] = to_num(tmp["Minutes played"]).fillna(0)
    chosen = (
        tmp[tmp["Player"].isin(selected_players)]
        .sort_values(["Player", "_minutes"], ascending=[True, False])
        .groupby("Player", as_index=False)
        .head(1)
        .copy()
    )

    # Label = Player (Season)
    chosen["Label"] = chosen.apply(lambda r: f"{r['Player']} ({r['Season']})" if pd.notna(r.get("Season")) else str(r["Player"]), axis=1)
    selected_rows_labeled = chosen.set_index("Label")
    labels = selected_rows_labeled.index.tolist()

else:
    # same player across seasons
    p = st.selectbox("Player", options=players_available)
    seasons_for_player = unique_sorted(df_f[df_f["Player"].astype(str) == str(p)]["Season"])

    if len(seasons_for_player) < 2:
        st.info("This player has fewer than 2 seasons in the current cohort filters. Expand Season/League filters.")
        st.stop()

    seasons_pick = st.multiselect(
        "Select seasons for this player",
        options=seasons_for_player,
        default=seasons_for_player[:2],
        help="Pick two (or more) seasons to compare for the same player.",
    )
    seasons_pick = seasons_pick[: int(n_players)]  # respect slots

    if len(seasons_pick) < 2:
        st.info("Select at least 2 seasons for this player.")
        st.stop()

    tmp = df_f[(df_f["Player"].astype(str) == str(p)) & (df_f["Season"].astype(str).isin([str(s) for s in seasons_pick]))].copy()
    tmp["_minutes"] = to_num(tmp["Minutes played"]).fillna(0)

    # If duplicates per season (e.g., mid-season transfer), choose max minutes row per season
    chosen = (
        tmp.sort_values(["Season", "_minutes"], ascending=[True, False])
        .groupby("Season", as_index=False)
        .head(1)
        .copy()
    )
    chosen["Label"] = chosen.apply(lambda r: f"{r['Player']} ({r['Season']})", axis=1)
    selected_rows_labeled = chosen.set_index("Label")
    labels = selected_rows_labeled.index.tolist()

    if len(labels) < 2:
        st.info("Not enough season rows found for this player.")
        st.stop()

# Update n_players to actual selected count (important if user selected fewer)
n_sel = len(labels)
if n_sel < 2:
    st.info("Select at least 2 comparable entries.")
    st.stop()

# Color map aligned to labels
color_map = {labels[i]: color_picks[i % len(color_picks)] for i in range(n_sel)}

# -----------------------------
# Metric selection: Profiles + manual
# -----------------------------
st.markdown("### Metrics")

profile_names = ["(Manual)"] + sorted(list(PROFILES.keys())) if PROFILES else ["(Manual)"]
profile_choice = st.selectbox("Metric profile (optional)", profile_names, index=0)

# Keep selection in session_state
if "compare_metrics" not in st.session_state:
    st.session_state["compare_metrics"] = []

if "compare_profile" not in st.session_state:
    st.session_state["compare_profile"] = "(Manual)"

# Apply profile if changed and exists
if profile_choice != st.session_state["compare_profile"]:
    st.session_state["compare_profile"] = profile_choice
    if profile_choice != "(Manual)" and profile_choice in PROFILES:
        # keep only metrics that exist
        st.session_state["compare_metrics"] = [m for m in PROFILES[profile_choice] if m in df2.columns]

# Manual selection always available (also works with profile as starting point)
selected_metrics = st.multiselect(
    "Select metrics for comparison",
    options=numeric_cols,
    default=st.session_state["compare_metrics"] if st.session_state["compare_metrics"] else [],
)

# Persist
st.session_state["compare_metrics"] = selected_metrics

if len(selected_metrics) < 3:
    st.info("Select at least 3 metrics (radar + z-score plot becomes meaningful).")
    st.stop()

# -----------------------------
# Cohort benchmark dataframe (for percentiles/z-scores)
#   Use df_f as benchmark base. For cleaner benchmarking, ignore Team filter if Team != All (optional).
# -----------------------------
bench_df = df_f.copy()

# Optional: you can benchmark against the whole league/posgroup even if a specific team is selected.
# If you want that, uncomment:
# if team_selected != "All":
#     bench_df = df2.copy()
#     if seasons_selected:
#         bench_df = bench_df[bench_df["Season"].astype(str).isin([str(s) for s in seasons_selected])]
#     if leagues_selected:
#         bench_df = bench_df[bench_df[LEAGUE_COL].astype(str).isin([str(l) for l in leagues_selected])]
#     bench_df["Minutes played"] = to_num(bench_df["Minutes played"]).fillna(0)
#     bench_df = bench_df[bench_df["Minutes played"] >= min_minutes]
#     if pos_selected != "All":
#         bench_df = bench_df[bench_df[POS_COL].astype(str) == str(pos_selected)]
#     if pos_group_selected != "All":
#         bench_df = bench_df[bench_df["Position Group"].astype(str) == str(pos_group_selected)]

# Ensure numeric columns are numeric in benchmark/players
for m in selected_metrics:
    bench_df[m] = to_num(bench_df[m])
    selected_rows_labeled[m] = to_num(selected_rows_labeled[m])

# -----------------------------
# Profiles (player cards)
# -----------------------------
st.markdown("### Profiles")
cols = st.columns(n_sel)

for i, label in enumerate(labels):
    row = selected_rows_labeled.loc[label]
    with cols[i]:
        st.markdown(f"**{label}**")
        # Small details (safe if missing)
        def sget(col):
            return row[col] if col in selected_rows_labeled.columns else np.nan

        st.caption(f"Season: {sget('Season')}")
        st.caption(f"Competition: {sget(LEAGUE_COL)}")
        st.caption(f"Team: {sget(TEAM_COL)}")
        st.caption(f"Main Position: {sget(POS_COL)}")
        if "Age" in selected_rows_labeled.columns:
            st.caption(f"Age: {to_num(pd.Series([sget('Age')])).iloc[0] if pd.notna(sget('Age')) else '—'}")
        st.caption(f"Minutes played: {int(to_num(pd.Series([sget('Minutes played')])).fillna(0).iloc[0])}")
        if "Matches played" in selected_rows_labeled.columns:
            st.caption(f"Matches played: {int(to_num(pd.Series([sget('Matches played')])).fillna(0).iloc[0])}")

st.divider()

# -----------------------------
# Raw values table (no scale) — round 2 decimals
# -----------------------------
st.markdown("### Stat breakdown table (no scale)")

raw_tbl = pd.DataFrame(index=selected_metrics)
for label in labels:
    raw_tbl[label] = [float(selected_rows_labeled.loc[label].get(m, np.nan)) for m in selected_metrics]

raw_tbl = raw_tbl.round(2)

# Show (keep it plain)
st.dataframe(raw_tbl, use_container_width=True, height=380)

# -----------------------------
# Percentiles table (0–100), optional highlighting
# -----------------------------
st.markdown("### Percentiles (0–100)")

pct_tbl = compute_percentiles(bench_df, selected_rows_labeled, selected_metrics).reindex(selected_metrics)
pct_tbl = pct_tbl.round(2)

# Styled (background gradient) with safe colormap (no custom cmap names)
# Use plotly-like scale via pandas built-in (matplotlib required) can be brittle in Streamlit.
# We'll do simple winner highlighting instead (robust).
def highlight_winner(row):
    vals = row.values.astype(float)
    out = [""] * len(vals)
    if np.all(np.isnan(vals)):
        return out
    j = int(np.nanargmax(vals))
    out[j] = "font-weight: 700;"
    return out

try:
    styled = pct_tbl.style.apply(highlight_winner, axis=1)
    st.dataframe(styled, use_container_width=True, height=380)
except Exception:
    # Fallback if pandas styler fails in environment
    st.dataframe(pct_tbl, use_container_width=True, height=380)

# -----------------------------
# Radar (percentiles) — include profile name if applicable
# -----------------------------
radar_title = "Radar (percentiles)"
if profile_choice != "(Manual)":
    radar_title += f" — {profile_choice}"

fig_radar = radar_percentiles(
    pct_tbl.fillna(0.0),
    title=radar_title,
    color_map=color_map,
)
st.plotly_chart(fig_radar, use_container_width=True)

# -----------------------------
# Z-scores (combined subset) — new feature
# -----------------------------
st.markdown("### Z-scores (combined subset)")

fig_z = zscore_stripplot(
    cohort_df=bench_df,
    players_df_labeled=selected_rows_labeled,
    metrics=selected_metrics,
    title=f"Z-scores within combined subset ({bench_label})",
    color_map=color_map,
)
st.plotly_chart(fig_z, use_container_width=True)

# -----------------------------
# Export (robust)
# -----------------------------
st.markdown("### Export")

# CSV exports
csv_raw = raw_tbl.reset_index().rename(columns={"index": "Metric"}).to_csv(index=False).encode("utf-8")
st.download_button("Download raw values (CSV)", csv_raw, file_name="compare_raw.csv", mime="text/csv")

csv_pct = pct_tbl.reset_index().rename(columns={"index": "Metric"}).to_csv(index=False).encode("utf-8")
st.download_button("Download percentiles (CSV)", csv_pct, file_name="compare_percentiles.csv", mime="text/csv")

# HTML export (quick single-file report)
html = io.StringIO()
html.write(f"<h2>Compare report</h2>")
html.write(f"<p><b>Benchmark:</b> {bench_label}</p>")
html.write(f"<p><b>Profile:</b> {profile_choice}</p>")
html.write("<h3>Raw values</h3>")
html.write(raw_tbl.to_html())
html.write("<h3>Percentiles</h3>")
html.write(pct_tbl.to_html())
html_bytes = html.getvalue().encode("utf-8")
st.download_button("Download report tables (HTML)", html_bytes, file_name="compare_report_tables.html", mime="text/html")

# Optional: radar PNG via kaleido (works only if Kaleido available + Chromium available)
with st.expander("Radar PNG export (optional)", expanded=False):
    try:
        import plotly.io as pio

        radar_png = pio.to_image(fig_radar, format="png", width=1000, height=700)
        st.download_button("Download radar (PNG)", radar_png, file_name="radar.png", mime="image/png")
        st.success("Radar PNG export ready.")
    except Exception as e:
        st.info(
            "Radar PNG export unavailable. If you're on Streamlit Cloud, ensure "
            "`kaleido==0.2.1.post1` is in requirements.txt (bundled Chromium)."
        )
        st.caption(f"Export debug: {e}")
