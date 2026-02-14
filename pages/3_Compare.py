# pages/3_Compare.py
# FULL Compare page (with fixes):
# - Sidebar filters: Season(s), League(s), Minimum minutes, ✅ Age range (below minutes),
#   Team (dependent on chosen league/season), Main Position (from Main Position), Position Group benchmark.
# - Compare modes: Compare players OR Same player across seasons
# - Metrics: Manual or Profile presets (src/profiles.py)
# - Tables: Raw values (rounded 2), Percentiles (0–100)
# - Radar (percentiles) with profile name in title
# - ✅ Z-score stripplot with rich hover on selected diamonds (player/team/season/league/minutes/age/raw/z)

import re
import io
import numpy as np
import pandas as pd
import streamlit as st
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
    m = (metric_name or "").lower()
    bad_tokens = [
        "conceded", "against", "foul", "yellow", "red", "error",
        "shots against", "xg against", "cards",
    ]
    return any(t in m for t in bad_tokens)

def safe_metric_list(df_: pd.DataFrame) -> list[str]:
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
    rows=metrics, cols=player labels, values=percentiles (0-100).
    Inverts if lower is better.
    """
    result = {}
    for m in metrics:
        if m not in cohort.columns:
            continue
        s = to_num(cohort[m]).dropna()
        if len(s) < 10:
            continue
        cohort_vals_sorted = np.sort(s.values)

        def pct_of_value(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return np.nan
            return float(np.searchsorted(cohort_vals_sorted, v, side="right")) / len(cohort_vals_sorted) * 100.0

        col = {}
        for label, row in players.iterrows():
            v = to_num(pd.Series([row.get(m, np.nan)])).iloc[0]
            p = pct_of_value(v)
            if is_lower_better(m) and np.isfinite(p):
                p = 100.0 - p
            col[label] = p
        result[m] = col

    return pd.DataFrame(result).T

def radar_percentiles(pct_df: pd.DataFrame, title: str, color_map: dict[str, str]) -> go.Figure:
    metrics = pct_df.index.tolist()
    fig = go.Figure()

    for label in pct_df.columns:
        vals = pct_df[label].astype(float).values
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
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=520,
        margin=dict(l=30, r=30, t=60, b=30),
        legend=dict(orientation="v"),
    )
    return fig

def zscore_stripplot(
    cohort_df: pd.DataFrame,
    players_df_labeled: pd.DataFrame,
    metrics: list[str],
    title: str,
    color_map: dict[str, str],
    team_col: str,
    league_col: str,
) -> go.Figure:
    """
    Grey dots = cohort z distribution; diamonds = selected players (labeled index).
    Hover on diamonds shows player/team/season/league/minutes/age/raw/z.
    Right = better (inverts lower-better metrics).
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

    # Cohort stats per metric
    mu_map, sd_map, z_cohort = {}, {}, {}
    for m in usable:
        s = to_num(cohort_df[m])
        mu = float(s.mean(skipna=True))
        sd = float(s.std(skipna=True))
        mu_map[m] = mu
        sd_map[m] = sd if sd > 1e-9 else np.nan

        if not np.isfinite(sd_map[m]):
            z = s * np.nan
        else:
            z = (s - mu_map[m]) / sd_map[m]
            if is_lower_better(m):
                z = -z
        z_cohort[m] = z.dropna().values

    fig = make_subplots(
        rows=len(usable),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=usable,
    )

    for i, m in enumerate(usable, start=1):
        zvals = z_cohort[m]
        if len(zvals) == 0:
            continue

        # Cohort distribution dots (hover just shows z)
        fig.add_trace(
            go.Scatter(
                x=zvals,
                y=np.zeros(len(zvals)),
                mode="markers",
                marker=dict(size=6, color="rgba(140,140,140,0.45)"),
                hovertemplate=str(m) + "<br>z=%{x:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=i,
            col=1,
        )

        # Avg line
        fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="rgba(80,80,80,0.65)", row=i, col=1)

        # Selected diamonds with rich hover
        for label, row in players_df_labeled.iterrows():
            raw_val = to_num(pd.Series([row.get(m, np.nan)])).iloc[0]
            if pd.isna(raw_val) or not np.isfinite(sd_map[m]):
                continue

            z = (float(raw_val) - mu_map[m]) / sd_map[m]
            if is_lower_better(m):
                z = -z

            season = row.get("Season", "—")
            team = row.get(team_col, "—")
            league = row.get(league_col, "—")
            minutes = row.get("Minutes played", "—")
            age = row.get("Age", "—") if "Age" in row.index else "—"

            custom = [[team, season, league, minutes, age, float(raw_val)]]

            fig.add_trace(
                go.Scatter(
                    x=[z],
                    y=[0],
                    mode="markers+text",
                    text=[label],
                    textposition="top center",
                    marker=dict(size=10, symbol="diamond", color=color_map.get(label, "#1f77b4")),
                    customdata=custom,
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Squad: %{customdata[0][0]}<br>"
                        "Season: %{customdata[0][1]}<br>"
                        "League: %{customdata[0][2]}<br>"
                        "Minutes: %{customdata[0][3]}<br>"
                        "Age: %{customdata[0][4]}<br>"
                        + str(m) + ": %{customdata[0][5]:.2f}<br>"
                        "Z: %{x:.2f}"
                        "<extra></extra>"
                    ),
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
        height=max(420, 150 * len(usable)),
        margin=dict(l=20, r=20, t=70, b=40),
    )
    fig.add_annotation(x=-2.5, y=1.0, xref="x", yref="paper", text="Worse", showarrow=False, opacity=0.55)
    fig.add_annotation(x=2.5, y=1.0, xref="x", yref="paper", text="Better", showarrow=False, opacity=0.55)
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

numeric_cols = safe_metric_list(df2)
if not numeric_cols:
    st.error("No numeric metrics detected.")
    st.stop()

# -----------------------------
# Sidebar filters (with Age slider below minutes)
# -----------------------------
with st.sidebar:
    st.header("Player Filters")

    seasons_all = unique_sorted(df2["Season"])
    seasons_selected = st.multiselect(
        "Season(s)",
        options=seasons_all,
        default=seasons_all[:1] if seasons_all else [],
    )

    df_f = df2.copy()
    if seasons_selected:
        df_f = df_f[df_f["Season"].astype(str).isin([str(s) for s in seasons_selected])]

    leagues_all = unique_sorted(df_f[LEAGUE_COL])
    leagues_selected = st.multiselect(
        "League(s)",
        options=leagues_all,
        default=leagues_all[:1] if leagues_all else [],
    )
    if leagues_selected:
        df_f = df_f[df_f[LEAGUE_COL].astype(str).isin([str(l) for l in leagues_selected])]

    # minutes
    df_f["Minutes played"] = to_num(df_f["Minutes played"]).fillna(0)
    max_mins = int(df_f["Minutes played"].max()) if not df_f.empty else 0
    min_minutes = st.slider(
        "Minimum minutes",
        0,
        max(200, max_mins),
        min(600, max_mins) if max_mins else 0,
        step=50
    )
    df_f = df_f[df_f["Minutes played"] >= min_minutes]

    # ✅ Age range below minutes
    age_range = None
    if "Age" in df_f.columns:
        df_f["Age"] = to_num(df_f["Age"])
        age_vals = df_f["Age"].dropna()
        if not age_vals.empty:
            a_min = int(np.floor(age_vals.min()))
            a_max = int(np.ceil(age_vals.max()))
            age_range = st.slider("Age range", a_min, a_max, (a_min, a_max), step=1)
            df_f = df_f[(df_f["Age"].isna()) | ((df_f["Age"] >= age_range[0]) & (df_f["Age"] <= age_range[1]))]

    # Team (dependent)
    teams_all = unique_sorted(df_f[TEAM_COL])
    team_selected = st.selectbox("Team", ["All"] + teams_all, index=0)
    if team_selected != "All":
        df_f = df_f[df_f[TEAM_COL].astype(str) == str(team_selected)]

    # Main Position
    pos_all = unique_sorted(df_f[POS_COL])
    pos_selected = st.selectbox("Main Position", ["All"] + pos_all, index=0)
    if pos_selected != "All":
        df_f = df_f[df_f[POS_COL].astype(str) == str(pos_selected)]

    # Position Group benchmark
    df_f["Position Group"] = df_f[POS_COL].apply(position_group)
    pos_group_all = unique_sorted(df_f["Position Group"])
    pos_group_selected = st.selectbox("Position Group (benchmark)", ["All"] + pos_group_all, index=0)
    if pos_group_selected != "All":
        df_f = df_f[df_f["Position Group"].astype(str) == str(pos_group_selected)]

if df_f.empty:
    st.info("No players match the cohort filters. Adjust Season/League/Minutes/Age.")
    st.stop()

league_label = " / ".join(leagues_selected) if leagues_selected else "All leagues"
season_label = " / ".join(seasons_selected) if seasons_selected else "All seasons"
bench_label = f"{pos_group_selected if pos_group_selected != 'All' else 'All positions'} | {league_label} | {season_label}"

# -----------------------------
# Selection
# -----------------------------
st.markdown("### Selection")

mode = st.radio(
    "Compare mode",
    ["Compare players", "Same player across seasons"],
    horizontal=True,
)

n_slots = st.number_input("Number of players to compare", min_value=2, max_value=5, value=2, step=1)

with st.expander("🎨 Customize player colors", expanded=False):
    default_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    color_picks = []
    for i in range(int(n_slots)):
        color_picks.append(st.color_picker(f"Slot {i+1} color", value=default_palette[i % len(default_palette)]))

players_available = unique_sorted(df_f["Player"])

selected_rows_labeled = pd.DataFrame()
labels = []

if mode == "Compare players":
    selected_players = st.multiselect(
        "Players to compare",
        options=players_available,
        default=players_available[: int(n_slots)] if len(players_available) >= int(n_slots) else players_available,
    )
    selected_players = selected_players[: int(n_slots)]

    if len(selected_players) < 2:
        st.info("Select at least 2 players.")
        st.stop()

    tmp = df_f.copy()
    tmp["_minutes"] = to_num(tmp["Minutes played"]).fillna(0)

    chosen = (
        tmp[tmp["Player"].isin(selected_players)]
        .sort_values(["Player", "_minutes"], ascending=[True, False])
        .groupby("Player", as_index=False)
        .head(1)
        .copy()
    )

    chosen["Label"] = chosen.apply(
        lambda r: f"{r['Player']} ({r['Season']})" if pd.notna(r.get("Season")) else str(r["Player"]),
        axis=1
    )
    selected_rows_labeled = chosen.set_index("Label")
    labels = selected_rows_labeled.index.tolist()

else:
    p = st.selectbox("Player", options=players_available)
    seasons_for_player = unique_sorted(df_f[df_f["Player"].astype(str) == str(p)]["Season"])
    if len(seasons_for_player) < 2:
        st.info("This player has fewer than 2 seasons in the current cohort filters. Expand Season/League.")
        st.stop()

    seasons_pick = st.multiselect(
        "Select seasons for this player",
        options=seasons_for_player,
        default=seasons_for_player[:2],
    )
    seasons_pick = seasons_pick[: int(n_slots)]

    if len(seasons_pick) < 2:
        st.info("Select at least 2 seasons.")
        st.stop()

    tmp = df_f[(df_f["Player"].astype(str) == str(p)) & (df_f["Season"].astype(str).isin([str(s) for s in seasons_pick]))].copy()
    tmp["_minutes"] = to_num(tmp["Minutes played"]).fillna(0)

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
    st.info("Select at least 2 comparable entries.")
    st.stop()

color_map = {labels[i]: color_picks[i % len(color_picks)] for i in range(len(labels))}

# -----------------------------
# Metrics
# -----------------------------
st.markdown("### Metrics")

profile_names = ["(Manual)"] + sorted(list(PROFILES.keys())) if PROFILES else ["(Manual)"]

if "compare_profile" not in st.session_state:
    st.session_state["compare_profile"] = "(Manual)"
if "compare_metrics" not in st.session_state:
    st.session_state["compare_metrics"] = []

profile_choice = st.selectbox(
    "Metric profile (optional)",
    profile_names,
    index=profile_names.index(st.session_state["compare_profile"]) if st.session_state["compare_profile"] in profile_names else 0
)

if profile_choice != st.session_state["compare_profile"]:
    st.session_state["compare_profile"] = profile_choice
    if profile_choice != "(Manual)" and profile_choice in PROFILES:
        st.session_state["compare_metrics"] = [m for m in PROFILES[profile_choice] if m in df2.columns]

selected_metrics = st.multiselect(
    "Select metrics for comparison",
    options=numeric_cols,
    default=st.session_state["compare_metrics"] if st.session_state["compare_metrics"] else [],
)
st.session_state["compare_metrics"] = selected_metrics

if len(selected_metrics) < 3:
    st.info("Select at least 3 metrics (radar + z-scores becomes meaningful).")
    st.stop()

# Benchmark cohort
bench_df = df_f.copy()
for m in selected_metrics:
    bench_df[m] = to_num(bench_df[m])
    selected_rows_labeled[m] = to_num(selected_rows_labeled[m])

# -----------------------------
# Profiles (cards)
# -----------------------------
st.markdown("### Profiles")
cols = st.columns(len(labels))

for i, label in enumerate(labels):
    row = selected_rows_labeled.loc[label]
    with cols[i]:
        st.markdown(f"**{label}**")
        st.caption(f"Season: {row.get('Season', '—')}")
        st.caption(f"Competition: {row.get(LEAGUE_COL, '—')}")
        st.caption(f"Squad: {row.get(TEAM_COL, '—')}")
        st.caption(f"Main Position: {row.get(POS_COL, '—')}")
        if "Age" in selected_rows_labeled.columns:
            age_v = to_num(pd.Series([row.get("Age", np.nan)])).iloc[0]
            st.caption(f"Age: {int(age_v) if pd.notna(age_v) else '—'}")
        mins_v = int(to_num(pd.Series([row.get("Minutes played", 0)])).fillna(0).iloc[0])
        st.caption(f"Minutes played: {mins_v}")

st.divider()

# -----------------------------
# Raw table
# -----------------------------
st.markdown("### Stat breakdown table (no scale)")
raw_tbl = pd.DataFrame(index=selected_metrics)
for label in labels:
    raw_tbl[label] = [float(selected_rows_labeled.loc[label].get(m, np.nan)) for m in selected_metrics]
raw_tbl = raw_tbl.round(2)
st.dataframe(raw_tbl, use_container_width=True, height=380)

# -----------------------------
# Percentiles
# -----------------------------
st.markdown("### Percentiles (0–100)")
pct_tbl = compute_percentiles(bench_df, selected_rows_labeled, selected_metrics).reindex(selected_metrics).round(2)

def highlight_winner(row):
    vals = row.values.astype(float)
    out = [""] * len(vals)
    if np.all(np.isnan(vals)):
        return out
    j = int(np.nanargmax(vals))
    out[j] = "font-weight: 700;"
    return out

try:
    st.dataframe(pct_tbl.style.apply(highlight_winner, axis=1), use_container_width=True, height=380)
except Exception:
    st.dataframe(pct_tbl, use_container_width=True, height=380)

# -----------------------------
# Radar
# -----------------------------
radar_title = "Radar (percentiles)"
if profile_choice != "(Manual)":
    radar_title += f" — {profile_choice}"

fig_radar = radar_percentiles(pct_tbl.fillna(0.0), radar_title, color_map)
st.plotly_chart(fig_radar, use_container_width=True)

# -----------------------------
# Z-score stripplot (rich hover)
# -----------------------------
st.markdown("### Z-scores (combined subset)")
fig_z = zscore_stripplot(
    cohort_df=bench_df,
    players_df_labeled=selected_rows_labeled,
    metrics=selected_metrics,
    title=f"Z-scores within combined subset ({bench_label})",
    color_map=color_map,
    team_col=TEAM_COL,
    league_col=LEAGUE_COL,
)
st.plotly_chart(fig_z, use_container_width=True)

# -----------------------------
# Export
# -----------------------------
st.markdown("### Export")

csv_raw = raw_tbl.reset_index().rename(columns={"index": "Metric"}).to_csv(index=False).encode("utf-8")
st.download_button("Download raw values (CSV)", csv_raw, file_name="compare_raw.csv", mime="text/csv")

csv_pct = pct_tbl.reset_index().rename(columns={"index": "Metric"}).to_csv(index=False).encode("utf-8")
st.download_button("Download percentiles (CSV)", csv_pct, file_name="compare_percentiles.csv", mime="text/csv")

html = io.StringIO()
html.write("<h2>Compare report</h2>")
html.write(f"<p><b>Benchmark:</b> {bench_label}</p>")
html.write(f"<p><b>Profile:</b> {profile_choice}</p>")
html.write("<h3>Raw values</h3>")
html.write(raw_tbl.to_html())
html.write("<h3>Percentiles</h3>")
html.write(pct_tbl.to_html())
html_bytes = html.getvalue().encode("utf-8")
st.download_button("Download report tables (HTML)", html_bytes, file_name="compare_report_tables.html", mime="text/html")
