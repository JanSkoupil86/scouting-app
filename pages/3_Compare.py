import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Compare", layout="wide")
st.title("Compare")

# ----------------------------
# Load data
# ----------------------------
if "data" not in st.session_state:
    st.warning("No data loaded yet. Go to the **app** page and upload a CSV.")
    st.stop()

df = st.session_state["data"]
if not isinstance(df, pd.DataFrame) or df.empty:
    st.error("Loaded dataset is empty or invalid. Re-upload the CSV on the **app** page.")
    st.stop()

required = ["Player", "Minutes played"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Your CSV is missing required columns: {missing}")
    st.stop()

# Column preferences
league_col = "Competition" if "Competition" in df.columns else ("League" if "League" in df.columns else None)
team_col = (
    "Team within selected timeframe"
    if "Team within selected timeframe" in df.columns
    else ("Team" if "Team" in df.columns else None)
)
pos_col = "Main Position" if "Main Position" in df.columns else ("Position" if "Position" in df.columns else None)

if league_col is None or team_col is None or pos_col is None:
    st.error("Missing required columns. Need League/Competition, Team, and Main Position/Position columns.")
    st.stop()

# Clean strings
for col in ["Player", league_col, team_col, pos_col]:
    df[col] = df[col].astype(str).str.strip()

# ----------------------------
# Helpers
# ----------------------------
def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def percentile_rank(value, population: pd.Series):
    pop = to_num(population).dropna()
    if pop.empty or pd.isna(value):
        return np.nan
    # strict percentile (stable with ties)
    return round((pop < float(value)).mean() * 100.0, 1)


def map_position_group(position_str: str) -> str:
    if not position_str:
        return "OTHER"
    pos = str(position_str).upper()
    if "GK" in pos:
        return "GK"
    if any(p in pos for p in ["CB", "LCB", "RCB", "LB", "RB", "LWB", "RWB"]):
        return "DEF"
    if any(p in pos for p in ["DMF", "CMF", "AMF", "LMF", "RMF"]):
        return "MID"
    if any(p in pos for p in ["CF", "SS", "ST", "LWF", "RWF", "LW", "RW"]):
        return "FWD"
    return "OTHER"


def unique_sorted(series: pd.Series) -> list[str]:
    s = series.dropna().astype(str).str.strip()
    s = s[(s != "") & (s.str.lower() != "nan")]
    return sorted(s.unique().tolist())


# ----------------------------
# Top controls
# ----------------------------
c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    n_players = st.number_input("Number of players to compare", 2, 6, 2, 1)

with c2:
    season_options = ["All"]
    if "Season" in df.columns:
        season_options += unique_sorted(df["Season"])
    season = st.selectbox("Season", season_options, index=0)

with c3:
    color_scale = st.selectbox(
        "Color scale",
        options=[
            "Viridis",
            "Cividis",
            "Plasma",
            "Magma",
            "Inferno",
            "Turbo",
            "Blues",
            "Greens",
            "Reds",
            "RdBu",
            "RdYlGn",
        ],
        index=0,
    )

# Apply global season filter (affects all pickers + all peer pools)
df_base = df.copy()
if season != "All" and "Season" in df_base.columns:
    df_base = df_base[df_base["Season"].astype(str) == str(season)]

if df_base.empty:
    st.info("No rows available after Season filtering.")
    st.stop()

st.divider()

# ----------------------------
# Metrics catalogue (your list)
# ----------------------------
METRICS_CATALOGUE = [
    "Goals", "xG", "Assists", "xA",
    "Duels per 90", "Duels won, %", "Successful defensive actions per 90",
    "Defensive duels per 90", "Defensive duels won, %", "Aerial duels per 90", "Aerial duels won, %",
    "Sliding tackles per 90", "PAdj Sliding tackles", "Shots blocked per 90",
    "Interceptions per 90", "PAdj Interceptions",
    "Fouls per 90", "Yellow cards", "Yellow cards per 90", "Red cards", "Red cards per 90",
    "Successful attacking actions per 90",
    "Goals per 90", "Non-penalty goals", "Non-penalty goals per 90",
    "xG per 90", "Head goals", "Head goals per 90",
    "Shots", "Shots per 90", "Shots on target, %", "Goal conversion, %",
    "Assists per 90",
    "Crosses per 90", "Accurate crosses, %",
    "Crosses from left flank per 90", "Accurate crosses from left flank, %",
    "Crosses from right flank per 90", "Accurate crosses from right flank, %",
    "Crosses to goalie box per 90",
    "Dribbles per 90", "Successful dribbles, %",
    "Offensive duels per 90", "Offensive duels won, %",
    "Touches in box per 90", "Progressive runs per 90", "Accelerations per 90",
    "Received passes per 90", "Received long passes per 90",
    "Fouls suffered per 90",
    "Passes per 90", "Accurate passes, %",
    "Forward passes per 90", "Accurate forward passes, %",
    "Back passes per 90", "Accurate back passes, %",
    "Lateral passes per 90", "Accurate lateral passes, %",
    "Short / medium passes per 90", "Accurate short / medium passes, %",
    "Long passes per 90", "Accurate long passes, %",
    "Average pass length, m", "Average long pass length, m",
    "xA per 90", "Shot assists per 90", "Second assists per 90", "Third assists per 90",
    "Smart passes per 90", "Accurate smart passes, %",
    "Key passes per 90",
    "Passes to final third per 90", "Accurate passes to final third, %",
    "Passes to penalty area per 90", "Accurate passes to penalty area, %",
    "Through passes per 90", "Accurate through passes, %",
    "Deep completions per 90", "Deep completed crosses per 90",
    "Progressive passes per 90", "Accurate progressive passes, %",
    "Conceded goals", "Conceded goals per 90",
    "Shots against", "Shots against per 90",
    "Clean sheets", "Save rate, %", "xG against", "xG against per 90",
    "Prevented goals", "Prevented goals per 90",
    "Back passes received as GK per 90", "Exits per 90", "Aerial duels per 90.1",
    "Free kicks per 90", "Direct free kicks per 90", "Direct free kicks on target, %",
    "Corners per 90", "Penalties taken", "Penalty conversion, %",
]

# Lower is better → invert percentiles
LOWER_BETTER = {
    "Fouls per 90",
    "Yellow cards", "Yellow cards per 90",
    "Red cards", "Red cards per 90",
    "Conceded goals", "Conceded goals per 90",
    "Shots against", "Shots against per 90",
    "xG against", "xG against per 90",
}

available_metrics = [m for m in METRICS_CATALOGUE if m in df_base.columns]

st.subheader("Metrics")
if not available_metrics:
    st.warning("None of the requested metrics were found in your dataset columns.")
    st.stop()

default_metrics = [
    m for m in [
        "Goals per 90", "xG per 90", "Assists per 90", "xA per 90",
        "Duels won, %", "Interceptions per 90", "Progressive passes per 90", "Key passes per 90",
    ]
    if m in available_metrics
]
if not default_metrics:
    default_metrics = available_metrics[:10]

metrics = st.multiselect(
    "Select metrics for comparison",
    options=available_metrics,
    default=default_metrics,
)

if not metrics:
    st.stop()

st.divider()

# ----------------------------
# Side-by-side player pickers
# ----------------------------
st.subheader("Pick players (side by side)")
st.caption("Each slot is filtered: League → Team → Main Position → Player (Season is applied globally at the top).")

slots_per_row = 3
rows = (int(n_players) + slots_per_row - 1) // slots_per_row

selections = []
slot_idx = 0

for r in range(rows):
    cols = st.columns(min(slots_per_row, int(n_players) - r * slots_per_row))
    for col in cols:
        slot_idx += 1
        with col:
            st.markdown(f"## Player {slot_idx}")

            leagues = unique_sorted(df_base[league_col])
            if not leagues:
                st.error("No leagues available.")
                st.stop()

            league_val = st.selectbox(
                f"League {slot_idx}",
                options=leagues,
                key=f"league_{slot_idx}",
            )
            df_l = df_base[df_base[league_col].astype(str) == str(league_val)]

            teams = unique_sorted(df_l[team_col]) if not df_l.empty else []
            if not teams:
                st.warning("No teams available for this league.")
                continue

            team_val = st.selectbox(
                f"Team {slot_idx}",
                options=teams,
                key=f"team_{slot_idx}",
            )
            df_t = df_l[df_l[team_col].astype(str) == str(team_val)]

            positions = unique_sorted(df_t[pos_col]) if not df_t.empty else []
            if not positions:
                st.warning("No positions available for this team.")
                continue

            pos_val = st.selectbox(
                f"Position {slot_idx}",
                options=positions,
                key=f"pos_{slot_idx}",
            )
            df_p = df_t[df_t[pos_col].astype(str) == str(pos_val)]

            players = unique_sorted(df_p["Player"]) if not df_p.empty else []
            if not players:
                st.warning("No players available for this position.")
                continue

            player_val = st.selectbox(
                f"Player {slot_idx}",
                options=players,
                key=f"player_{slot_idx}",
            )

            # Resolve row (if duplicates, take highest minutes)
            cand = df_p[df_p["Player"].astype(str) == str(player_val)].copy()
            cand["__mins"] = to_num(cand["Minutes played"]).fillna(0)
            cand = cand.sort_values("__mins", ascending=False)

            chosen = cand.iloc[0].drop(labels=["__mins"])
            selections.append(chosen)

sel_df = pd.DataFrame(selections).reset_index(drop=True)

if len(sel_df) != int(n_players):
    st.info("Please complete selection in all player slots to continue.")
    st.stop()

# Add position group (for peer pool options)
sel_df = sel_df.copy()
sel_df["Position group"] = sel_df[pos_col].apply(map_position_group)

st.divider()

# ----------------------------
# Profiles
# ----------------------------
st.subheader("Profiles")

profile_fields = [
    league_col, team_col, pos_col, "Position group",
    "Age", "Minutes played", "Matches played", "Market value",
]
profile_fields = [c for c in profile_fields if c in sel_df.columns]

pcols = st.columns(len(sel_df))
for i, col in enumerate(pcols):
    with col:
        st.markdown(f"### {sel_df.iloc[i]['Player']}")
        for f in profile_fields:
            v = sel_df.iloc[i].get(f, "")
            if f in ["Minutes played", "Matches played", "Age"] and pd.notna(v):
                try:
                    v = int(float(v))
                except Exception:
                    pass
            if f == "Market value" and pd.notna(v):
                try:
                    v = f"{int(float(v)):,}"
                except Exception:
                    pass
            st.write(f"**{f}:** {v}")

st.divider()

# ----------------------------
# Percentile peer pool mode
# ----------------------------
st.subheader("Values and percentiles")

peer_mode = st.radio(
    "Percentile peer pool",
    options=[
        "Within each player's League + Position group",
        "Within each player's League (no position grouping)",
        "Global (all filtered data)",
    ],
    index=0,
    horizontal=False,
)

# Precompute global pool (season filtered)
global_pool = df_base.copy()
global_pool["Position group"] = global_pool[pos_col].apply(map_position_group)

values_tbl = pd.DataFrame(index=metrics)
pct_tbl = pd.DataFrame(index=metrics)

for i, r in sel_df.iterrows():
    player_name = str(r["Player"]).strip()
    player_league = str(r[league_col]).strip()
    player_group = str(r["Position group"]).strip()

    if peer_mode == "Global (all filtered data)":
        pool = global_pool
    else:
        pool = df_base[df_base[league_col].astype(str) == player_league].copy()
        pool["Position group"] = pool[pos_col].apply(map_position_group)
        if peer_mode == "Within each player's League + Position group":
            pool = pool[pool["Position group"] == player_group]

    vals = []
    pcts = []
    for m in metrics:
        v = to_num(pd.Series([r.get(m, np.nan)])).iloc[0]
        pop = pool[m] if m in pool.columns else pd.Series(dtype=float)
        pct = percentile_rank(v, pop)

        if m in LOWER_BETTER and not pd.isna(pct):
            pct = round(100.0 - pct, 1)

        vals.append(v)
        pcts.append(pct)

    values_tbl[player_name] = vals
    pct_tbl[player_name] = pcts

values_tbl = values_tbl.round(3)
pct_tbl = pct_tbl.round(1)

st.write("**Values (raw columns)**")
st.dataframe(values_tbl, use_container_width=True)

st.write("**Percentiles (0–100)**")
styled = pct_tbl.style.background_gradient(axis=None, cmap=color_scale, vmin=0, vmax=100)
st.dataframe(styled, use_container_width=True)

# Download
csv_out = pct_tbl.reset_index().rename(columns={"index": "Metric"}).to_csv(index=False).encode("utf-8")
st.download_button(
    "Download percentiles as CSV",
    data=csv_out,
    file_name="compare_percentiles.csv",
    mime="text/csv",
)

st.divider()

# ----------------------------
# Radar chart (percentiles)
# ----------------------------
st.subheader("Radar (percentiles)")

radar_metrics = pct_tbl.index.tolist()
if len(radar_metrics) < 3:
    st.info("Select at least 3 metrics to show a radar chart.")
    st.stop()

fig = go.Figure()
for col in pct_tbl.columns:
    vals = pct_tbl[col].tolist()
    fig.add_trace(
        go.Scatterpolar(
            r=vals + [vals[0]],
            theta=radar_metrics + [radar_metrics[0]],
            fill="none",
            name=col,
        )
    )

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
    showlegend=True,
    height=520,
    margin=dict(l=20, r=20, t=30, b=20),
)

st.plotly_chart(fig, use_container_width=True)
