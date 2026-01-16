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
team_col = "Team within selected timeframe" if "Team within selected timeframe" in df.columns else ("Team" if "Team" in df.columns else None)
pos_col = "Main Position" if "Main Position" in df.columns else ("Position" if "Position" in df.columns else None)

if league_col is None or team_col is None or pos_col is None:
    st.error("Missing required context columns. Need League/Competition, Team, and Position columns.")
    st.stop()

# Clean strings
for col in ["Player", league_col, team_col, pos_col]:
    df[col] = df[col].astype(str).str.strip()

# ----------------------------
# Top controls
# ----------------------------
c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    n_players = st.number_input("Number of players to compare", 2, 6, 2, 1)

with c2:
    # Optional season filter (if you have Season parsed)
    season_options = ["All"]
    if "Season" in df.columns:
        s = df["Season"].dropna().astype(str).str.strip()
        s = s[(s != "") & (s.str.lower() != "nan")]
        season_options += sorted(s.unique().tolist())
    season = st.selectbox("Season", season_options, index=0)

with c3:
    color_scale = st.selectbox(
        "Color scale",
        options=["Viridis", "Cividis", "Plasma", "Magma", "Inferno", "Turbo", "Blues", "Greens", "Reds", "RdBu", "RdYlGn"],
        index=0,
    )

# Apply season filter globally (affects all pickers)
df_base = df.copy()
if season != "All" and "Season" in df_base.columns:
    df_base = df_base[df_base["Season"].astype(str) == str(season)]

st.divider()

# ----------------------------
# Helpers
# ----------------------------
def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def per90(values: pd.Series, minutes: pd.Series) -> pd.Series:
    v = to_num(values)
    m = to_num(minutes).replace(0, np.nan)
    return (v / m) * 90.0

def percentile_rank(value, population: pd.Series):
    pop = to_num(population).dropna()
    if pop.empty or pd.isna(value):
        return np.nan
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

# Metrics (safe default, adjust later to match your export column names)
candidate_totals = [
    "Goals", "Assists", "xG", "xA", "Shots", "Key passes",
    "Successful dribbles", "Progressive passes", "Progressive runs",
    "Interceptions", "Tackles", "Duels",
]
available_totals = [c for c in candidate_totals if c in df_base.columns]

st.subheader("Metrics")
if not available_totals:
    st.info("No default metrics found. We can map your exact Wyscout columns next.")
    st.stop()

metrics = st.multiselect(
    "Select metrics (totals; we will compute per 90)",
    options=available_totals,
    default=available_totals[:8],
)
if not metrics:
    st.stop()

st.divider()

# ----------------------------
# Side-by-side player selectors (N columns)
# ----------------------------
st.subheader("Pick players (side by side)")
st.caption("Each slot is filtered: League → Team → Main Position → Player.")

# For responsive layout: 2–3 columns per row
slots_per_row = 3
rows = (int(n_players) + slots_per_row - 1) // slots_per_row

selections = []

slot_idx = 0
for r in range(rows):
    cols = st.columns(min(slots_per_row, int(n_players) - r * slots_per_row))
    for c in cols:
        slot_idx += 1
        with c:
            st.markdown(f"## Player {slot_idx}")

            # League
            leagues = sorted(df_base[league_col].dropna().astype(str).unique().tolist())
            league_val = st.selectbox(
                f"League {slot_idx}",
                options=leagues,
                key=f"league_{slot_idx}",
            )

            df_l = df_base[df_base[league_col].astype(str) == str(league_val)]

            # Team (within timeframe)
            teams = sorted(df_l[team_col].dropna().astype(str).unique().tolist())
            team_val = st.selectbox(
                f"Team {slot_idx}",
                options=teams,
                key=f"team_{slot_idx}",
            )

            df_t = df_l[df_l[team_col].astype(str) == str(team_val)]

            # Main position
            positions = sorted(df_t[pos_col].dropna().astype(str).unique().tolist())
            pos_val = st.selectbox(
                f"Position {slot_idx}",
                options=positions,
                key=f"pos_{slot_idx}",
            )

            df_p = df_t[df_t[pos_col].astype(str) == str(pos_val)]

            # Player
            players = sorted(df_p["Player"].dropna().astype(str).unique().tolist())
            player_val = st.selectbox(
                f"Player {slot_idx}",
                options=players,
                key=f"player_{slot_idx}",
            )

            # Resolve the row (if duplicates exist, pick highest minutes)
            cand = df_p[df_p["Player"].astype(str) == str(player_val)].copy()
            cand["__mins"] = to_num(cand["Minutes played"]).fillna(0)
            cand = cand.sort_values("__mins", ascending=False)

            chosen = cand.iloc[0].drop(labels=["__mins"])
            selections.append(chosen)

# Build selected dataframe
sel_df = pd.DataFrame(selections).reset_index(drop=True)
if sel_df.empty:
    st.stop()

sel_df["Position group"] = sel_df[pos_col].apply(map_position_group)

st.divider()

# ----------------------------
# Profiles row
# ----------------------------
st.subheader("Profiles")

profile_fields = [
    league_col,
    team_col,
    pos_col,
    "Age",
    "Minutes played",
    "Matches played",
    "Market value",
]
profile_fields = [c for c in profile_fields if c in sel_df.columns]

cols = st.columns(len(sel_df))
for i, col in enumerate(cols):
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
# Per-90 and percentile tables
# ----------------------------
st.subheader("Per-90 values and percentiles")

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

per90_tbl = pd.DataFrame(index=[f"{m} per 90" for m in metrics])
pct_tbl = pd.DataFrame(index=[f"{m} per 90" for m in metrics])

# Precompute global pool if needed
global_pool = df_base.copy()
global_pool["Position group"] = global_pool[pos_col].apply(map_position_group)

for i, r in sel_df.iterrows():
    player_col_name = f"{r['Player']}"

    mins = r.get("Minutes played", np.nan)

    # define player-specific peer pool
    if peer_mode.startswith("Within each player's League"):
        pool = df_base[df_base[league_col].astype(str) == str(r[league_col])].copy()
        pool["Position group"] = pool[pos_col].apply(map_position_group)
        if "Position group" in sel_df.columns and "Position group" in pool.columns:
            if "Position group" in pool.columns and peer_mode.endswith("Position group"):
                pool = pool[pool["Position group"] == r["Position group"]]
    else:
        pool = global_pool

    per_vals = []
    pct_vals = []
    for m in metrics:
        v = per90(pd.Series([r.get(m, np.nan)]), pd.Series([mins])).iloc[0]
        pool_per = per90(pool[m], pool["Minutes played"])
        p = percentile_rank(v, pool_per)
        per_vals.append(v)
        pct_vals.append(p)

    per90_tbl[player_col_name] = per_vals
    pct_tbl[player_col_name] = pct_vals

per90_tbl = per90_tbl.round(3)
pct_tbl = pct_tbl.round(1)

st.write("**Per-90 values**")
st.dataframe(per90_tbl, use_container_width=True)

st.write("**Percentiles (0–100)**")
styled = pct_tbl.style.background_gradient(axis=None, cmap=color_scale, vmin=0, vmax=100)
st.dataframe(styled, use_container_width=True)

st.divider()

# ----------------------------
# Radar chart (percentiles)
# ----------------------------
st.subheader("Radar (percentiles)")

radar_metrics = pct_tbl.index.tolist()
if len(radar_metrics) >= 3:
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
else:
    st.info("Select at least 3 metrics to show a radar chart.")
