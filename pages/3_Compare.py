import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.filters import apply_filters
from src.ui import sidebar_controls

st.set_page_config(page_title="Compare", layout="wide")
st.title("Compare")

# ----------------------------
# Global controls (TOP of page)
# ----------------------------
top_left, top_right = st.columns([1, 1])

with top_left:
    n_players = st.number_input(
        "Number of players to compare",
        min_value=2,
        max_value=6,
        value=2,
        step=1,
    )

with top_right:
    # Plotly continuous scales (names)
    color_scale = st.selectbox(
        "Color scale",
        options=[
            "Blues", "Greens", "Reds", "Purples", "Oranges",
            "Viridis", "Cividis", "Plasma", "Magma", "Inferno",
            "Turbo", "RdBu", "RdYlGn",
        ],
        index=5,
        help="Used for percentile heatmaps and radar styling consistency.",
    )

st.divider()

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

# Prefer timeframe team + main position where available
team_col = "Team within selected timeframe" if "Team within selected timeframe" in df.columns else "Team"
pos_col = "Main Position" if "Main Position" in df.columns else "Position"

for col in ["Player", team_col, pos_col]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# ----------------------------
# Sidebar filters (cascading)
# ----------------------------
season, competition, minutes_min, team, position, name_query = sidebar_controls(df)

df_f = apply_filters(
    df,
    season=season,
    competition=competition,
    minutes_min=minutes_min,
    team=team,
    position=position,
    name_query=name_query,
)

if df_f.empty:
    st.info("No players match the current filters. Adjust Season/League/Team/Position/Minutes.")
    st.stop()

# ----------------------------
# Helpers
# ----------------------------
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

# ----------------------------
# Player selection (enforced N)
# ----------------------------
st.subheader("Selection")
st.caption("Select players after narrowing the pool with sidebar filters.")

df_pick = df_f[["Player"]].copy()
if team_col in df_f.columns:
    df_pick[team_col] = df_f[team_col]
else:
    df_pick[team_col] = ""

if pos_col in df_f.columns:
    df_pick[pos_col] = df_f[pos_col]
else:
    df_pick[pos_col] = ""

df_pick = df_pick.dropna().copy()
df_pick["label"] = df_pick["Player"] + " — " + df_pick[team_col] + " — " + df_pick[pos_col]

options = sorted(df_pick["label"].unique().tolist())

selected_labels = st.multiselect(
    f"Players to compare (choose exactly {int(n_players)})",
    options=options,
    default=[],
)

if len(selected_labels) != int(n_players):
    st.info(f"Please select exactly {int(n_players)} players. Currently selected: {len(selected_labels)}.")
    st.stop()

# Resolve rows
selected_rows = []
for lab in selected_labels:
    parts = lab.split(" — ")
    p_name = parts[0].strip()
    p_team = parts[1].strip() if len(parts) > 1 else ""
    row = df[(df["Player"] == p_name)]
    if team_col in df.columns and p_team:
        row = row[row[team_col] == p_team]
    if not row.empty:
        selected_rows.append(row.iloc[0])

sel_df = pd.DataFrame(selected_rows)
if sel_df.empty:
    st.error("Could not resolve selected players. Adjust filters and try again.")
    st.stop()

sel_df = sel_df.copy()
sel_df["Position group"] = sel_df[pos_col].apply(map_position_group)

# Reference player determines peer group position group
ref_label = st.selectbox(
    "Reference player (peer-group position group defaults to this player)",
    options=[f"{r['Player']} ({r.get(team_col,'')})" for _, r in sel_df.iterrows()],
    index=0,
)
ref_player = ref_label.split(" (")[0]
ref_group = sel_df[sel_df["Player"] == ref_player].iloc[0]["Position group"]

peer_mode = st.radio(
    "Peer group for percentiles",
    options=[f"Same position group as reference ({ref_group})", "All positions"],
    index=0,
    horizontal=True,
)

# Peer pool defined by current filters (df_f) + optional position group restriction
peer_pool = df_f.copy()
peer_pool["Position group"] = peer_pool[pos_col].apply(map_position_group)
if peer_mode.startswith("Same position group"):
    peer_pool = peer_pool[peer_pool["Position group"] == ref_group]

st.caption(f"Peer pool size: {len(peer_pool):,}")

st.divider()

# ----------------------------
# Profiles
# ----------------------------
st.subheader("Profiles")

profile_fields = [
    team_col,
    pos_col,
    "Age",
    "Minutes played",
    "Matches played",
    "Market value",
]
profile_fields = [c for c in profile_fields if c in sel_df.columns]

cols = st.columns(len(sel_df))
for i, c in enumerate(cols):
    with c:
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
# Metrics configuration
# ----------------------------
st.subheader("Scouting metrics")

# A conservative default set (works for most outfield players)
candidate_totals = [
    "Goals", "Assists", "xG", "xA", "Shots", "Key passes",
    "Successful dribbles", "Progressive passes", "Progressive runs",
    "Interceptions", "Tackles", "Duels",
]
available_totals = [c for c in candidate_totals if c in df.columns]

if not available_totals:
    st.info("No common metrics found in this dataset. We can map your exact Wyscout columns next.")
    st.stop()

selected_totals = st.multiselect(
    "Select total metrics (we will convert to per 90 automatically)",
    options=available_totals,
    default=available_totals[:8],
)

if not selected_totals:
    st.info("Select at least one metric.")
    st.stop()

# ----------------------------
# Build per90 + percentile tables
# ----------------------------
per90_tbl = pd.DataFrame(index=[f"{m} per 90" for m in selected_totals])
pct_tbl = pd.DataFrame(index=[f"{m} per 90" for m in selected_totals])

for _, r in sel_df.iterrows():
    col_name = f"{r['Player']} ({r.get(team_col,'')})"
    mins = r.get("Minutes played", np.nan)

    per90_vals = []
    pct_vals = []

    for m in selected_totals:
        v = per90(pd.Series([r.get(m, np.nan)]), pd.Series([mins])).iloc[0]
        pool = per90(peer_pool[m], peer_pool["Minutes played"])
        p = percentile_rank(v, pool)
        per90_vals.append(v)
        pct_vals.append(p)

    per90_tbl[col_name] = per90_vals
    pct_tbl[col_name] = pct_vals

per90_tbl = per90_tbl.round(3)
pct_tbl = pct_tbl.round(1)

st.subheader("Per-90 values")
st.dataframe(per90_tbl, use_container_width=True)

st.subheader("Percentiles (0–100)")
# Color styling (heatmap-style) driven by selected color scale
styled = pct_tbl.style.background_gradient(axis=None, cmap=color_scale, vmin=0, vmax=100)
st.dataframe(styled, use_container_width=True)

st.divider()

# ----------------------------
# Radar chart (percentiles)
# ----------------------------
st.subheader("Radar (percentiles)")

metrics = pct_tbl.index.tolist()
if len(metrics) < 3:
    st.info("Select at least 3 metrics to display a radar chart.")
    st.stop()

fig = go.Figure()
for col in pct_tbl.columns:
    vals = pct_tbl[col].tolist()
    fig.add_trace(
        go.Scatterpolar(
            r=vals + [vals[0]],
            theta=metrics + [metrics[0]],
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
