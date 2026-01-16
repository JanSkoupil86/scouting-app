import re
import numpy as np
import pandas as pd
import streamlit as st

from src.filters import apply_filters
from src.ui import sidebar_controls

st.title("Compare")

# ----------------------------
# Load data (from app uploader)
# ----------------------------
if "data" not in st.session_state:
    st.warning("No data loaded yet. Go to the **app** page and upload a CSV.")
    st.stop()

df = st.session_state["data"]
if not isinstance(df, pd.DataFrame) or df.empty:
    st.error("Loaded dataset is empty or invalid. Re-upload the CSV on the **app** page.")
    st.stop()

required = ["Player", "Team", "Position", "Minutes played"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Your CSV is missing required columns: {missing}")
    st.stop()

# Basic cleanup
for col in ["Player", "Team", "Position"]:
    df[col] = df[col].astype(str).str.strip()

# ----------------------------
# Sidebar filters (Season first)
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

st.subheader("Select players")
st.caption("Use the sidebar filters to narrow the pool, then select 2–5 players to compare.")

if df_f.empty:
    st.info("No players match the current filters. Adjust Season/League/Minutes/Team/Position.")
    st.stop()

# Build labels to reduce ambiguity
df_pick = df_f[["Player", "Team", "Position"]].dropna().copy()
df_pick["label"] = df_pick["Player"] + " — " + df_pick["Team"] + " — " + df_pick["Position"]
label_options = sorted(df_pick["label"].unique().tolist())

selected_labels = st.multiselect("Players to compare (2–5)", label_options, default=[])

if len(selected_labels) < 2:
    st.info("Select at least 2 players to compare.")
    st.stop()

if len(selected_labels) > 5:
    st.warning("Please select at most 5 players for readability. Using the first 5 selections.")
    selected_labels = selected_labels[:5]

# Resolve selected rows (prefer exact match by Player + Team)
selected_rows = []
for lab in selected_labels:
    parts = lab.split(" — ")
    p_name = parts[0].strip()
    p_team = parts[1].strip() if len(parts) > 1 else ""
    row = df[(df["Player"] == p_name) & (df["Team"] == p_team)]
    if not row.empty:
        selected_rows.append(row.iloc[0])

if not selected_rows:
    st.error("Could not resolve selected players. Adjust filters and try again.")
    st.stop()

sel_df = pd.DataFrame(selected_rows)

# ----------------------------
# Helpers (position groups, per90, percentiles)
# ----------------------------
def map_position_group(position_str: str) -> str:
    if not position_str:
        return "OTHER"
    pos = str(position_str).upper()

    if "GK" in pos:
        return "GK"

    # DEF
    if any(p in pos for p in ["CB", "LCB", "RCB", "LB", "RB", "LWB", "RWB"]):
        return "DEF"

    # MID
    if any(p in pos for p in ["DMF", "CMF", "AMF", "LMF", "RMF"]):
        return "MID"

    # FWD/Wide
    if any(p in pos for p in ["CF", "SS", "ST", "LWF", "RWF", "LW", "RW"]):
        return "FWD"

    return "OTHER"


def per90(values: pd.Series, minutes: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    mins = pd.to_numeric(minutes, errors="coerce").replace(0, np.nan)
    return (vals / mins) * 90


def percentile_rank(value, population: pd.Series):
    pop = pd.to_numeric(population, errors="coerce").dropna()
    if pop.empty or pd.isna(value):
        return None
    # strict-less-than percentile (robust for ties)
    return round((pop < float(value)).mean() * 100, 1)


# Add position groups to selected and peer pool
sel_df = sel_df.copy()
sel_df["Position group"] = sel_df["Position"].apply(map_position_group)

# ----------------------------
# Profiles
# ----------------------------
st.subheader("Profiles")

card_fields = ["Team", "Position", "Position group", "Age", "Minutes played", "Matches played", "Market value"]
card_fields = [c for c in card_fields if c in sel_df.columns]

cols = st.columns(len(sel_df))
for i, col in enumerate(cols):
    with col:
        st.markdown(f"### {sel_df.iloc[i]['Player']}")
        for f in card_fields:
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
# Raw metrics comparison (as-is)
# ----------------------------
st.subheader("Raw metrics comparison")

candidate_metrics = [
    "Goals", "Assists", "xG", "xA",
    "Shots", "Key passes",
    "Shots per 90", "Key passes per 90",
    "Passes per 90", "Accurate passes, %",
    "Duels per 90", "Duels won, %",
    "Successful dribbles per 90", "Successful dribbles, %",
    "Progressive passes per 90", "Progressive runs per 90",
]
available_metrics = [m for m in candidate_metrics if m in sel_df.columns]

if available_metrics:
    default_metrics = available_metrics[:10]
    raw_metrics = st.multiselect("Select raw metrics", options=available_metrics, default=default_metrics, key="raw_metrics")
    if raw_metrics:
        raw_out = pd.DataFrame(index=raw_metrics)
        for _, r in sel_df.iterrows():
            col_name = f"{r['Player']} ({r['Team']})"
            raw_out[col_name] = [r.get(m, None) for m in raw_metrics]
        st.dataframe(raw_out, use_container_width=True)
else:
    st.info("No standard raw metrics found in this dataset for the comparison table.")

st.divider()

# ----------------------------
# Scouting comparison (per-90 + percentiles)
# ----------------------------
st.subheader("Scouting comparison (percentiles)")
st.caption("Percentiles are computed vs a peer pool defined by the current sidebar filters.")

# Choose peer pool behavior
mode = st.radio(
    "Peer group definition",
    options=["Same position group as first selected player", "Use all positions (no grouping)"],
    index=0,
    horizontal=False
)

ref_group = sel_df.iloc[0]["Position group"]
peer_pool = df_f.copy()
peer_pool["Position group"] = peer_pool["Position"].apply(map_position_group)

if mode == "Same position group as first selected player":
    peer_pool = peer_pool[peer_pool["Position group"] == ref_group]

st.caption(f"Peer pool size: {len(peer_pool):,} | Reference position group: {ref_group}")

# Define scouting metrics with per90 fallback where appropriate.
# If your dataset already has per90 columns, we use them; otherwise we compute from totals.
scouting_metrics = [
    # label, type, total_column, per90_column(optional)
    ("Goals per 90", "per90", "Goals", "Goals per 90"),
    ("Assists per 90", "per90", "Assists", "Assists per 90"),
    ("xG per 90", "per90", "xG", "xG per 90"),
    ("xA per 90", "per90", "xA", "xA per 90"),
    ("Shots per 90", "per90", "Shots", "Shots per 90"),
    ("Key passes per 90", "per90", "Key passes", "Key passes per 90"),
    ("Duels won, %", "raw", "Duels won, %", None),
    ("Accurate passes, %", "raw", "Accurate passes, %", None),
    ("Successful dribbles per 90", "per90", "Successful dribbles", "Successful dribbles per 90"),
    ("Progressive passes per 90", "per90", "Progressive passes", "Progressive passes per 90"),
    ("Progressive runs per 90", "per90", "Progressive runs", "Progressive runs per 90"),
]

# Keep only metrics we can compute (total exists OR per90 exists)
usable = []
for label, mtype, total_col, per90_col in scouting_metrics:
    if mtype == "raw":
        if total_col in df.columns:
            usable.append((label, mtype, total_col, per90_col))
    else:
        if (per90_col and per90_col in df.columns) or (total_col in df.columns):
            usable.append((label, mtype, total_col, per90_col))

if not usable:
    st.info("No scouting metrics available for per-90 / percentile comparison in this dataset.")
    st.stop()

metric_labels = [u[0] for u in usable]
default_labels = metric_labels[:8]

selected_metric_labels = st.multiselect(
    "Select scouting metrics (percentiles)",
    options=metric_labels,
    default=default_labels,
    key="scouting_metrics"
)

if not selected_metric_labels:
    st.info("Select at least one scouting metric.")
    st.stop()

# Build output: rows = metrics, columns = players
pct_out = pd.DataFrame(index=selected_metric_labels)

for _, r in sel_df.iterrows():
    player_col = f"{r['Player']} ({r['Team']})"

    values_for_player = []
    for label in selected_metric_labels:
        (lab, mtype, total_col, per90_col) = next(x for x in usable if x[0] == label)

        if mtype == "raw":
            val = pd.to_numeric(pd.Series([r.get(total_col, np.nan)]), errors="coerce").iloc[0]
            pool = peer_pool[total_col] if total_col in peer_pool.columns else pd.Series(dtype=float)
            pct = percentile_rank(val, pool)

        else:
            # Prefer existing per90 column if present
            if per90_col and per90_col in df.columns and per90_col in peer_pool.columns:
                val = pd.to_numeric(pd.Series([r.get(per90_col, np.nan)]), errors="coerce").iloc[0]
                pool = peer_pool[per90_col]
                pct = percentile_rank(val, pool)
            else:
                # Compute per90 from total + minutes
                if total_col not in peer_pool.columns:
                    pct = None
                else:
                    val = per90(pd.Series([r.get(total_col, np.nan)]), pd.Series([r.get("Minutes played", np.nan)])).iloc[0]
                    pool = per90(peer_pool[total_col], peer_pool["Minutes played"])
                    pct = percentile_rank(val, pool)

        values_for_player.append(pct)

    pct_out[player_col] = values_for_player

st.dataframe(pct_out, use_container_width=True)

st.caption("Percentiles: higher is better for these metrics. We can invert metrics later where lower is better (e.g., fouls, cards).")
