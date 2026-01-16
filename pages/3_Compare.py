import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.filters import apply_filters
from src.ui import sidebar_controls

st.set_page_config(page_title="Compare", layout="wide")
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

# Normalize key identifiers
for col in ["Player", "Team", "Position"]:
    df[col] = df[col].astype(str).str.strip()

# ----------------------------
# Sidebar filters (reuse Players filters)
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
    st.info("No players match the current filters. Adjust Season/League/Minutes/Team/Position.")
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
    # strict percentile; stable with ties
    return round((pop < float(value)).mean() * 100.0, 1)


def first_existing_column(candidates: list[str], columns: pd.Index) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


# Metric templates (aliases handle different export naming)
TEMPLATES = {
    "Attacker (FWD)": [
        ("Goals", ["Goals", "Gls"]),
        ("xG", ["xG", "Expected goals"]),
        ("Shots", ["Shots", "Shots total"]),
        ("Assists", ["Assists", "Ast"]),
        ("xA", ["xA", "Expected assists"]),
        ("Key passes", ["Key passes", "Key passes total"]),
        ("Successful dribbles", ["Successful dribbles", "Dribbles successful"]),
        ("Touches in box", ["Touches in box", "Touches in penalty area"]),
        ("Progressive runs", ["Progressive runs", "Progressive carries", "Progressive runs total"]),
        ("Progressive passes", ["Progressive passes", "Progressive passes total"]),
    ],
    "Midfielder (MID)": [
        ("Assists", ["Assists", "Ast"]),
        ("xA", ["xA", "Expected assists"]),
        ("Key passes", ["Key passes", "Key passes total"]),
        ("Passes", ["Passes", "Passes total", "Passes attempted"]),
        ("Accurate passes %", ["Accurate passes, %", "Pass accuracy %", "Pass accuracy, %"]),
        ("Progressive passes", ["Progressive passes", "Progressive passes total"]),
        ("Duels", ["Duels", "Duels total"]),
        ("Duels won %", ["Duels won, %", "Duels won %"]),
        ("Interceptions", ["Interceptions", "Interceptions total"]),
        ("Tackles", ["Tackles", "Tackles total"]),
    ],
    "Defender (DEF)": [
        ("Duels", ["Duels", "Duels total"]),
        ("Duels won %", ["Duels won, %", "Duels won %"]),
        ("Aerial duels", ["Aerial duels", "Aerial duels total"]),
        ("Aerial won %", ["Aerial duels won, %", "Aerial duels won %"]),
        ("Interceptions", ["Interceptions", "Interceptions total"]),
        ("Tackles", ["Tackles", "Tackles total"]),
        ("Clearances", ["Clearances", "Clearances total"]),
        ("Blocked shots", ["Blocked shots", "Shots blocked"]),
        ("Progressive passes", ["Progressive passes", "Progressive passes total"]),
        ("Accurate passes %", ["Accurate passes, %", "Pass accuracy %", "Pass accuracy, %"]),
    ],
    "Goalkeeper (GK)": [
        ("Saves", ["Saves", "Saves total"]),
        ("Save %", ["Save rate, %", "Save %", "Save percentage"]),
        ("Goals conceded", ["Goals conceded", "Goals against"]),
        ("Clean sheets", ["Clean sheets"]),
        ("Accurate passes %", ["Accurate passes, %", "Pass accuracy %", "Pass accuracy, %"]),
        ("Long passes", ["Long passes", "Long passes total"]),
        ("Long passes accurate %", ["Accurate long passes, %", "Long pass accuracy %"]),
    ],
}

# ----------------------------
# Player selection
# ----------------------------
st.subheader("Selection")
st.caption("Select 2–6 players. Use sidebar filters to narrow the pool.")

df_pick = df_f[["Player", "Team", "Position"]].dropna().copy()
df_pick["Position group"] = df_pick["Position"].apply(map_position_group)
df_pick["label"] = df_pick["Player"] + " — " + df_pick["Team"] + " — " + df_pick["Position"]
label_options = sorted(df_pick["label"].unique().tolist())

selected_labels = st.multiselect("Players to compare", label_options, default=[])

if len(selected_labels) < 2:
    st.info("Select at least 2 players.")
    st.stop()

if len(selected_labels) > 6:
    st.warning("Please select at most 6 players for readability. Using the first 6 selections.")
    selected_labels = selected_labels[:6]

# Resolve selected rows
selected_rows = []
for lab in selected_labels:
    p, t, *_ = lab.split(" — ")
    row = df[(df["Player"] == p.strip()) & (df["Team"] == t.strip())]
    if not row.empty:
        selected_rows.append(row.iloc[0])

sel_df = pd.DataFrame(selected_rows)
if sel_df.empty:
    st.error("Could not resolve selected players. Adjust filters and try again.")
    st.stop()

sel_df = sel_df.copy()
sel_df["Position group"] = sel_df["Position"].apply(map_position_group)

ref_label = st.selectbox(
    "Reference player (peer group position group defaults to this player)",
    options=[f"{r['Player']} ({r['Team']})" for _, r in sel_df.iterrows()],
    index=0,
)
ref_player_name = ref_label.split(" (")[0]
ref_group = sel_df[sel_df["Player"] == ref_player_name].iloc[0]["Position group"]

peer_group_mode = st.radio(
    "Peer group definition for percentiles",
    options=[
        f"Same position group as reference ({ref_group})",
        "All positions (no grouping)",
    ],
    index=0,
    horizontal=False,
)

# ----------------------------
# Layout controls
# ----------------------------
left, right = st.columns([1, 1])

with left:
    template_choice = st.selectbox(
        "Metric template",
        options=list(TEMPLATES.keys()),
        index=0,
        help="Templates are role-based. We will automatically skip metrics missing from your dataset.",
    )

with right:
    max_metrics = st.slider("Max metrics shown (radar readability)", 6, 12, 10, 1)

# Build metrics list from template, resolving actual columns
template = TEMPLATES[template_choice]
resolved = []
for label, aliases in template:
    col = first_existing_column(aliases, df.columns)
    if col:
        resolved.append((label, col))

if not resolved:
    st.warning("None of the template metrics exist in your dataset. We can map your column names next.")
    st.dataframe(df.head(20), use_container_width=True)
    st.stop()

resolved = resolved[:max_metrics]

# ----------------------------
# Profiles row
# ----------------------------
st.subheader("Profiles")

card_fields = ["Team", "Position", "Position group", "Age", "Minutes played", "Matches played", "Market value"]
card_fields = [c for c in card_fields if c in sel_df.columns]

cols = st.columns(len(sel_df))
for i, c in enumerate(cols):
    with c:
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
# Peer pool for percentiles
# ----------------------------
peer_pool = df_f.copy()
peer_pool["Position group"] = peer_pool["Position"].apply(map_position_group)

if peer_group_mode.startswith("Same position group"):
    peer_pool = peer_pool[peer_pool["Position group"] == ref_group]

st.caption(f"Peer pool size: {len(peer_pool):,}")

# ----------------------------
# Build per-90 value table and percentile table
# ----------------------------
minutes_col = "Minutes played"
metric_labels = [m[0] for m in resolved]
metric_cols = [m[1] for m in resolved]

# Per-90 table
per90_tbl = pd.DataFrame(index=metric_labels)
pct_tbl = pd.DataFrame(index=metric_labels)

for _, r in sel_df.iterrows():
    pname = f"{r['Player']} ({r['Team']})"
    mins = r.get(minutes_col, np.nan)

    per90_vals = []
    pct_vals = []

    for label, col in resolved:
        # If column looks like a percentage column, treat as raw
        is_pct = "%" in label or str(col).strip().endswith("%") or "% " in str(col) or ", %" in str(col)

        if is_pct:
            val = to_num(pd.Series([r.get(col, np.nan)])).iloc[0]
            pool = peer_pool[col] if col in peer_pool.columns else pd.Series(dtype=float)
            pct = percentile_rank(val, pool)
            per90_vals.append(val)
            pct_vals.append(pct)
        else:
            # per90 computed from totals + minutes (even if already per90 in export, totals work reliably)
            val = per90(pd.Series([r.get(col, np.nan)]), pd.Series([mins])).iloc[0]
            pool = per90(peer_pool[col], peer_pool[minutes_col]) if col in peer_pool.columns else pd.Series(dtype=float)
            pct = percentile_rank(val, pool)
            per90_vals.append(val)
            pct_vals.append(pct)

    per90_tbl[pname] = per90_vals
    pct_tbl[pname] = pct_vals

# Formatting
per90_tbl = per90_tbl.apply(pd.to_numeric, errors="ignore").round(3)
pct_tbl = pct_tbl.apply(pd.to_numeric, errors="ignore").round(1)

# ----------------------------
# Display tables
# ----------------------------
st.subheader("Per-90 values")
st.caption("Totals are normalized by minutes. Percentage metrics are shown as raw values.")
st.dataframe(per90_tbl, use_container_width=True)

st.subheader("Percentiles vs peer pool")
st.caption("0–100 scale. Higher is better for these metrics (we can invert selected metrics later if needed).")
st.dataframe(pct_tbl, use_container_width=True)

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
    st.info("Not enough metrics for radar (need at least 3). Increase Max metrics or choose another template.")
    st.stop()

fig = go.Figure()

for col in pct_tbl.columns:
    values = pct_tbl[col].tolist()
    # close the loop
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
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
