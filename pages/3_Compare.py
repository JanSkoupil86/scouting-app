import streamlit as st
import pandas as pd

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

# Ensure consistent string formatting
for col in ["Player", "Team", "Position"]:
    df[col] = df[col].astype(str).str.strip()

# ----------------------------
# Sidebar filters (reuse same controls as Players)
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

# Build selection labels to reduce ambiguity
df_pick = df_f[["Player", "Team", "Position"]].dropna().copy()
df_pick["label"] = df_pick["Player"] + " — " + df_pick["Team"] + " — " + df_pick["Position"]
label_options = sorted(df_pick["label"].unique().tolist())

selected_labels = st.multiselect("Players to compare (2–5)", label_options, default=[])

if len(selected_labels) < 2:
    st.info("Select at least 2 players to compare.")
    st.stop()

if len(selected_labels) > 5:
    st.warning("Please select at most 5 players for readability.")
    selected_labels = selected_labels[:5]

# Resolve selected rows
selected_rows = []
for lab in selected_labels:
    parts = lab.split(" — ")
    p_name = parts[0]
    p_team = parts[1] if len(parts) > 1 else ""
    # match by player+team for uniqueness
    row = df[(df["Player"] == p_name) & (df["Team"] == p_team)]
    if not row.empty:
        selected_rows.append(row.iloc[0])

if not selected_rows:
    st.error("Could not resolve selected players. Adjust filters and try again.")
    st.stop()

sel_df = pd.DataFrame(selected_rows)

# ----------------------------
# Profile cards row
# ----------------------------
st.subheader("Profiles")

card_fields = ["Team", "Position", "Age", "Minutes played", "Matches played", "Market value"]
card_fields = [c for c in card_fields if c in sel_df.columns]

cols = st.columns(len(selected_rows))
for i, col in enumerate(cols):
    with col:
        st.markdown(f"### {sel_df.iloc[i]['Player']}")
        for f in card_fields:
            v = sel_df.iloc[i].get(f, "")
            # clean numeric display
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
# Metrics comparison table (layout-first)
# ----------------------------
st.subheader("Metrics comparison")

# Offer a safe set of candidate metrics likely present in Wyscout exports.
candidate_metrics = [
    "Goals", "Assists", "xG", "xA",
    "Goals per 90", "Assists per 90", "xG per 90", "xA per 90",
    "Shots per 90", "Key passes per 90",
    "Accurate passes, %", "Passes per 90",
    "Duels per 90", "Duels won, %",
    "Successful dribbles per 90", "Successful dribbles, %",
    "Progressive passes per 90", "Progressive runs per 90",
]

available_metrics = [m for m in candidate_metrics if m in sel_df.columns]

if not available_metrics:
    st.info("No standard metrics found to compare. We can map the exact Wyscout column names next.")
    st.dataframe(sel_df.head(), use_container_width=True)
    st.stop()

default_metrics = available_metrics[:10]

metrics = st.multiselect(
    "Select metrics to compare",
    options=available_metrics,
    default=default_metrics
)

if not metrics:
    st.info("Select at least one metric.")
    st.stop()

# Build a comparison table: rows = metrics, columns = players
out = pd.DataFrame(index=metrics)

for _, r in sel_df.iterrows():
    col_name = f"{r['Player']} ({r['Team']})"
    out[col_name] = [r.get(m, None) for m in metrics]

# Convert numeric where possible for clean display
for m in metrics:
    out.loc[m] = pd.to_numeric(out.loc[m], errors="ignore")

st.dataframe(out, use_container_width=True)
