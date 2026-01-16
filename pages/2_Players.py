import streamlit as st
import pandas as pd

from src.filters import apply_filters
from src.ui import sidebar_controls, player_header

st.title("Players")

# ----------------------------
# Load data (from main page uploader)
# ----------------------------
if "data" not in st.session_state:
    st.warning("No data loaded yet. Go to the main page (app) and upload a CSV.")
    st.stop()

df = st.session_state["data"]
if not isinstance(df, pd.DataFrame) or df.empty:
    st.error("Loaded data is empty or invalid. Please re-upload the CSV on the main page.")
    st.stop()

required = ["Player", "Team", "Position", "Minutes played"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Your CSV is missing required columns: {missing}")
    st.stop()

# Ensure consistent string formatting for selection/filtering
for col in ["Player", "Team", "Position"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# ----------------------------
# Sidebar filters
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

st.subheader("Scouting list")
st.caption("Filter, sort, and select a player to open their individual profile.")

# Limit display size for performance
top_n = st.slider("Max players to display in table", 50, 2000, 300, 50)

# ----------------------------
# Scouting list table
# ----------------------------
display_cols = ["Player", "Team", "Position", "Age", "Minutes played", "Matches played", "Market value"]
display_cols = [c for c in display_cols if c in df_f.columns]

table = df_f[display_cols].copy()
if "Minutes played" in table.columns:
    table["Minutes played"] = pd.to_numeric(table["Minutes played"], errors="coerce")
    table = table.sort_values("Minutes played", ascending=False)

st.dataframe(table.head(top_n), use_container_width=True, height=380)

# ----------------------------
# Player selector (label includes team + position)
# ----------------------------
pick_cols = ["Player", "Team", "Position"]
df_pick = df_f[pick_cols].dropna().copy()
df_pick["label"] = df_pick["Player"] + " — " + df_pick["Team"] + " — " + df_pick["Position"]

labels = [""] + sorted(df_pick["label"].unique().tolist())
selected_label = st.selectbox("Select a player", labels)

if not selected_label:
    st.info("Select a player to view their profile.")
    st.stop()

selected_player = selected_label.split(" — ")[0]

# If names can repeat across teams, choose the matching row by label instead of name only
selected_team = selected_label.split(" — ")[1] if " — " in selected_label else None

player_matches = df[(df["Player"] == selected_player)]
if selected_team and "Team" in df.columns:
    player_matches = player_matches[player_matches["Team"] == selected_team]

if player_matches.empty:
    st.error("Could not find the selected player in the dataset. Please adjust filters and try again.")
    st.stop()

player_row = player_matches.iloc[0]  # pandas Series

st.subheader(selected_player)
player_header(player_row)

# ----------------------------
# Tabs (layout)
# ----------------------------
tabs = st.tabs(["Overview (layout)", "Profile", "Raw"])

with tabs[0]:
    st.info("Next: add overview widgets (percentiles, key metrics, role templates).")

with tabs[1]:
    # FIX: player_row is a Series; Series has .to_frame(), DataFrame does not.
    st.dataframe(player_row.to_frame("Value"), use_container_width=True)

with tabs[2]:
    st.dataframe(pd.DataFrame(player_row).T, use_container_width=True)
