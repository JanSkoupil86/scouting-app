import streamlit as st
import pandas as pd

from src.data import load_csv
from src.filters import apply_filters
from src.ui import sidebar_controls, player_header

st.title("Players")

CSV_PATH = "data/Wyscout_League_Export.csv"

# Load data
try:
    df = load_csv(CSV_PATH)
except FileNotFoundError:
    st.error("Missing data file. Upload `Wyscout_League_Export.csv` into the `data/` folder.")
    st.stop()

# Sidebar filters
minutes_min, team, position, name_query = sidebar_controls(df)

# Apply filters
df_f = apply_filters(df, minutes_min=minutes_min, team=team, position=position, name_query=name_query)

st.subheader("Player list")

# Show a compact table for selection
cols = [c for c in ["Player", "Team", "Position", "Age", "Minutes played", "Matches played", "Market value"] if c in df_f.columns]
table = df_f[cols].copy()
if "Minutes played" in table.columns:
    table = table.sort_values("Minutes played", ascending=False)

st.dataframe(table, use_container_width=True, height=360)

players = sorted(df_f["Player"].dropna().unique().tolist()) if "Player" in df_f.columns else []
selected = st.selectbox("Select a player", [""] + players)

if not selected:
    st.info("Select a player to view their profile header.")
    st.stop()

player_row = df[df["Player"] == selected].iloc[0]
st.subheader(selected)
player_header(player_row)

tabs = st.tabs(["Overview (layout)", "Profile", "Raw"])

with tabs[0]:
    st.info("Next: add the scouting overview widgets (percentiles, key metrics, role templates).")

with tabs[1]:
    st.write(pd.DataFrame(player_row).to_frame("Value"))

with tabs[2]:
    st.dataframe(pd.DataFrame(player_row).T, use_container_width=True)

