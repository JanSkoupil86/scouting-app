import streamlit as st
import pandas as pd

from src.filters import apply_filters
from src.ui import sidebar_controls, player_header

st.title("Players")

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

st.subheader("Scouting list")
st.caption("Filter, sort, and select a player to open their individual profile.")

top_n = st.slider("Max players to display in table", 50, 2000, 300, 50)

display_cols = ["Player", "Team", "Position", "Season", "Competition", "Age", "Minutes played", "Matches played", "Market value"]
display_cols = [c for c in display_cols if c in df_f.columns]

table = df_f[display_cols].copy()
table["Minutes played"] = pd.to_numeric(table["Minutes played"], errors="coerce")
table = table.sort_values("Minutes played", ascending=False)

st.dataframe(table.head(top_n), use_container_width=True, height=380)

# ----------------------------
# Player selector (label includes Team and Position)
# ----------------------------
df_pick = df_f[["Player", "Team", "Position"]].dropna().copy()
df_pick["label"] = df_pick["Player"] + " — " + df_pick["Team"] + " — " + df_pick["Position"]

labels = [""] + sorted(df_pick["label"].unique().tolist())
selected_label = st.selectbox("Select a player", labels)

if not selected_label:
    st.info("Select a player to view their profile.")
    st.stop()

selected_player = selected_label.split(" — ")[0]
selected_team = selected_label.split(" — ")[1]

player_matches = df[(df["Player"] == selected_player) & (df["Team"] == selected_team)]
if player_matches.empty:
    st.error("Could not find the selected player. Adjust filters and try again.")
    st.stop()

player_row = player_matches.iloc[0]  # Series

st.subheader(selected_player)
player_header(player_row)

tabs = st.tabs(["Overview (layout)", "Profile", "Raw"])

with tabs[0]:
    st.info("Next: add overview widgets (percentiles, key metrics, role templates).")

with tabs[1]:
    st.dataframe(player_row.to_frame("Value"), use_container_width=True)

with tabs[2]:
    st.dataframe(pd.DataFrame(player_row).T, use_container_width=True)
