import streamlit as st
import pandas as pd

def sidebar_controls(df: pd.DataFrame):
    with st.sidebar:
        st.header("Player Filters")

        max_minutes = 0
        if "Minutes played" in df.columns:
            max_minutes = int(df["Minutes played"].fillna(0).max())

        minutes_min = st.slider("Minimum minutes", 0, max_minutes, 600, 30)

        teams = ["All"]
        if "Team" in df.columns:
            teams += sorted(df["Team"].dropna().unique().tolist())
        team = st.selectbox("Team", teams)

        positions = ["All"]
        if "Position" in df.columns:
            positions += sorted(df["Position"].dropna().unique().tolist())
        position = st.selectbox("Position", positions)

        name_query = st.text_input("Search player name", "")

    return minutes_min, team, position, name_query


def player_header(player_row: pd.Series):
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Team", str(player_row.get("Team", "")))
    c2.metric("Position", str(player_row.get("Position", "")))

    age = player_row.get("Age", None)
    c3.metric("Age", "" if pd.isna(age) else str(int(age)))

    mins = player_row.get("Minutes played", 0)
    c4.metric("Minutes", "" if pd.isna(mins) else str(int(mins)))

    mv = player_row.get("Market value", None)
    if mv is None or pd.isna(mv):
        c5.metric("Market value", "")
    else:
        c5.metric("Market value", f"{int(mv):,}")
