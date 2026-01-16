import streamlit as st
import pandas as pd

def sidebar_controls(df: pd.DataFrame):
    with st.sidebar:
        st.header("Player Filters")

        # 1) Season FIRST (parsed into df["Season"] in app.py)
        seasons = ["All"]
        if "Season" in df.columns:
            s = df["Season"].astype(str)
            s = s[(s != "") & (s.lower() != "nan")]
            seasons += sorted(s.dropna().unique().tolist())
        season = st.selectbox("Season", seasons)

        # 2) League/Competition SECOND (prefer df["Competition"], fallback df["League"])
        leagues = ["All"]
        if "Competition" in df.columns:
            leagues += sorted(df["Competition"].dropna().astype(str).unique().tolist())
        elif "League" in df.columns:
            leagues += sorted(df["League"].dropna().astype(str).unique().tolist())
        competition = st.selectbox("League", leagues)

        # 3) Minutes
        max_minutes = 0
        if "Minutes played" in df.columns:
            max_minutes = int(pd.to_numeric(df["Minutes played"], errors="coerce").fillna(0).max())
        minutes_min = st.slider("Minimum minutes", 0, max_minutes, 600, 30)

        # 4) Team
        teams = ["All"]
        if "Team" in df.columns:
            teams += sorted(df["Team"].dropna().astype(str).unique().tolist())
        team = st.selectbox("Team", teams)

        # 5) Position
        positions = ["All"]
        if "Position" in df.columns:
            positions += sorted(df["Position"].dropna().astype(str).unique().tolist())
        position = st.selectbox("Position", positions)

        # 6) Name search
        name_query = st.text_input("Search player name", "")

    # IMPORTANT: return exactly 6 values
    return season, competition, minutes_min, team, position, name_query


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
