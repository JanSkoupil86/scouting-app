import streamlit as st
import pandas as pd

def _unique_sorted(series: pd.Series) -> list[str]:
    s = series.dropna().astype(str).str.strip()
    s = s[(s != "") & (s.str.lower() != "nan")]
    return sorted(s.unique().tolist())

def sidebar_controls(df: pd.DataFrame):
    """
    Cascading filters:
      Season -> League -> Team (within selected timeframe) -> Main Position -> Minutes -> Name
    """
    with st.sidebar:
        st.header("Player Filters")

        # ----- Season -----
        seasons = ["All"]
        if "Season" in df.columns:
            seasons += _unique_sorted(df["Season"])
        season = st.selectbox("Season", seasons)

        df1 = df
        if season != "All" and "Season" in df1.columns:
            df1 = df1[df1["Season"].astype(str) == str(season)]

        # ----- League / Competition -----
        league_col = "Competition" if "Competition" in df1.columns else ("League" if "League" in df1.columns else None)
        leagues = ["All"]
        if league_col:
            leagues += _unique_sorted(df1[league_col])
        competition = st.selectbox("League", leagues)

        df2 = df1
        if competition != "All" and league_col:
            df2 = df2[df2[league_col].astype(str) == str(competition)]

        # ----- Team (within timeframe preferred) -----
        team_col = "Team within selected timeframe" if "Team within selected timeframe" in df2.columns else "Team"
        teams = ["All"]
        if team_col in df2.columns:
            teams += _unique_sorted(df2[team_col])
        team = st.selectbox("Team", teams)

        df3 = df2
        if team != "All" and team_col in df3.columns:
            df3 = df3[df3[team_col].astype(str) == str(team)]

        # ----- Position (Main Position preferred) -----
        pos_col = "Main Position" if "Main Position" in df3.columns else "Position"
        positions = ["All"]
        if pos_col in df3.columns:
            positions += _unique_sorted(df3[pos_col])
        position = st.selectbox("Position", positions)

        # ----- Minutes -----
        max_minutes = 0
        if "Minutes played" in df.columns:
            max_minutes = int(pd.to_numeric(df["Minutes played"], errors="coerce").fillna(0).max())
        minutes_min = st.slider("Minimum minutes", 0, max_minutes, 600, 30)

        # ----- Name search -----
        name_query = st.text_input("Search player name", "")

    return season, competition, minutes_min, team, position, name_query


def player_header(player_row: pd.Series):
    c1, c2, c3, c4, c5 = st.columns(5)

    team_val = player_row.get("Team within selected timeframe", player_row.get("Team", ""))
    c1.metric("Team", str(team_val))

    pos_val = player_row.get("Main Position", player_row.get("Position", ""))
    c2.metric("Position", str(pos_val))

    age = player_row.get("Age", None)
    c3.metric("Age", "" if pd.isna(age) else str(int(age)))

    mins = player_row.get("Minutes played", 0)
    c4.metric("Minutes", "" if pd.isna(mins) else str(int(mins)))

    mv = player_row.get("Market value", None)
    if mv is None or pd.isna(mv):
        c5.metric("Market value", "")
    else:
        c5.metric("Market value", f"{int(mv):,}")
