import streamlit as st
import pandas as pd

def _unique_sorted(series: pd.Series) -> list[str]:
    s = series.dropna().astype(str).str.strip()
    s = s[(s != "") & (s.str.lower() != "nan")]
    return sorted(s.unique().tolist())

def sidebar_controls(df: pd.DataFrame):
    """
    Cascading filters:
      Season -> League -> Team -> Position -> Minutes -> Name
    Team options are restricted to the selected League (and Season).
    """
    with st.sidebar:
        st.header("Player Filters")

        # ----- Season (top-level) -----
        seasons = ["All"]
        if "Season" in df.columns:
            seasons += _unique_sorted(df["Season"])
        season = st.selectbox("Season", seasons)

        # filter df by season for dependent dropdowns
        df1 = df
        if season != "All" and "Season" in df1.columns:
            df1 = df1[df1["Season"].astype(str) == str(season)]

        # ----- League / Competition (depends on season) -----
        league_col = "Competition" if "Competition" in df1.columns else ("League" if "League" in df1.columns else None)
        leagues = ["All"]
        if league_col:
            leagues += _unique_sorted(df1[league_col])
        competition = st.selectbox("League", leagues)

        # filter df by league for dependent dropdowns
        df2 = df1
        if competition != "All" and league_col:
            df2 = df2[df2[league_col].astype(str) == str(competition)]

        # ----- Team (depends on season + league) -----
        teams = ["All"]
        if "Team" in df2.columns:
            teams += _unique_sorted(df2["Team"])
        team = st.selectbox("Team", teams)

        # filter df by team for dependent dropdowns (optional but improves Position list)
        df3 = df2
        if team != "All" and "Team" in df3.columns:
            df3 = df3[df3["Team"].astype(str) == str(team)]

        # ----- Position (depends on season + league + team) -----
        positions = ["All"]
        if "Position" in df3.columns:
            positions += _unique_sorted(df3["Position"])
        position = st.selectbox("Position", positions)

        # ----- Minutes -----
        max_minutes = 0
        if "Minutes played" in df.columns:
            max_minutes = int(pd.to_numeric(df["Minutes played"], errors="coerce").fillna(0).max())
        minutes_min = st.slider("Minimum minutes", 0, max_minutes, 600, 30)

        # ----- Name search -----
        name_query = st.text_input("Search player name", "")

    # return signature must match your pages
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
