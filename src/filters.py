import pandas as pd

def apply_filters(
    df: pd.DataFrame,
    season: str = "All",
    competition: str = "All",
    minutes_min: int = 0,
    team: str = "All",
    position: str = "All",
    name_query: str = "",
) -> pd.DataFrame:
    out = df.copy()

    # Season
    if season != "All" and "Season" in out.columns:
        out = out[out["Season"].astype(str) == str(season)]

    # League/Competition (prefer parsed Competition; fallback to League)
    if competition != "All":
        if "Competition" in out.columns:
            out = out[out["Competition"].astype(str) == str(competition)]
        elif "League" in out.columns:
            out = out[out["League"].astype(str) == str(competition)]

    # Minutes
    if "Minutes played" in out.columns:
        mins = pd.to_numeric(out["Minutes played"], errors="coerce").fillna(0)
        out = out[mins >= minutes_min]

    # Team (use Team within selected timeframe if present)
    team_col = "Team within selected timeframe" if "Team within selected timeframe" in out.columns else "Team"
    if team != "All" and team_col in out.columns:
        out = out[out[team_col].astype(str) == str(team)]

    # Position
    if position != "All" and "Position" in out.columns:
        out = out[out["Position"].astype(str) == str(position)]

    # Name query
    if name_query.strip() and "Player" in out.columns:
        q = name_query.strip().lower()
        out = out[out["Player"].astype(str).str.lower().str.contains(q, na=False)]

    return out
