import pandas as pd

def apply_filters(
    df: pd.DataFrame,
    season: str = "All",
    minutes_min: int = 0,
    team: str = "All",
    position: str = "All",
    name_query: str = "",
) -> pd.DataFrame:
    out = df.copy()

    # Season
    if season != "All" and "Season" in out.columns:
        out = out[out["Season"].astype(str) == str(season)]

    # Minutes
    if "Minutes played" in out.columns:
        mins = pd.to_numeric(out["Minutes played"], errors="coerce").fillna(0)
        out = out[mins >= minutes_min]

    # Team
    if team != "All" and "Team" in out.columns:
        out = out[out["Team"] == team]

    # Position
    if position != "All" and "Position" in out.columns:
        out = out[out["Position"] == position]

    # Name query
    if name_query.strip() and "Player" in out.columns:
        q = name_query.strip().lower()
        out = out[out["Player"].astype(str).str.lower().str.contains(q, na=False)]

    return out
