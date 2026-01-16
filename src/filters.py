import pandas as pd

def apply_filters(
    df: pd.DataFrame,
    minutes_min: int,
    team: str = "All",
    position: str = "All",
    name_query: str = "",
) -> pd.DataFrame:
    out = df.copy()

    if "Minutes played" in out.columns:
        out = out[out["Minutes played"].fillna(0) >= minutes_min]

    if team != "All" and "Team" in out.columns:
        out = out[out["Team"] == team]

    if position != "All" and "Position" in out.columns:
        out = out[out["Position"] == position]

    if name_query.strip() and "Player" in out.columns:
        q = name_query.strip().lower()
        out = out[out["Player"].astype(str).str.lower().str.contains(q, na=False)]

    return out
