import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    """
    Generic CSV loader with basic cleaning.
    """
    df = pd.read_csv(path)

    # Strip whitespace from key identifiers
    for col in ["Player", "Team", "Position"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df
