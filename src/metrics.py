# src/metrics.py
import pandas as pd
import numpy as np

def per90(series: pd.Series, minutes: pd.Series) -> pd.Series:
    mins = pd.to_numeric(minutes, errors="coerce").replace(0, np.nan)
    vals = pd.to_numeric(series, errors="coerce")
    return (vals / mins) * 90


def percentile_rank(value, population: pd.Series) -> float:
    pop = pd.to_numeric(population, errors="coerce").dropna()
    if pop.empty or pd.isna(value):
        return None
    return round((pop < value).mean() * 100, 1)
