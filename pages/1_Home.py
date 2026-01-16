import streamlit as st
from src.data import load_csv

st.title("Home")

CSV_PATH = "data/Wyscout_League_Export.csv"

st.markdown(
    """
    ### Scouting App
    This application is designed for **individual player scouting** using Wyscout data exports.

    **Workflow**
    1. Load a league export
    2. Filter players
    3. Analyse individual profiles
    """
)

st.subheader("Data status")

try:
    df = load_csv(CSV_PATH)
    st.success(f"Data loaded successfully — {len(df):,} rows")
except FileNotFoundError:
    st.warning(
        "No data file found yet.\n\n"
        "Please upload `Wyscout_League_Export.csv` into the `data/` folder."
    )
