import streamlit as st
import pandas as pd
from src.data import load_csv

st.title("Home")

st.markdown(
    """
    ### Scouting App
    Upload a Wyscout league export to begin scouting individual players.
    """
)

uploaded_file = st.file_uploader(
    "Upload Wyscout CSV",
    type=["csv"],
    accept_multiple_files=False,
)

if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)
df = load_csv(uploaded_file)

st.session_state["data"] = df

st.success(f"Data loaded successfully — {len(df):,} rows")
