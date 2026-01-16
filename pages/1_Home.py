import streamlit as st
import pandas as pd

st.title("Home")

st.markdown(
    """
    ### Scouting App
    This app is designed for **individual player scouting** using Wyscout league exports.

    **Workflow**
    1. Open **app** and upload a CSV (only once)
    2. Go to **Players** to filter and select individuals
    3. Add scouting views (percentiles, role templates, comparisons)
    """
)

st.subheader("Data status")

if "data" not in st.session_state:
    st.warning("No dataset loaded yet. Go to the **app** page and upload a CSV.")
    st.stop()

df = st.session_state["data"]
st.success(f"Dataset loaded — {len(df):,} rows × {len(df.columns):,} columns")

# Show a small preview
st.caption("Preview (first 20 rows)")
st.dataframe(df.head(20), use_container_width=True)
