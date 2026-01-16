import streamlit as st
import pandas as pd

st.set_page_config(page_title="Scouting App", layout="wide")

st.title("Scouting App")
st.caption("Upload a Wyscout CSV to begin. Then open Players to scout individuals.")

uploaded_file = st.file_uploader(
    "Upload Wyscout League Export (CSV)",
    type=["csv"],
    accept_multiple_files=False,
    key="wyscout_uploader",
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Basic cleaning for key identifiers
    for col in ["Player", "Team", "Position"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    st.session_state["data"] = df
    st.success(f"Loaded {len(df):,} rows and {len(df.columns):,} columns.")
else:
    st.info("No file uploaded yet.")

# Optional: navigation buttons (sidebar pages already exist)
st.subheader("Navigate")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Players"):
        st.switch_page("pages/2_Players.py")

with col2:
    if st.button("Compare"):
        st.switch_page("pages/3_Compare.py")

with col3:
    if st.button("Shortlists"):
        st.switch_page("pages/4_Shortlists.py")
