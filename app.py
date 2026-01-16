import streamlit as st
import pandas as pd

st.set_page_config(page_title="Scouting App", layout="wide")

st.title("Scouting App")
st.caption("Upload a Wyscout league export (CSV) to begin. Then open Players to scout individuals.")

# ----------------------------
# Upload
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload Wyscout League Export (CSV)",
    type=["csv"],
    accept_multiple_files=False,
    key="wyscout_uploader",
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Basic cleaning for key identifiers (safe if columns missing)
        for col in ["Player", "Team", "Position"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Store in session state for use in pages/
        st.session_state["data"] = df

        st.success(f"Loaded {len(df):,} rows and {len(df.columns):,} columns.")

        # Quick status cards (optional but helpful)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Players", f"{df['Player'].nunique():,}" if "Player" in df.columns else "—")
        c2.metric("Teams", f"{df['Team'].nunique():,}" if "Team" in df.columns else "—")
        c3.metric("Positions", f"{df['Position'].nunique():,}" if "Position" in df.columns else "—")
        c4.metric("Columns", f"{len(df.columns):,}")

    except Exception as e:
        st.error("Could not read the uploaded CSV. Please verify the file format and try again.")
        st.exception(e)
else:
    st.info("No file uploaded yet.")

st.divider()

# ----------------------------
# Navigation (optional; sidebar pages should also appear)
# ----------------------------
st.subheader("Navigate")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Players"):
        st.switch_page("pages/2_Players.py")

with col2:
    if st.button("Compare"):
        # Only works if the file exists
        st.switch_page("pages/3_Compare.py")

with col3:
    # Only works if the file exists
    if st.button("Shortlists"):
        st.switch_page("pages/4_Shortlists.py")

st.caption("Tip: You can also use the left sidebar page navigation.")
