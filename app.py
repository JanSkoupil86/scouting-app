import streamlit as st
import pandas as pd

st.set_page_config(page_title="Scouting App", layout="wide")

st.title("Scouting App")
st.caption("Upload a Wyscout CSV once here. Then use the pages on the left.")

# ----------------------------
# Dataset settings (Season)
# ----------------------------
season_options = ["24/25", "25/26", "2024", "2025"]
selected_season = st.selectbox("Season", season_options, index=0)

# ----------------------------
# Upload (ONLY place in the app)
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

        # Clean key identifiers
        for col in ["Player", "Team", "Position"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Ensure Season exists (use existing season column if present; otherwise assign from selector)
        possible_season_cols = ["Season", "season", "Season name", "seasonName", "Year"]
        season_col = next((c for c in possible_season_cols if c in df.columns), None)

        if season_col is None:
            df["Season"] = selected_season
        else:
            df["Season"] = df[season_col].astype(str).str.strip()

        # Normalize common season formats
        def _norm_season(x: str) -> str:
            s = (x or "").strip().replace(" ", "").replace("-", "/")
            if s in ["2024/2025", "2024/25"]:
                return "24/25"
            if s in ["2025/2026", "2025/26"]:
                return "25/26"
            if s in ["2024", "2025", "24/25", "25/26"]:
                return s
            return s

        df["Season"] = df["Season"].astype(str).apply(_norm_season)

        # Store dataset for all pages
        st.session_state["data"] = df
        st.session_state["selected_season"] = selected_season

        st.success(f"Loaded {len(df):,} rows and {len(df.columns):,} columns.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Players", f"{df['Player'].nunique():,}" if "Player" in df.columns else "—")
        c2.metric("Teams", f"{df['Team'].nunique():,}" if "Team" in df.columns else "—")
        c3.metric("Positions", f"{df['Position'].nunique():,}" if "Position" in df.columns else "—")
        c4.metric("Season", selected_season)

    except Exception as e:
        st.error("Could not read the uploaded CSV. Please verify the file format and try again.")
        st.exception(e)
else:
    st.info("No file uploaded yet. Upload a CSV to enable Players/Compare/Shortlists pages.")

st.divider()

st.subheader("Navigate")
st.caption("You can also use the left sidebar page navigation.")

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
