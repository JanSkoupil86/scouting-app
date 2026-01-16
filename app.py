import re
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Scouting App", layout="wide")

st.title("Scouting App")
st.caption("Upload a Wyscout CSV once here. Then use Players to filter and scout individuals.")

uploaded_file = st.file_uploader(
    "Upload Wyscout League Export (CSV)",
    type=["csv"],
    accept_multiple_files=False,
    key="wyscout_uploader",
)

def _norm_season(token: str) -> str | None:
    """
    Normalize season tokens to one of:
    - 24/25, 25/26 (autumn-spring)
    - 2024, 2025 (spring-fall)
    Returns None if not recognized.
    """
    if token is None:
        return None
    s = str(token).strip().replace(" ", "")
    s = s.replace("-", "/").replace("_", "/")

    # common patterns
    if s in {"2024/25", "2024/2025"}:
        return "24/25"
    if s in {"2025/26", "2025/2026"}:
        return "25/26"
    if s in {"24/25", "25/26", "2024", "2025"}:
        return s

    # handle "2024-25" already converted to "2024/25"
    if s == "2024/25":
        return "24/25"
    if s == "2025/26":
        return "25/26"

    return None

def _extract_season_from_league(league_value: str) -> tuple[str, str]:
    """
    Returns (competition_name, season_string)
    competition_name: league without trailing season tokens
    season_string: normalized season if detected, else ""
    """
    if league_value is None or pd.isna(league_value):
        return "", ""

    raw = str(league_value).strip()

    # Try to detect patterns like:
    # "Netherlands Eredivisie 2024-25"
    # "Brazil Serie A 2024"
    # We'll search for a trailing season-like token.
    m = re.search(r"(\b20\d{2}\s*[-/]\s*\d{2}\b|\b20\d{2}\b|\b\d{2}\s*/\s*\d{2}\b)\s*$", raw)
    season = ""
    competition = raw

    if m:
        season_token = m.group(1)
        norm = _norm_season(season_token)
        if norm:
            season = norm
            competition = raw[: m.start()].strip()
        else:
            # keep competition as raw if we can't normalize
            season = ""

    return competition, season

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Clean key identifiers
        for col in ["Player", "Team", "Position"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Derive Season + Competition from League if League exists
        if "League" in df.columns:
            comp_season = df["League"].apply(_extract_season_from_league)
            df["Competition"] = comp_season.apply(lambda x: x[0])
            df["Season"] = comp_season.apply(lambda x: x[1])

        # If Season is still missing/blank, leave as empty string (filters will handle)
        if "Season" not in df.columns:
            df["Season"] = ""

        st.session_state["data"] = df

        st.success(f"Loaded {len(df):,} rows and {len(df.columns):,} columns.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Players", f"{df['Player'].nunique():,}" if "Player" in df.columns else "—")
        c2.metric("Teams", f"{df['Team'].nunique():,}" if "Team" in df.columns else "—")
        c3.metric("Competitions", f"{df['Competition'].nunique():,}" if "Competition" in df.columns else "—")
        c4.metric("Seasons", f"{df['Season'].replace('', pd.NA).dropna().nunique():,}" if "Season" in df.columns else "—")

    except Exception as e:
        st.error("Could not read the uploaded CSV. Please verify the file format and try again.")
        st.exception(e)
else:
    st.info("No file uploaded yet. Upload a CSV to enable Players/Compare/Shortlists pages.")

st.divider()
st.subheader("Navigate")
st.caption("Use the left sidebar page navigation.")
