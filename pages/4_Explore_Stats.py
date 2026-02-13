import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Explore Stats", layout="wide")
st.title("Explore Stats")

# -----------------------------
# Require uploaded data
# -----------------------------
if "data" not in st.session_state:
    st.warning("No data loaded yet. Go to **Home** and upload a CSV.")
    st.stop()

df = st.session_state["data"]
if df is None or df.empty:
    st.warning("Dataset is empty. Re-upload your CSV on Home.")
    st.stop()

# -----------------------------
# Column mapping (your conventions)
# -----------------------------
league_col = "Competition" if "Competition" in df.columns else ("League" if "League" in df.columns else None)
team_col = "Team within selected timeframe" if "Team within selected timeframe" in df.columns else ("Team" if "Team" in df.columns else None)
pos_col = "Main Position" if "Main Position" in df.columns else ("Position" if "Position" in df.columns else None)

required = ["Player", "Minutes played"]
missing = [c for c in required if c not in df.columns]
if league_col is None:
    missing.append("Competition/League")
if pos_col is None:
    missing.append("Main Position/Position")
if team_col is None:
    missing.append("Team within selected timeframe/Team")

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Clean text cols
for c in ["Player", league_col, team_col, pos_col]:
    df[c] = df[c].astype(str).str.strip()

# -----------------------------
# Helpers
# -----------------------------
def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def unique_sorted(series):
    s = series.dropna().astype(str).str.strip()
    s = s[(s != "") & (s.str.lower() != "nan")]
    return sorted(s.unique().tolist())

def extract_season_from_text(text: str) -> str | None:
    """
    Extract season token from League/Competition text.
    Supports:
      - 2024-25 / 2024/25 -> 24/25
      - 24/25 -> 24/25
      - 2024 / 2025 (spring-fall) -> 2024
    """
    if text is None:
        return None
    t = str(text)

    m = re.search(r"\b(20\d{2})\s*[-/]\s*(\d{2})\b", t)
    if m:
        y1 = int(m.group(1))
        y2 = int(m.group(2))
        return f"{str(y1)[2:]}/{y2:02d}"

    m = re.search(r"\b(\d{2})\s*/\s*(\d{2})\b", t)
    if m:
        return f"{int(m.group(1)):02d}/{int(m.group(2)):02d}"

    m = re.search(r"\b(20\d{2})\b", t)
    if m:
        return m.group(1)

    return None

def position_group(main_pos: str) -> str:
    if not main_pos:
        return "Other"
    p = str(main_pos).upper()
    if "GK" in p:
        return "Goalkeeper"
    if any(x in p for x in ["CB", "LCB", "RCB"]):
        return "Center Back"
    if any(x in p for x in ["LB", "LWB"]):
        return "Left Back/WB"
    if any(x in p for x in ["RB", "RWB"]):
        return "Right Back/WB"
    if any(x in p for x in ["DMF"]):
        return "Defensive Midfield"
    if any(x in p for x in ["CMF", "LCMF", "RCMF"]):
        return "Center Midfield"
    if any(x in p for x in ["AMF"]):
        return "Attacking Midfield"
    if any(x in p for x in ["LWF", "LW"]):
        return "Left Wing"
    if any(x in p for x in ["RWF", "RW"]):
        return "Right Wing"
    if any(x in p for x in ["CF", "ST", "SS"]):
        return "Center Forward"
    return "Other"

# -----------------------------
# Ensure Season exists
# -----------------------------
df2 = df.copy()
if "Season" not in df2.columns:
    df2["Season"] = df2[league_col].apply(extract_season_from_text)

df2["Season"] = df2["Season"].astype(str).str.strip()
df2.loc[df2["Season"].str.lower().isin(["nan", "none", ""]), "Season"] = np.nan

# -----------------------------
# Derive Position Group
# -----------------------------
df2["Position Group"] = df2[pos_col].apply(position_group)

# -----------------------------
# Metric options (numeric-like columns)
# -----------------------------
# Keep it simple: pick from numeric columns (excluding obvious IDs)
exclude_cols = {league_col, team_col, pos_col, "Position Group", "Season", "Player"}
candidate_cols = [c for c in df2.columns if c not in exclude_cols]

numeric_cols = []
for c in candidate_cols:
    # keep if at least some values are numeric
    s = to_num(df2[c])
    if s.notna().sum() >= max(20, int(0.05 * len(df2))):
        numeric_cols.append(c)

numeric_cols = sorted(numeric_cols)

# Add some typical defaults if present
default_x = "Successful defensive actions per 90" if "Successful defensive actions per 90" in df2.columns else (numeric_cols[0] if numeric_cols else None)
default_y = "Defensive duels per 90" if "Defensive duels per 90" in df2.columns else (numeric_cols[1] if len(numeric_cols) > 1 else default_x)

# -----------------------------
# Controls (top layout like screenshot)
# -----------------------------
c1, c2 = st.columns(2)
with c1:
    season_opt = ["All"] + unique_sorted(df2["Season"])
    season = st.selectbox("Season", season_opt, index=0)

with c2:
    league_list = unique_sorted(df2[league_col])
    league = st.selectbox("League", league_list, index=0 if league_list else None)

dff = df2.copy()
if season != "All":
    dff = dff[dff["Season"].astype(str) == str(season)]
dff = dff[dff[league_col].astype(str) == str(league)]

c3, c4 = st.columns(2)
with c3:
    pos_groups = unique_sorted(dff["Position Group"])
    pos_group = st.selectbox("Position Group", pos_groups, index=0 if pos_groups else None)

with c4:
    show_names = st.checkbox("Show player names on scatter", value=False)

dff = dff[dff["Position Group"].astype(str) == str(pos_group)]

c5, c6 = st.columns(2)
with c5:
    x_metric = st.selectbox("X metric", options=numeric_cols, index=numeric_cols.index(default_x) if default_x in numeric_cols else 0)
with c6:
    y_metric = st.selectbox("Y metric", options=numeric_cols, index=numeric_cols.index(default_y) if default_y in numeric_cols else min(1, len(numeric_cols)-1))

# Minimum minutes slider
max_mins = int(to_num(dff["Minutes played"]).fillna(0).max()) if not dff.empty else 0
min_minutes = st.slider("Minimum Minutes", min_value=0, max_value=max(200, max_mins), value=min(200, max_mins), step=50)

# Apply minutes + numeric conversions
dff["Minutes played"] = to_num(dff["Minutes played"]).fillna(0)
dff = dff[dff["Minutes played"] >= min_minutes].copy()

dff[x_metric] = to_num(dff[x_metric])
dff[y_metric] = to_num(dff[y_metric])
dff = dff.dropna(subset=[x_metric, y_metric])

st.caption(
    f"Rows in plot: {len(dff)} | League={league} | Position Group={pos_group} | Minutes≥{min_minutes}"
)

if dff.empty:
    st.info("No players match the filters (minutes/metrics).")
    st.stop()

# -----------------------------
# Scatter plot (X vs Y)
# -----------------------------
title = f"{league} — {pos_group} — {x_metric} vs {y_metric}"
fig = px.scatter(
    dff,
    x=x_metric,
    y=y_metric,
    hover_name="Player",
    hover_data={
        team_col: True,
        "Season": True,
        "Minutes played": True,
    },
)

if show_names:
    fig.update_traces(text=dff["Player"], textposition="top center")

fig.update_layout(
    title=title,
    height=520,
    margin=dict(l=20, r=20, t=60, b=20),
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Table: players in plot
# -----------------------------
st.subheader("Players in plot")

table_cols = ["Player", team_col, "Position Group", x_metric, y_metric, "Minutes played"]
table_cols = [c for c in table_cols if c in dff.columns]

table_df = dff[table_cols].copy()
table_df = table_df.sort_values(y_metric, ascending=False).reset_index(drop=True)

# Format decimals
for m in [x_metric, y_metric]:
    table_df[m] = to_num(table_df[m]).round(2)
table_df["Minutes played"] = to_num(table_df["Minutes played"]).round(0).astype(int)

st.dataframe(table_df, use_container_width=True, height=360)

# -----------------------------
# Scatter 2: Dynamic (same cohort as plot above)
# -----------------------------
st.subheader("Custom scatter (same players as plot above)")

# Build candidate numeric columns for this scatter (including Minutes/Age/Market value if present)
always_try = ["Age", "Market value", "Minutes played", "Matches played"]
candidates = []
for c in always_try + numeric_cols:
    if c in dff.columns and c not in candidates:
        candidates.append(c)

# Fallback if nothing numeric
if not candidates:
    st.info("No numeric columns available for the custom scatter.")
else:
    s1, s2, s3 = st.columns([1, 1, 1])

    with s1:
        x2 = st.selectbox(
            "X axis",
            options=candidates,
            index=candidates.index("Market value") if "Market value" in candidates else 0,
            key="scatter2_x",
        )

    with s2:
        # default Y to Age if present
        y2 = st.selectbox(
            "Y axis",
            options=candidates,
            index=candidates.index("Age") if "Age" in candidates else min(1, len(candidates) - 1),
            key="scatter2_y",
        )

    with s3:
        # how to treat missing values for the selected axes
        missing_mode = st.selectbox(
            "Missing values",
            options=["Drop rows", "Set missing to 0"],
            index=0,
            key="scatter2_missing",
        )

    dd = dff.copy()

    # Convert chosen axes to numeric
    dd[x2] = to_num(dd[x2])
    dd[y2] = to_num(dd[y2])

    if missing_mode == "Set missing to 0":
        dd[x2] = dd[x2].fillna(0)
        dd[y2] = dd[y2].fillna(0)
    else:
        dd = dd.dropna(subset=[x2, y2])

    if dd.empty:
        st.info("No rows available after missing-value handling.")
    else:
        # If Market value is selected, it's common that values are missing or 0
        if x2 == "Market value":
            st.caption("Note: Some market values may be missing; choose 'Set missing to 0' or switch axis.")
        if y2 == "Market value":
            st.caption("Note: Some market values may be missing; choose 'Set missing to 0' or switch axis.")

        fig2 = px.scatter(
            dd,
            x=x2,
            y=y2,
            hover_name="Player",
            hover_data={
                team_col: True,
                "Season": True,
                "Minutes played": True,
            },
        )

        fig2.update_layout(
            title=f"{league} — {pos_group} — {x2} vs {y2}",
            height=420,
            margin=dict(l=20, r=20, t=60, b=20),
        )

        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"Rows in plot: {len(dd)}")
