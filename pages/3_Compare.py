import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.profiles import PROFILES

st.set_page_config(page_title="Compare", layout="wide")
st.title("Compare")

# ----------------------------
# Load data
# ----------------------------
if "data" not in st.session_state:
    st.warning("No data loaded yet. Go to the **app** page and upload a CSV.")
    st.stop()

df = st.session_state["data"]
if not isinstance(df, pd.DataFrame) or df.empty:
    st.error("Loaded dataset is empty or invalid. Re-upload the CSV on the **app** page.")
    st.stop()

required = ["Player", "Minutes played"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Your CSV is missing required columns: {missing}")
    st.stop()

# Column preferences
league_col = "Competition" if "Competition" in df.columns else ("League" if "League" in df.columns else None)
team_col = (
    "Team within selected timeframe"
    if "Team within selected timeframe" in df.columns
    else ("Team" if "Team" in df.columns else None)
)
pos_col = "Main Position" if "Main Position" in df.columns else ("Position" if "Position" in df.columns else None)

if league_col is None or team_col is None or pos_col is None:
    st.error("Missing required columns. Need League/Competition, Team, and Main Position/Position columns.")
    st.stop()

# Standardize string columns
for col in ["Player", league_col, team_col, pos_col]:
    df[col] = df[col].astype(str).str.strip()

# ----------------------------
# Helpers
# ----------------------------
def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def percentile_rank(value, population: pd.Series):
    pop = to_num(population).dropna()
    if pop.empty or pd.isna(value):
        return np.nan
    return round((pop < float(value)).mean() * 100.0, 2)


def map_position_group(position_str: str) -> str:
    if not position_str:
        return "OTHER"
    pos = str(position_str).upper()
    if "GK" in pos:
        return "GK"
    if any(p in pos for p in ["CB", "LCB", "RCB", "LB", "RB", "LWB", "RWB"]):
        return "DEF"
    if any(p in pos for p in ["DMF", "CMF", "AMF", "LMF", "RMF"]):
        return "MID"
    if any(p in pos for p in ["CF", "SS", "ST", "LWF", "RWF", "LW", "RW"]):
        return "FWD"
    return "OTHER"


def unique_sorted(series: pd.Series) -> list[str]:
    s = series.dropna().astype(str).str.strip()
    s = s[(s != "") & (s.str.lower() != "nan")]
    return sorted(s.unique().tolist())


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def extract_season_from_text(text: str) -> str | None:
    """
    Extract season tokens from league text.
    Handles:
      - 2024-25 or 2024/25 -> 24/25
      - 24/25 -> 24/25
      - standalone year 2024 -> 2024 (spring-fall leagues)
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


# ----------------------------
# Ensure Season exists
# ----------------------------
df2 = df.copy()
if "Season" not in df2.columns:
    df2["Season"] = df2[league_col].apply(extract_season_from_text)

df2["Season"] = df2["Season"].astype(str).str.strip()
df2.loc[df2["Season"].str.lower().isin(["nan", "none", ""]), "Season"] = np.nan

if df2["Season"].dropna().empty:
    st.error(
        "Could not derive a Season column. League/Competition must contain tokens like '2024-25', '24/25', or '2024'."
    )
    st.stop()

# ----------------------------
# Top controls
# ----------------------------
c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    n_players = st.number_input("Number of players to compare", 2, 6, 2, 1)

with c2:
    cross_season = st.toggle(
        "Compare across seasons (per slot)",
        value=True,
        help="ON: each player slot chooses its own season. OFF: one global season filter for all slots.",
    )

with c3:
    fill_opacity = st.slider("Radar fill opacity", 0.05, 0.60, 0.25, 0.05)

global_season = None
if not cross_season:
    seasons = unique_sorted(df2["Season"])
    global_season = st.selectbox("Season (global)", ["All"] + seasons, index=0)

df_base_global = df2.copy()
if (not cross_season) and global_season and global_season != "All":
    df_base_global = df_base_global[df_base_global["Season"].astype(str) == str(global_season)]

if df_base_global.empty:
    st.info("No rows available after Season filtering.")
    st.stop()

st.divider()

# ----------------------------
# Metrics catalogue (same list used to validate available metrics)
# ----------------------------
METRICS_CATALOGUE = [
    "Goals", "xG", "Assists", "xA", "Duels per 90", "Duels won, %", "Successful defensive actions per 90",
    "Defensive duels per 90", "Defensive duels won, %", "Aerial duels per 90", "Aerial duels won, %",
    "Sliding tackles per 90", "PAdj Sliding tackles", "Shots blocked per 90", "Interceptions per 90",
    "PAdj Interceptions", "Fouls per 90", "Yellow cards", "Yellow cards per 90", "Red cards", "Red cards per 90",
    "Successful attacking actions per 90", "Goals per 90", "Non-penalty goals", "Non-penalty goals per 90",
    "xG per 90", "Head goals", "Head goals per 90", "Shots", "Shots per 90", "Shots on target, %", "Goal conversion, %",
    "Assists per 90", "Crosses per 90", "Accurate crosses, %", "Crosses from left flank per 90",
    "Accurate crosses from left flank, %", "Crosses from right flank per 90", "Accurate crosses from right flank, %",
    "Crosses to goalie box per 90", "Dribbles per 90", "Successful dribbles, %", "Offensive duels per 90",
    "Offensive duels won, %", "Touches in box per 90", "Progressive runs per 90", "Accelerations per 90",
    "Received passes per 90", "Received long passes per 90", "Fouls suffered per 90", "Passes per 90",
    "Accurate passes, %", "Forward passes per 90", "Accurate forward passes, %", "Back passes per 90",
    "Accurate back passes, %", "Lateral passes per 90", "Accurate lateral passes, %", "Short / medium passes per 90",
    "Accurate short / medium passes, %", "Long passes per 90", "Accurate long passes, %", "Average pass length, m",
    "Average long pass length, m", "xA per 90", "Shot assists per 90", "Second assists per 90", "Third assists per 90",
    "Smart passes per 90", "Accurate smart passes, %", "Key passes per 90", "Passes to final third per 90",
    "Accurate passes to final third, %", "Passes to penalty area per 90", "Accurate passes to penalty area, %",
    "Through passes per 90", "Accurate through passes, %", "Deep completions per 90", "Deep completed crosses per 90",
    "Progressive passes per 90", "Accurate progressive passes, %", "Conceded goals", "Conceded goals per 90",
    "Shots against", "Shots against per 90", "Clean sheets", "Save rate, %", "xG against", "xG against per 90",
    "Prevented goals", "Prevented goals per 90", "Back passes received as GK per 90", "Exits per 90",
    "Aerial duels per 90.1", "Free kicks per 90", "Direct free kicks per 90", "Direct free kicks on target, %",
    "Corners per 90", "Penalties taken", "Penalty conversion, %",
]

LOWER_BETTER = {
    "Fouls per 90",
    "Yellow cards", "Yellow cards per 90",
    "Red cards", "Red cards per 90",
    "Conceded goals", "Conceded goals per 90",
    "Shots against", "Shots against per 90",
    "xG against", "xG against per 90",
}

available_metrics = [m for m in METRICS_CATALOGUE if m in df_base_global.columns]

# ----------------------------
# Metrics selection: Manual + Profiles
# ----------------------------
st.subheader("Metrics")

if not available_metrics:
    st.warning("None of the requested metrics were found in your dataset columns.")
    st.stop()

profile_names = ["Manual (no profile)"] + sorted(PROFILES.keys())

mcol1, mcol2, mcol3, mcol4 = st.columns([2, 1, 1, 1])

with mcol1:
    selected_profile = st.selectbox(
        "Profiles (optional)",
        options=profile_names,
        index=0,
        help="Pick a preset profile to populate metrics. You can still edit manually after applying.",
    )

with mcol2:
    apply_mode = st.radio(
        "Apply mode",
        options=["Replace", "Add"],
        index=0,
        horizontal=True,
        help="Replace overwrites current metrics. Add appends profile metrics.",
    )

with mcol3:
    clear_metrics = st.button("Clear", use_container_width=True)

with mcol4:
    apply_profile = st.button(
        "Apply",
        use_container_width=True,
        disabled=(selected_profile == "Manual (no profile)"),
    )

# default compare metrics
if "compare_metrics" not in st.session_state:
    defaults = [m for m in ["Goals per 90", "xG per 90", "Assists per 90", "xA per 90", "Duels won, %"] if m in available_metrics]
    if not defaults:
        defaults = available_metrics[:10]
    st.session_state["compare_metrics"] = defaults

if clear_metrics:
    st.session_state["compare_metrics"] = []

if apply_profile and selected_profile != "Manual (no profile)":
    prof_metrics = [m for m in PROFILES[selected_profile] if m in available_metrics]
    if not prof_metrics:
        st.warning("This profile has no metrics available in the current dataset.")
    else:
        if apply_mode == "Replace":
            st.session_state["compare_metrics"] = prof_metrics
        else:
            merged = list(dict.fromkeys(st.session_state["compare_metrics"] + prof_metrics))
            st.session_state["compare_metrics"] = merged

if selected_profile != "Manual (no profile)":
    missing_in_data = [m for m in PROFILES[selected_profile] if m not in available_metrics]
    if missing_in_data:
        with st.expander("Profile metrics not found in this dataset"):
            st.write(missing_in_data)

metrics = st.multiselect(
    "Select metrics for comparison",
    options=available_metrics,
    default=st.session_state["compare_metrics"],
)
st.session_state["compare_metrics"] = metrics

if not metrics:
    st.stop()

st.divider()

# ----------------------------
# Side-by-side selectors (Season first)
# ----------------------------
st.subheader("Pick players (side by side)")
st.caption("Each slot: Season → League → Team → Main Position → Player. This prevents duplicate player rows across seasons.")

slots_per_row = 3
rows = (int(n_players) + slots_per_row - 1) // slots_per_row

selections = []
slot_idx = 0

for r in range(rows):
    cols = st.columns(min(slots_per_row, int(n_players) - r * slots_per_row))
    for col in cols:
        slot_idx += 1
        with col:
            st.markdown(f"## Player {slot_idx}")

            base_for_slot = df2.copy()
            if (not cross_season) and global_season and global_season != "All":
                base_for_slot = base_for_slot[base_for_slot["Season"].astype(str) == str(global_season)]

            seasons = unique_sorted(base_for_slot["Season"])
            if not seasons:
                st.warning("No seasons available.")
                continue

            if cross_season:
                season_val = st.selectbox(f"Season {slot_idx}", options=seasons, key=f"season_{slot_idx}")
            else:
                season_val = global_season if global_season != "All" else seasons[0]
                st.selectbox(
                    f"Season {slot_idx}",
                    options=[season_val],
                    index=0,
                    key=f"season_{slot_idx}_fixed",
                    disabled=True,
                )

            df_s = base_for_slot[base_for_slot["Season"].astype(str) == str(season_val)]
            leagues = unique_sorted(df_s[league_col])
            if not leagues:
                st.warning("No leagues available for this season.")
                continue

            league_val = st.selectbox(f"League {slot_idx}", options=leagues, key=f"league_{slot_idx}")
            df_l = df_s[df_s[league_col].astype(str) == str(league_val)]

            teams = unique_sorted(df_l[team_col])
            if not teams:
                st.warning("No teams available for this league.")
                continue

            team_val = st.selectbox(f"Team {slot_idx}", options=teams, key=f"team_{slot_idx}")
            df_t = df_l[df_l[team_col].astype(str) == str(team_val)]

            positions = unique_sorted(df_t[pos_col])
            if not positions:
                st.warning("No positions available for this team.")
                continue

            pos_val = st.selectbox(f"Position {slot_idx}", options=positions, key=f"pos_{slot_idx}")
            df_p = df_t[df_t[pos_col].astype(str) == str(pos_val)]

            players = unique_sorted(df_p["Player"])
            if not players:
                st.warning("No players available.")
                continue

            player_val = st.selectbox(f"Player {slot_idx}", options=players, key=f"player_{slot_idx}")

            # resolve (if still duplicates, pick max minutes)
            cand = df_p[df_p["Player"].astype(str) == str(player_val)].copy()
            cand["__mins"] = to_num(cand["Minutes played"]).fillna(0)
            cand = cand.sort_values("__mins", ascending=False)
            chosen = cand.iloc[0].drop(labels=["__mins"])
            selections.append(chosen)

sel_df = pd.DataFrame(selections).reset_index(drop=True)
if len(sel_df) != int(n_players):
    st.info("Please complete selection in all player slots to continue.")
    st.stop()

sel_df = sel_df.copy()
sel_df["Position group"] = sel_df[pos_col].apply(map_position_group)

st.divider()

# ----------------------------
# Profiles block
# ----------------------------
st.subheader("Profiles")

profile_fields = ["Season", league_col, team_col, pos_col, "Position group", "Age", "Minutes played", "Matches played", "Market value"]
profile_fields = [c for c in profile_fields if c in sel_df.columns]

pcols = st.columns(len(sel_df))
for i, col in enumerate(pcols):
    with col:
        st.markdown(f"### {sel_df.iloc[i]['Player']}")
        for f in profile_fields:
            v = sel_df.iloc[i].get(f, "")
            if f in ["Minutes played", "Matches played", "Age"] and pd.notna(v):
                try:
                    v = int(float(v))
                except Exception:
                    pass
            if f == "Market value" and pd.notna(v):
                try:
                    v = f"{int(float(v)):,}"
                except Exception:
                    pass
            st.write(f"**{f}:** {v}")

st.divider()

# ----------------------------
# Stat tables (winner highlight, no scale)
# ----------------------------
st.subheader("Stat breakdown table (no scale)")

peer_mode = st.radio(
    "Percentile peer pool",
    options=[
        "Within each player's Season + League + Position group",
        "Within each player's Season + League (no position grouping)",
        "Global (all filtered data)",
    ],
    index=0,
    horizontal=True,
)

global_pool = df_base_global.copy()
global_pool["Position group"] = global_pool[pos_col].apply(map_position_group)

values_tbl = pd.DataFrame(index=metrics)
pct_tbl = pd.DataFrame(index=metrics)

for _, r in sel_df.iterrows():
    player_name = str(r["Player"]).strip()
    player_league = str(r[league_col]).strip()
    player_season = str(r["Season"]).strip()
    player_group = str(r["Position group"]).strip()

    if peer_mode == "Global (all filtered data)":
        pool = global_pool
    else:
        pool = df2.copy()
        pool = pool[(pool["Season"].astype(str) == player_season) & (pool[league_col].astype(str) == player_league)].copy()
        pool["Position group"] = pool[pos_col].apply(map_position_group)
        if peer_mode == "Within each player's Season + League + Position group":
            pool = pool[pool["Position group"] == player_group]

    vals = []
    pcts = []
    for m in metrics:
        v = to_num(pd.Series([r.get(m, np.nan)])).iloc[0]
        pop = pool[m] if m in pool.columns else pd.Series(dtype=float)
        pct = percentile_rank(v, pop)
        if m in LOWER_BETTER and not pd.isna(pct):
            pct = round(100.0 - pct, 2)
        vals.append(v)
        pcts.append(pct)

    values_tbl[f"{player_name} ({player_season})"] = vals
    pct_tbl[f"{player_name} ({player_season})"] = pcts

values_tbl = values_tbl.round(2)
pct_tbl = pct_tbl.round(2)


def style_winners(values: pd.DataFrame):
    def _row_style(row: pd.Series):
        metric = row.name
        nums = row.apply(_safe_float)
        if nums.isna().all():
            return [""] * len(row)
        best = nums.min(skipna=True) if metric in LOWER_BETTER else nums.max(skipna=True)
        out = []
        for v in nums:
            if pd.isna(v) or pd.isna(best):
                out.append("")
            elif np.isclose(v, best, rtol=0, atol=1e-12):
                out.append("color: #1a7f37; font-weight: 700;")
            else:
                out.append("")
        return out
    return values.style.format("{:.2f}").apply(_row_style, axis=1)


def style_pct_winners(pcts: pd.DataFrame):
    def _row_style(row: pd.Series):
        nums = row.apply(_safe_float)
        if nums.isna().all():
            return [""] * len(row)
        best = nums.max(skipna=True)
        out = []
        for v in nums:
            if pd.isna(v) or pd.isna(best):
                out.append("")
            elif np.isclose(v, best, rtol=0, atol=1e-12):
                out.append("color: #1a7f37; font-weight: 700;")
            else:
                out.append("")
        return out
    return pcts.style.format("{:.2f}").apply(_row_style, axis=1)


st.write("**Values (raw columns)**")
st.dataframe(style_winners(values_tbl), use_container_width=True, height=420)

st.write("**Percentiles (0–100)**")
st.dataframe(style_pct_winners(pct_tbl), use_container_width=True, height=420)

st.divider()

# ----------------------------
# Radar chart (percentiles) — filled
# ----------------------------
st.subheader("Radar (percentiles)")

radar_metrics = pct_tbl.index.tolist()
if len(radar_metrics) < 3:
    st.info("Select at least 3 metrics to show a radar chart.")
    st.stop()

PLOTLY_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

fig = go.Figure()
for i, col_name in enumerate(pct_tbl.columns):
    vals = pct_tbl[col_name].tolist()
    color = PLOTLY_COLORS[i % len(PLOTLY_COLORS)]
    r_, g_, b_ = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    fig.add_trace(
        go.Scatterpolar(
            r=vals + [vals[0]],
            theta=radar_metrics + [radar_metrics[0]],
            name=col_name,
            mode="lines",
            line=dict(width=2, color=color),
            fill="toself",
            fillcolor=f"rgba({r_},{g_},{b_},{fill_opacity})",
        )
    )

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
    showlegend=True,
    height=560,
    margin=dict(l=20, r=20, t=30, b=20),
)

st.plotly_chart(fig, use_container_width=True)
