# src/profiles.py

PROFILES = {
    # 🧤 GOALKEEPERS
    "Classic Goalkeeper": [
        "Save rate, %", "Prevented goals per 90", "Conceded goals per 90", "Shots against per 90",
        "Aerial duels won, %", "Exits per 90", "Aerial duels per 90",
        "Accurate long passes, %", "Accurate passes, %", "Average pass length, m"
    ],
    "Sweeper Keeper": [
        "Exits per 90", "Aerial duels per 90", "Aerial duels won, %", "Shots against per 90",
        "Prevented goals per 90", "Save rate, %", "Progressive passes per 90",
        "Forward passes per 90", "Accurate long passes, %", "Passes to final third per 90"
    ],
    "Build-Up Keeper": [
        "Accurate passes, %", "Accurate long passes, %", "Progressive passes per 90",
        "Forward passes per 90", "Passes to final third per 90", "Average pass length, m",
        "Passes per 90", "Save rate, %", "Prevented goals per 90", "Exits per 90"
    ],

    # 🛡️ CENTRE-BACKS
    "Ball-Playing CB": [
        "Progressive passes per 90",
        "Accurate progressive passes, %",
        "Forward passes per 90",
        "Accurate passes, %",
        "Accurate long passes, %",
        "Passes per 90",
        "Average pass length, m",
        "Interceptions per 90",
        "Defensive duels won, %",
        "Aerial duels won, %"
    ],
    "Combative CB / Stopper": [
        "Defensive duels per 90",
        "Defensive duels won, %",
        "Aerial duels per 90",
        "Aerial duels won, %",
        "Shots blocked per 90",
        "Interceptions per 90",
        "Fouls per 90",
        "Successful defensive actions per 90",
        "Passes per 90",
        "Accurate passes, %"
    ],
    "Libero / Middle Pin CB": [
        "Progressive passes per 90",
        "Accurate long passes, %",
        "Passes to final third per 90",
        "Accurate passes, %",
        "Deep completions per 90",
        "Smart passes per 90",
        "xA per 90",
        "Interceptions per 90",
        "Aerial duels won, %",
        "Defensive duels won, %"
    ],
    "Wide CB (in 3)": [
        "Defensive duels per 90",
        "Defensive duels won, %",
        "Progressive runs per 90",
        "Interceptions per 90",
        "Aerial duels won, %",
        "Successful defensive actions per 90",
        "Crosses per 90",
        "Accurate crosses, %",
        "Touches in box per 90",
        "Progressive passes per 90"
    ],

    # ⚙️ MIDFIELDERS
    "Defensive Midfielder #6": [
        "Interceptions per 90", "Defensive duels per 90", "Defensive duels won, %",
        "Successful defensive actions per 90", "Accurate passes, %", "Forward passes per 90",
        "Passes to final third per 90", "Progressive passes per 90", "Average pass length, m",
        "Aerial duels won, %"
    ],
    "Attacking Midfielder #8": [
        "Progressive passes per 90", "Accurate progressive passes, %", "Progressive runs per 90",
        "xA per 90", "Shots per 90", "Touches in box per 90", "Interceptions per 90",
        "Key passes per 90", "Deep completions per 90", "Successful attacking actions per 90"
    ],
    "Deep-Lying Playmaker": [
        "Progressive passes per 90", "Accurate progressive passes, %", "Received passes per 90",
        "Accurate long passes, %", "Forward passes per 90", "Passes per 90",
        "Passes to final third per 90", "Interceptions per 90", "Defensive duels per 90",
        "Aerial duels won, %"
    ],
    "Box-to-Box Midfielder": [
        "Progressive runs per 90", "xG per 90", "Shots per 90", "Interceptions per 90",
        "Defensive duels per 90", "Defensive duels won, %", "Touches in box per 90",
        "Successful attacking actions per 90", "Forward passes per 90", "Passes to final third per 90"
    ],

    # 🌊 WIDE / ATTACKING ROLES
    "Full-Back": [
        "Defensive duels per 90", "Defensive duels won, %", "Interceptions per 90",
        "Crosses per 90", "Accurate crosses, %", "Progressive runs per 90",
        "Progressive passes per 90", "Forward passes per 90", "Successful defensive actions per 90",
        "Deep completions per 90"
    ],
    "Wing-Back": [
        "Progressive runs per 90", "Crosses per 90", "Accurate crosses, %",
        "Shot assists per 90", "Progressive passes per 90", "Interceptions per 90",
        "Defensive duels per 90", "Defensive duels won, %", "Touches in box per 90",
        "Successful attacking actions per 90"
    ],
    "Inverted Full-Back": [
        "Progressive passes per 90", "Progressive runs per 90", "Forward passes per 90",
        "Accurate passes, %", "Accurate short / medium passes, %", "Smart passes per 90",
        "Defensive duels won, %", "Interceptions per 90", "Successful defensive actions per 90",
        "Aerial duels won, %"
    ],
    "Classic Winger": [
        "Dribbles per 90", "Successful dribbles, %", "Progressive runs per 90", "Crosses per 90",
        "Accurate crosses, %", "Shot assists per 90", "Touches in box per 90",
        "Shots per 90", "xA per 90", "Successful attacking actions per 90"
    ],
    "Inverted Winger": [
        "Shots per 90", "xG per 90", "xA per 90", "Progressive runs per 90",
        "Shot assists per 90", "Touches in box per 90", "Dribbles per 90",
        "Successful dribbles, %", "Deep completions per 90", "Key passes per 90"
    ],
    "Playmaker #10": [
        "Progressive passes per 90", "Accurate progressive passes, %", "Deep completions per 90",
        "Key passes per 90", "xA per 90", "Shot assists per 90", "Shots per 90",
        "xG per 90", "Progressive runs per 90", "Successful attacking actions per 90"
    ],

    # ⚡ FORWARDS
    "Target Man #9": [
        "Aerial duels per 90", "Aerial duels won, %", "Received long passes per 90",
        "Passes to final third per 90", "Fouls suffered per 90", "xG per 90",
        "Shots per 90", "Non-penalty goals per 90", "Touches in box per 90", "Received passes per 90"
    ],
    "Poacher": [
        "Non-penalty goals per 90", "xG per 90", "Shots per 90", "Goal conversion, %",
        "Touches in box per 90", "Received passes per 90", "xA per 90",
        "Progressive runs per 90", "Key passes per 90", "Successful attacking actions per 90"
    ],
    "Pressing Forward": [
        "Defensive duels per 90", "Defensive duels won, %", "Interceptions per 90",
        "Successful defensive actions per 90", "Progressive runs per 90", "Shots per 90",
        "xG per 90", "xA per 90", "Touches in box per 90", "Successful attacking actions per 90"
    ],
    "Creative Forward / False 9": [
        "Progressive passes per 90",
        "Accurate progressive passes, %",
        "Deep completions per 90",
        "Key passes per 90",
        "xA per 90",
        "Progressive runs per 90",
        "Received passes per 90",
        "Shots per 90",
        "xG per 90",
        "Touches in box per 90"
    ],
    "Wide Forward / Inside 9": [
        "Progressive runs per 90",
        "Dribbles per 90",
        "Successful dribbles, %",
        "Shots per 90",
        "xG per 90",
        "xA per 90",
        "Touches in box per 90",
        "Deep completions per 90",
        "Key passes per 90",
        "Successful attacking actions per 90"
    ],
}
