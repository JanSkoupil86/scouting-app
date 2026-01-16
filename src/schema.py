# src/schema.py

POSITION_GROUPS = {
    "GK": ["GK"],
    "DEF": ["CB", "RCB", "LCB", "RB", "LB", "RWB", "LWB"],
    "MID": ["DMF", "CMF", "AMF", "LMF", "RMF"],
    "FWD": ["CF", "SS", "LWF", "RWF", "LW", "RW"],
}

def map_position_group(position_str: str) -> str:
    if not position_str:
        return "OTHER"

    pos = position_str.upper()
    for group, values in POSITION_GROUPS.items():
        if any(p in pos for p in values):
            return group
    return "OTHER"
