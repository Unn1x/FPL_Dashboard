# app.py
import os
import json
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="FPL Winning Tool", layout="wide")

# ----------------------------
# Helper functions
# ----------------------------
def parse_kickoff(kickoff_str):
    """Parse kickoff ISO string to timezone-aware datetime (UTC)."""
    if pd.isna(kickoff_str) or kickoff_str is None:
        return None
    try:
        return datetime.fromisoformat(kickoff_str.replace("Z", "+00:00"))
    except Exception:
        return None

def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def ensure_user_squad_file(username):
    fn = f"my_squad_{username}.json"
    if not os.path.exists(fn):
        pos_slots = {"Goalkeepers": 2, "Defenders": 5, "Midfielders": 5, "Forwards": 3}
        base = {pos: [""] * n for pos, n in pos_slots.items()}
        with open(fn, "w") as f:
            json.dump(base, f)
    return fn

def fdr_color_from_text(val):
    """Return background color based on FDR numeric value."""
    if val is None or val == "":
        return ""
    try:
        num = int(val.split("(")[-1].replace(")", ""))
        if num == 1:
            return "background-color: #85e085"  # green
        elif num == 2:
            return "background-color: #ffff99"  # yellow
        elif num == 3:
            return "background-color: #ffd699"  # orange
        else:
            return "background-color: #ff9999"  # red
    except Exception:
        return ""

# ----------------------------
# Load FPL data (cached)
# ----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_data():
    base = "https://fantasy.premierleague.com/api"

    # bootstrap
    bs = requests.get(f"{base}/bootstrap-static/").json()
    elements = pd.DataFrame(bs["elements"])
    teams = pd.DataFrame(bs["teams"])
    events = pd.DataFrame(bs["events"]) if "events" in bs else pd.DataFrame()
    fixtures_df = pd.DataFrame(requests.get(f"{base}/fixtures/").json())

    # maps
    team_map = teams.set_index("id")["name"].to_dict()
    pos_name_map = {1: "Goalkeepers", 2: "Defenders", 3: "Midfielders", 4: "Forwards"}

    # player df
    df = elements[[
        "id", "first_name", "second_name", "web_name", "team", "element_type",
        "now_cost", "total_points", "minutes", "selected_by_percent",
        "form", "points_per_game"
    ]].copy()

    # derived fields
    df["name"] = (df["first_name"].fillna("") + " " + df["second_name"].fillna("")).str.strip()
    df["team_name"] = df["team"].map(team_map)
    df["position"] = df["element_type"].map(pos_name_map)
    df["cost"] = (df["now_cost"] / 10).round(1)
    df["points_per_90"] = df.apply(lambda r: (r["total_points"] / (r["minutes"] / 90)) if r["minutes"] > 0 else 0.0, axis=1)

    # tidy numeric
    df["total_points"] = df["total_points"].astype(int)
    df["minutes"] = df["minutes"].astype(int)
    df["selected_by_percent"] = df["selected_by_percent"].astype(float).round(1)
    df["form"] = df["form"].astype(float).round(1)
    df["points_per_game"] = df["points_per_game"].astype(float).round(1)
    df["points_per_90"] = df["points_per_90"].astype(float).round(1)

    # FDR columns
    fdr_text_cols = [f"Next{i}_FDR" for i in range(1, 6)]
    fdr_num_cols = [f"Next{i}_FDR_num" for i in range(1, 6)]
    for c in fdr_text_cols + fdr_num_cols:
        df[c] = None

    # kickoff times
    fixtures_df["kickoff_dt"] = fixtures_df["kickoff_time"].apply(parse_kickoff)
    now_utc = datetime.now(timezone.utc)

    # current GW
    current_gw = None
    if not events.empty:
        cur = events[events["is_current"] == True]
        if not cur.empty:
            current_gw = int(cur["id"].iloc[0])
        else:
            nxt = events[events["is_next"] == True]
            if not nxt.empty:
                current_gw = int(nxt["id"].iloc[0])

    # upcoming fixtures
    fix = fixtures_df[fixtures_df["event"].notna()].copy()
    fix["is_upcoming"] = (~fix.get("finished", False).fillna(False)) | (fix["kickoff_dt"].notna() & (fix["kickoff_dt"] >= now_utc))
    if current_gw is not None:
        fix = fix[(fix["event"] >= current_gw) & (fix["is_upcoming"])]
    else:
        fix = fix[fix["is_upcoming"]]
    fix = fix.sort_values(["event", "kickoff_dt"], na_position="last")

    # team_next mapping
    team_next = defaultdict(list)
    for _, row in fix.iterrows():
        th = row.get("team_h")
        ta = row.get("team_a")
        dh = int(row.get("team_h_difficulty")) if pd.notna(row.get("team_h_difficulty")) else None
        da = int(row.get("team_a_difficulty")) if pd.notna(row.get("team_a_difficulty")) else None
        if pd.notna(th) and pd.notna(ta):
            team_next[th].append((team_map.get(ta, ""), dh))
            team_next[ta].append((team_map.get(th, ""), da))

    # Fill Next1..Next5
    for idx, r in df.iterrows():
        seq = team_next.get(r["team"], [])
        for i in range(5):
            opp, fdr = (seq[i] if i < len(seq) else ("", None))
            df.at[idx, f"Next{i+1}_FDR"] = f"{opp} ({fdr})" if opp and fdr is not None else ""
            df.at[idx, f"Next{i+1}_FDR_num"] = fdr if fdr is not None else None

    return df, fdr_text_cols, fdr_num_cols

# ----------------------------
# Login
# ----------------------------
if "username" not in st.session_state:
    st.title("⚽ FPL Winning Tool — Login")
    st.write("Enter a username to load/save your squad. Each username saves to its own file on this machine.")
    username_input = st.text_input("Username:")
    if st.button("Login") and username_input.strip() != "":
        st.session_state.username = username_input.strip()
        ensure_user_squad_file(st.session_state.username)
        st.rerun()
    st.stop()

username = st.session_state.username
st.markdown(f"**Logged in as:** {username} — your squad will be saved to `my_squad_{username}.json`")

# ----------------------------
# Load data
# ----------------------------
df, fdr_text_cols, fdr_num_cols = load_data()

# ----------------------------
# Tabs
# ----------------------------
tabs = st.tabs([
    "🧤 Goalkeepers", "🛡️ Defenders", "🎯 Midfielders", "⚡ Forwards",
    "📊 GW Insights", "📝 My Squad"
])
pos_order = ["Goalkeepers", "Defenders", "Midfielders", "Forwards"]

# ----------------------------
# Positional Tabs
# ----------------------------
for tab, pos in zip(tabs[:4], pos_order):
    with tab:
        st.subheader(pos)
        df_pos = df[df["position"] == pos].copy()
        show_cols = ["team_name", "name", "cost", "total_points", "minutes",
                     "selected_by_percent", "form", "points_per_game", "points_per_90"] + fdr_text_cols
        show_cols = [c for c in show_cols if c in df_pos.columns]
        styled = (
            df_pos[show_cols]
            .style.applymap(fdr_color_from_text, subset=fdr_text_cols)
            .format({
                "cost": "{:.1f}",
                "selected_by_percent": "{:.1f}",
                "form": "{:.1f}",
                "points_per_game": "{:.1f}",
                "points_per_90": "{:.1f}"
            })
        )
        st.dataframe(styled, use_container_width=True)

# ----------------------------
# My Squad Tab
# ----------------------------
with tabs[5]:
    st.subheader("📝 Select & Save Your Squad")
    pos_slots = {"Goalkeepers": 2, "Defenders": 5, "Midfielders": 5, "Forwards": 3}
    squad_file = f"my_squad_{username}.json"
    my_squad = {}
    if os.path.exists(squad_file):
        try:
            with open(squad_file, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    my_squad = loaded
        except Exception:
            my_squad = {}
    for pos, slots in pos_slots.items():
        if pos not in my_squad or not isinstance(my_squad[pos], list) or len(my_squad[pos]) != slots:
            my_squad[pos] = [""] * slots

    st.write("Choose players for each slot (use drop-downs). Click **Save Squad** when done.")
    new_squad = {}
    for pos in pos_order:
        st.markdown(f"**{pos}**")
        options = [""] + df[df["position"] == pos]["name"].tolist()
        slots = pos_slots[pos]
        row_cols = st.columns(slots)
        new_squad[pos] = []
        for i in range(slots):
            default_name = my_squad.get(pos, [""] * slots)[i]
            try:
                default_index = options.index(default_name) if default_name in options else 0
            except Exception:
                default_index = 0
            sel = row_cols[i].selectbox(f"{pos[:-1]} {i+1}", options, index=default_index, key=f"{pos}_{i}_{username}")
            new_squad[pos].append(sel)
    if st.button("Save Squad"):
        with open(squad_file, "w") as f:
            json.dump(new_squad, f)
        st.success(f"Squad saved to `{squad_file}`")
        my_squad = new_squad

# ----------------------------
# Flatten current players
# ----------------------------
current_players = [p for pos in pos_order for p in (my_squad.get(pos, []) if isinstance(my_squad.get(pos, []), list) else []) if p]

# ----------------------------
# GW Insights
# ----------------------------
with tabs[4]:
    st.subheader("📊 Gameweek Insights")
    work = df.copy()

    # remove element_type column ONLY here
    if "element_type" in work.columns:
        work.drop(columns=["element_type"], inplace=True)

    # --- rest of your GW Insights code stays exactly the same ---


# ----------------------------
# GW Insights tab (element_type removed)
# ----------------------------
with tabs[4]:
    st.subheader("📊 Gameweek Insights")

    def avg_next3(row):
        vals = [row[c] for c in fdr_num_cols[:3] if c in row.index and row[c] is not None]
        return sum(vals)/len(vals) if vals else None

    work = df.copy()
    work["avg_next3_fdr"] = work.apply(avg_next3, axis=1)

    # Suggested Transfers
    candidates = work[~work["name"].isin(current_players)].copy()
    candidates["transfer_score"] = candidates["points_per_game"]*0.55 + candidates["form"]*0.45 - candidates["avg_next3_fdr"].fillna(3.0)*0.35
    best_transfers = candidates.sort_values("transfer_score", ascending=False).head(12)
    best_transfers.insert(0, "Rank", range(1, len(best_transfers)+1))
    st.markdown("### 🔥 Best Transfers In (ranked)")
    st.dataframe(best_transfers.drop(columns=["element_type"]), use_container_width=True)

    # Captaincy pick
    top_captains = work.sort_values("points_per_game", ascending=False).head(10)
    st.markdown("### 👑 Captaincy Picks")
    st.dataframe(top_captains.drop(columns=["element_type"])[["name","team_name","points_per_game","form"]], use_container_width=True)

    # Top Selected
    top_selected = work.sort_values("selected_by_percent", ascending=False).head(10)
    st.markdown("### 📈 Top Selected Players")
    st.dataframe(top_selected.drop(columns=["element_type"])[["name","team_name","selected_by_percent"]], use_container_width=True)

    # Differential picks
    diff_players = work[work["selected_by_percent"] < 5].sort_values("points_per_game", ascending=False).head(10)
    st.markdown("### ⚡ Differential Players (<5% selected)")
    st.dataframe(diff_players.drop(columns=["element_type"])[["name","team_name","points_per_game","selected_by_percent"]], use_container_width=True)

    # Budget Enablers
    budget_enablers = work[work["cost"] <= 6.0].sort_values("points_per_game", ascending=False).head(10)
    st.markdown("### 💰 Budget Enablers (≤6.0)")
    st.dataframe(budget_enablers.drop(columns=["element_type"])[["name","team_name","cost","points_per_game"]], use_container_width=True)
