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

def ensure_user_squad_file(username):
    fn = f"my_squad_{username}.json"
    if not os.path.exists(fn):
        pos_slots = {"Goalkeepers": 2, "Defenders": 5, "Midfielders": 5, "Forwards": 3}
        base = {pos: [""] * n for pos, n in pos_slots.items()}
        with open(fn, "w") as f:
            json.dump(base, f)
    return fn

def fdr_color_from_text(val):
    """Return background color for FDR difficulty (1-5)."""
    try:
        val_int = int(val) if val is not None and val != "" else None
    except:
        val_int = None
    if val_int == 1:
        color = "#a6f1a6"  # green
    elif val_int == 2:
        color = "#ccf0a6"  # light green
    elif val_int == 3:
        color = "#ffff99"  # yellow
    elif val_int == 4:
        color = "#ffb366"  # orange
    elif val_int == 5:
        color = "#ff6666"  # red
    else:
        color = ""  # default
    return f"background-color: {color}"

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

    # fixtures
    fixtures_list = requests.get(f"{base}/fixtures/").json()
    fixtures_df = pd.DataFrame(fixtures_list)

    # maps
    team_map = teams.set_index("id")["name"].to_dict()
    pos_name_map = {1: "Goalkeepers", 2: "Defenders", 3: "Midfielders", 4: "Forwards"}

    # Basic player df
    df = elements[
        [
            "id", "first_name", "second_name", "web_name", "team", "element_type",
            "now_cost", "total_points", "minutes", "selected_by_percent",
            "form", "points_per_game"
        ]
    ].copy()

    # Derived fields
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

    # Prepare FDR columns
    fdr_text_cols = [f"Next{i}_FDR" for i in range(1, 6)]
    fdr_num_cols = [f"Next{i}_FDR_num" for i in range(1, 6)]
    for c in fdr_text_cols + fdr_num_cols:
        df[c] = None

    # Parse kickoff
    fixtures_df["kickoff_dt"] = fixtures_df["kickoff_time"].apply(parse_kickoff)
    now_utc = datetime.now(timezone.utc)

    # Determine current GW
    current_gw = None
    try:
        if not events.empty:
            cur = events[events["is_current"] == True]
            if not cur.empty:
                current_gw = int(cur["id"].iloc[0])
            else:
                nxt = events[events["is_next"] == True]
                if not nxt.empty:
                    current_gw = int(nxt["id"].iloc[0])
    except Exception:
        current_gw = None

    # Build candidate fixtures
    fix = fixtures_df[fixtures_df["event"].notna()].copy()
    fix["is_upcoming"] = (~fix.get("finished", False).fillna(False)) | (fix["kickoff_dt"].notna() & (fix["kickoff_dt"] >= now_utc))
    if current_gw is not None:
        fix = fix[(fix["event"] >= current_gw) & (fix["is_upcoming"])]
    else:
        fix = fix[fix["is_upcoming"]]

    fix = fix.sort_values(["event", "kickoff_dt"], na_position="last")

    # Build team_next dict
    team_next = defaultdict(list)
    for _, row in fix.iterrows():
        th, ta = row.get("team_h"), row.get("team_a")
        dh, da = row.get("team_h_difficulty"), row.get("team_a_difficulty")
        dh = int(dh) if pd.notna(dh) else None
        da = int(da) if pd.notna(da) else None
        if pd.notna(th) and pd.notna(ta):
            team_next[th].append((team_map.get(ta, ""), dh))
            team_next[ta].append((team_map.get(th, ""), da))

    # Fallback for missing teams
    for team_id in teams["id"].tolist():
        if team_id not in team_next or len(team_next[team_id]) == 0:
            team_future = fixtures_df[
                ((fixtures_df["team_h"] == team_id) | (fixtures_df["team_a"] == team_id)) &
                (fixtures_df["kickoff_dt"].notna()) & (fixtures_df["kickoff_dt"] >= now_utc)
            ].sort_values(["event", "kickoff_dt"])
            for _, row in team_future.iterrows():
                th, ta = row.get("team_h"), row.get("team_a")
                dh, da = row.get("team_h_difficulty"), row.get("team_a_difficulty")
                dh = int(dh) if pd.notna(dh) else None
                da = int(da) if pd.notna(da) else None
                team_next[th].append((team_map.get(ta, ""), dh))
                team_next[ta].append((team_map.get(th, ""), da))

    # Fill per-player Next1..Next5
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
# Tabs: Positional + GW Insights + My Squad
# ----------------------------
tabs = st.tabs([
    "🧤 Goalkeepers", "🛡️ Defenders", "🎯 Midfielders", "⚡ Forwards",
    "📊 GW Insights", "📝 My Squad"
])

# ----------------------------
# Positional Tabs (with FDR colors)
# ----------------------------
pos_order = ["Goalkeepers", "Defenders", "Midfielders", "Forwards"]
for tab, pos in zip(tabs[:4], pos_order):
    with tab:
        st.subheader(pos)
        df_pos = df[df["position"] == pos].copy()
        show_cols = (
            ["team_name", "name", "cost", "total_points", "minutes",
             "selected_by_percent", "form", "points_per_game", "points_per_90"]
            + fdr_text_cols
        )
        show_cols = [c for c in show_cols if c in df_pos.columns]
        styled = df_pos[show_cols].style.applymap(fdr_color_from_text, subset=fdr_text_cols).format({
            "cost": "{:.1f}",
            "selected_by_percent": "{:.1f}",
            "form": "{:.1f}",
            "points_per_game": "{:.1f}",
            "points_per_90": "{:.1f}"
        })
        st.dataframe(styled, use_container_width=True)

# ----------------------------
# My Squad tab
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
        except:
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
            default_index = options.index(default_name) if default_name in options else 0
            sel = row_cols[i].selectbox(f"{pos[:-1]} {i+1}", options, index=default_index, key=f"{pos}_{i}_{username}")
            new_squad[pos].append(sel)

    if st.button("Save Squad"):
        with open(squad_file, "w") as f:
            json.dump(new_squad, f)
        st.success(f"Squad saved to `{squad_file}`")
        my_squad = new_squad

current_players = [p for pos in pos_order for p in my_squad.get(pos, []) if p]

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
