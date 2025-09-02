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

# ----------------------------
# FDR coloring helper (text black)
# ----------------------------
def fdr_color_from_text(val):
    if val is None or val == "":
        return ""
    # simple grading based on FDR number in parentheses
    import re
    m = re.search(r"\((\d)\)", str(val))
    if m:
        fdr = int(m.group(1))
        if fdr == 1:
            color = "#b6d7a8"
        elif fdr == 2:
            color = "#fff2cc"
        elif fdr == 3:
            color = "#f9cb9c"
        elif fdr == 4:
            color = "#e06666"
        else:
            color = "#990000"
        return f"background-color: {color}; color: black"
    return ""

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
# Load FPL data
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
        ["id", "first_name", "second_name", "web_name", "team", "element_type",
         "now_cost", "total_points", "minutes", "selected_by_percent",
         "form", "points_per_game"]
    ].copy()

    df["name"] = (df["first_name"].fillna("") + " " + df["second_name"].fillna("")).str.strip()
    df["team_name"] = df["team"].map(team_map)
    df["position"] = df["element_type"].map(pos_name_map)
    df["cost"] = (df["now_cost"] / 10).round(1)
    df["points_per_90"] = df.apply(lambda r: (r["total_points"] / (r["minutes"] / 90)) if r["minutes"] > 0 else 0.0, axis=1)
    df["total_points"] = df["total_points"].astype(int)
    df["minutes"] = df["minutes"].astype(int)
    df["selected_by_percent"] = df["selected_by_percent"].astype(float).round(1)
    df["form"] = df["form"].astype(float).round(1)
    df["points_per_game"] = df["points_per_game"].astype(float).round(1)
    df["points_per_90"] = df["points_per_90"].astype(float).round(1)

    fdr_text_cols = [f"Next{i}_FDR" for i in range(1, 6)]
    fdr_num_cols = [f"Next{i}_FDR_num" for i in range(1, 6)]
    for c in fdr_text_cols + fdr_num_cols:
        df[c] = None

    # Parse kickoff times
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

    # Filter upcoming fixtures
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
        th = row.get("team_h")
        ta = row.get("team_a")
        dh = int(row.get("team_h_difficulty")) if pd.notna(row.get("team_h_difficulty")) else None
        da = int(row.get("team_a_difficulty")) if pd.notna(row.get("team_a_difficulty")) else None
        if pd.notna(th) and pd.notna(ta):
            team_next[th].append((team_map.get(ta, ""), dh))
            team_next[ta].append((team_map.get(th, ""), da))

    # Fallback for teams with no upcoming fixtures
    for team_id in teams["id"].tolist():
        if team_id not in team_next or len(team_next[team_id]) == 0:
            team_future = fixtures_df[
                ((fixtures_df["team_h"] == team_id) | (fixtures_df["team_a"] == team_id)) &
                (fixtures_df["kickoff_dt"].notna()) & (fixtures_df["kickoff_dt"] >= now_utc)
            ].sort_values(["event", "kickoff_dt"])
            for _, row in team_future.iterrows():
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
# Load data
# ----------------------------
df, fdr_text_cols, fdr_num_cols = load_data()

# ----------------------------
# Compute avg_next3_fdr to prevent KeyError
# ----------------------------
def avg_next3(series_row):
    vals = []
    for c in fdr_num_cols[:3]:
        if c in series_row.index:
            v = series_row[c]
            if v is not None:
                vals.append(v)
    return sum(vals) / len(vals) if vals else None

df["avg_next3_fdr"] = df.apply(avg_next3, axis=1)

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
st.markdown(f"**Logged in as:** {username}")

# ----------------------------
# Load data
# ----------------------------
df, fdr_text_cols, fdr_num_cols = load_data()

# ----------------------------
# Tabs
# ----------------------------
tabs = st.tabs(["🧤 Goalkeepers", "🛡️ Defenders", "🎯 Midfielders", "⚡ Forwards",
                "📊 GW Insights", "📝 My Squad"])
pos_order = ["Goalkeepers", "Defenders", "Midfielders", "Forwards"]

# ----------------------------
# Positional tabs
# ----------------------------
for tab, pos in zip(tabs[:4], pos_order):
    with tab:
        st.subheader(pos)
        df_pos = df[df["position"] == pos].copy()
        show_cols = ["team_name", "name", "cost", "total_points", "minutes",
                     "selected_by_percent", "form", "points_per_game", "points_per_90"] + fdr_text_cols
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

current_players = [p for pos in pos_order for p in (my_squad.get(pos, []) if isinstance(my_squad.get(pos, []), list) else []) if p]

# ----------------------------
# GW Insights tab
# ----------------------------
with tabs[4]:
    st.subheader("📊 Gameweek Insights")
    work = df.copy()
    # remove element_type from table
    def avg_next3(series_row):
        vals = []
        for c in fdr_num_cols[:3]:
            v = series_row.get(c, None)
            if v is not None:
                vals.append(v)
        return sum(vals) / len(vals) if vals else None
    work["Average Next 3 Difficulty Rating"] = work.apply(avg_next3, axis=1)

    # Suggested Transfers
# Ensure avg_next3_fdr exists
if "avg_next3_fdr" not in candidates.columns:
    candidates["avg_next3_fdr"] = candidates.apply(avg_next3, axis=1)

candidates["transfer_score"] = (
    candidates["points_per_game"] * 0.55
    + candidates["form"] * 0.45
    - candidates["avg_next3_fdr"].fillna(3.0) * 0.35
)

    )
    best_transfers = candidates.sort_values("transfer_score", ascending=False).loc[:, [
        "name", "team_name", "position", "cost", "form", "points_per_game", "selected_by_percent", "avg_next3_fdr"
    ]].head(12).reset_index(drop=True)
    best_transfers.insert(0, "Rank", best_transfers.index + 1)

    st.markdown("### 🔥 Best Transfers In (ranked)")
    st.caption("Higher points_per_game & form, and kinder next-3 FDR => higher rank")
    existing_fdr_cols = [c for c in fdr_text_cols if c in best_transfers.columns]
    st.dataframe(best_transfers.style.applymap(fdr_color_from_text, subset=existing_fdr_cols).format({
        "cost": "{:.1f}", "form": "{:.1f}", "points_per_game": "{:.1f}", "selected_by_percent": "{:.1f}", "avg_next3_fdr": "{:.1f}"
    }), use_container_width=True)

    # Captaincy
    work["captain_score"] = work["points_per_game"]*0.5 + work["form"]*0.5 - work["avg_next3_fdr"].fillna(3.0)*0.3
    top_overall = work.sort_values("captain_score", ascending=False).loc[:, [
        "name", "team_name", "position", "form", "points_per_game", "selected_by_percent", "avg_next3_fdr", "captain_score"
    ]].head(10).reset_index(drop=True)
    top_overall.insert(0, "Rank", top_overall.index + 1)
    st.markdown("### 🏆 Captaincy — Top Overall (Top 10)")
    st.dataframe(top_overall.style.format({
        "form": "{:.1f}", "points_per_game": "{:.1f}", "selected_by_percent": "{:.1f}", "avg_next3_fdr": "{:.1f}", "captain_score": "{:.2f}"
    }), use_container_width=True)

    in_squad = work[work["name"].isin(current_players)].copy()
    if not in_squad.empty:
        top_squad = in_squad.sort_values("captain_score", ascending=False).loc[:, [
            "name", "team_name", "position", "form", "points_per_game", "selected_by_percent", "avg_next3_fdr", "captain_score"
        ]].head(10).reset_index(drop=True)
        top_squad.insert(0, "Rank", top_squad.index + 1)
        st.markdown("### 🧢 Captaincy — From Your Squad (ranked)")
        st.dataframe(top_squad.style.format({
            "form": "{:.1f}", "points_per_game": "{:.1f}", "selected_by_percent": "{:.1f}", "avg_next3_fdr": "{:.1f}", "captain_score": "{:.2f}"
        }), use_container_width=True)
    else:
        st.info("No players selected in your squad yet — add players in the 'My Squad' tab to see squad-specific captain picks.")

    # Top selected
    top_selected = work.sort_values("selected_by_percent", ascending=False).head(3).loc[:, [
        "name", "team_name", "position", "form", "points_per_game", "selected_by_percent"
    ]].reset_index(drop=True)
    top_selected.insert(0, "Rank", top_selected.index + 1)
    st.markdown("### 👥 Top Selected by Managers (Top 3)")
    st.dataframe(top_selected.style.format({
        "form": "{:.1f}", "points_per_game": "{:.1f}", "selected_by_percent": "{:.1f}"
    }), use_container_width=True)

    # Differentials
    diffs = work[~work["name"].isin(current_players)].copy()
    diffs = diffs[(diffs["selected_by_percent"]<=15.0) & ((diffs["points_per_game"]>=3.0)|(diffs["form"]>=4.0)) & (diffs["total_points"]>0)]
    diffs = diffs[diffs["avg_next3_fdr"].fillna(3.0) <= 3.0]
    diffs["diff_score"] = diffs["points_per_game"]*0.6 + diffs["form"]*0.4 - diffs["avg_next3_fdr"].fillna(3.0)*0.5
    diffs = diffs.sort_values("diff_score", ascending=False).loc[:, [
        "name","team_name","position","selected_by_percent","form","points_per_game","avg_next3_fdr","diff_score"
    ]].head(10).reset_index(drop=True)
    diffs.insert(0, "Rank", diffs.index + 1)
    st.markdown("### 🌟 Differential Picks — High Upside Low Ownership (Top 10)")
    st.dataframe(diffs.style.format({
        "selected_by_percent": "{:.1f}", "form": "{:.1f}", "points_per_game": "{:.1f}", "avg_next3_fdr": "{:.1f}", "diff_score": "{:.2f}"
    }), use_container_width=True)
