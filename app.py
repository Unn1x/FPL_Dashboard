import pandas as pd
import requests
import streamlit as st
import json
import os
import re
from datetime import datetime, timezone

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="FPL Dashboard", layout="wide")
st.title("⚽ FPL Winning Tool")

# ----------------------------
# Helpers
# ----------------------------
def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

# Color styling disabled (kept as hook)
def fdr_color_from_text(_val: str):
    return ""

# ----------------------------
# Load FPL Data (cached)
# ----------------------------
@st.cache_data(ttl=300, show_spinner=False)  # cache for 5 minutes
def load_data():
    base = "https://fantasy.premierleague.com/api"

    # Bootstrap
    data = requests.get(f"{base}/bootstrap-static/").json()
    elements = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])
    # element_types = pd.DataFrame(data["element_types"])  # not used directly

    # Fixtures (all season)
    fixtures = requests.get(f"{base}/fixtures/").json()
    fixtures_df = pd.DataFrame(fixtures)

    # Maps
    team_map = teams.set_index("id")["name"].to_dict()
    pos_name_map = {
        1: "Goalkeepers",
        2: "Defenders",
        3: "Midfielders",
        4: "Forwards",
    }

    # Base player DF
    df = elements[
        [
            "id", "first_name", "second_name", "web_name", "team", "element_type",
            "now_cost", "total_points", "minutes", "selected_by_percent",
            "form", "points_per_game"
        ]
    ].copy()

    # Derived fields
    df["name"] = (df["first_name"] + " " + df["second_name"]).str.strip()
    df["team_name"] = df["team"].map(team_map)
    df["position"] = df["element_type"].map(pos_name_map)
    df["cost"] = (df["now_cost"] / 10).round(1)

    # P90 (avoid div by zero)
    df["points_per_90"] = df.apply(
        lambda r: (r["total_points"] / (r["minutes"] / 90)) if r["minutes"] > 0 else 0.0,
        axis=1
    )

    # Round numeric presentation
    df["total_points"] = df["total_points"].astype(int)
    df["minutes"] = df["minutes"].astype(int)
    df["selected_by_percent"] = df["selected_by_percent"].astype(float).round(1)
    df["form"] = df["form"].astype(float).round(1)
    df["points_per_game"] = df["points_per_game"].astype(float).round(1)
    df["points_per_90"] = df["points_per_90"].astype(float).round(1)

    # ---- Build next 5 fixtures (opponent text + numeric FDR) per team ----
    fdr_text_cols = [f"Next{i}_FDR" for i in range(1, 6)]
    fdr_num_cols = [f"Next{i}_FDR_num" for i in range(1, 6)]
    for c in fdr_text_cols + fdr_num_cols:
        df[c] = None

    # Normalize datetime and keep only *upcoming* fixtures.
    # Use 'finished' flag primarily; fallback to kickoff_time >= now for safety.
    fix = fixtures_df.copy()

    # Parse kickoff_time to aware UTC datetime (some rows can be None)
    def parse_kt(x):
        if pd.isna(x) or not isinstance(x, str):
            return None
        try:
            # FPL returns ISO strings in UTC (e.g. "2025-09-14T15:30:00Z")
            return datetime.fromisoformat(x.replace("Z", "+00:00"))
        except Exception:
            return None

    fix["kickoff_dt"] = fix["kickoff_time"].apply(parse_kt)
    now_utc = datetime.now(timezone.utc)

    # Keep only fixtures that belong to a GW and are not finished.
    # If 'finished' is missing, use kickoff time in the future.
    not_finished_mask = (~fix.get("finished", False).fillna(False)) | (fix["kickoff_dt"] >= now_utc)
    fix = fix[fix["event"].notna() & not_finished_mask].copy()

    # Sort strictly by kickoff time to guarantee correct next-up ordering
    fix = fix.sort_values(["kickoff_dt", "event"], na_position="last")

    # Build upcoming list per team with correct home/away difficulty
    from collections import defaultdict
    team_next = defaultdict(list)  # team_id -> list of (opp_name, fdr_int)

    for _, row in fix.iterrows():
        th, ta = row["team_h"], row["team_a"]
        dh = row.get("team_h_difficulty", None)
        da = row.get("team_a_difficulty", None)

        dh = int(dh) if pd.notna(dh) else None
        da = int(da) if pd.notna(da) else None

        # Home perspective
        team_next[th].append((team_map.get(ta, ""), dh))
        # Away perspective
        team_next[ta].append((team_map.get(th, ""), da))

    # Fill each player's next 5 based on their team
    for idx, r in df.iterrows():
        seq = team_next.get(r["team"], [])
        for i in range(5):
            opp, fdr = (seq[i] if i < len(seq) else ("", None))
            df.at[idx, f"Next{i+1}_FDR"] = f"{opp} ({fdr})" if opp and fdr is not None else ""
            df.at[idx, f"Next{i+1}_FDR_num"] = fdr if fdr is not None else None

    return df, fdr_text_cols, fdr_num_cols

# ----------------------------
# Username (per-user save)
# ----------------------------
with st.sidebar:
    st.markdown("### Settings")
    username = st.text_input("Enter your username (for your own saved squad):", "")
    if st.button("🔄 Refresh live data (clear cache)"):
        st.cache_data.clear()
        st.experimental_rerun()

if username:
    squad_file = f"my_squad_{username}.json"
else:
    squad_file = None
    st.warning("Enter a username (left sidebar) to enable saving your squad locally to this app.")

# ----------------------------
# Fetch data
# ----------------------------
df, fdr_text_cols, fdr_num_cols = load_data()

# ----------------------------
# Tabs: 4 positional + GW Insights + My Squad
# ----------------------------
tabs = st.tabs([
    "🧤 Goalkeepers", "🛡️ Defenders", "🎯 Midfielders", "⚡ Forwards",
    "📊 GW Insights", "📝 My Squad"
])

# ----------------------------
# Positional Tabs
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

        styled = (
            df_pos[show_cols]
            .style.applymap(fdr_color_from_text, subset=fdr_text_cols)
            .format({
                "cost": "{:.1f}",
                "selected_by_percent": "{:.1f}",
                "form": "{:.1f}",
                "points_per_game": "{:.1f}",
                "points_per_90": "{:.1f}",
            })
        )
        st.dataframe(styled, use_container_width=True)

# ----------------------------
# My Squad tab
# ----------------------------
with tabs[5]:
    st.subheader("📝 Select & Save Your Squad")

    pos_slots = {"Goalkeepers": 2, "Defenders": 5, "Midfielders": 5, "Forwards": 3}

    # Load saved squad safely
    my_squad = {}
    if squad_file and os.path.exists(squad_file):
        try:
            with open(squad_file, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    my_squad = loaded
        except Exception:
            my_squad = {}

    # Build UI + defaults
    new_squad = {}
    for pos in pos_order:
        st.markdown(f"**{pos}**")
        options = [""] + df[df["position"] == pos]["name"].tolist()
        slots = pos_slots[pos]
        saved = my_squad.get(pos, [""] * slots)
        row = st.columns(slots)
        new_squad[pos] = []
        for i in range(slots):
            default_idx = options.index(saved[i]) if i < len(saved) and saved[i] in options else 0
            sel = row[i].selectbox(f"{pos[:-1]} {i+1}", options, index=default_idx, key=f"{pos}_{i}")
            new_squad[pos].append(sel)

    # Persist
    my_squad = new_squad
    if squad_file:
        with open(squad_file, "w") as f:
            json.dump(my_squad, f)

    if username:
        st.success(f"Squad saved for user: {username}")

# Flatten current players for insights
current_players = [
    p for pos in pos_order
    for p in (my_squad.get(pos, []) if isinstance(my_squad.get(pos, []), list) else [])
    if p
]

# ----------------------------
# GW Insights tab
# ----------------------------
with tabs[4]:
    st.subheader("📊 Gameweek Insights")

    def avg_next3(series_row):
        vals = [series_row.get(c) for c in fdr_num_cols[:3] if c in series_row.index]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    work = df.copy()
    work["avg_next3_fdr"] = work.apply(avg_next3, axis=1)

    # ---------- Suggested Transfers ----------
    candidates = work[~work["name"].isin(current_players)].copy()
    candidates["transfer_score"] = (
        candidates["points_per_game"] * 0.55
        + candidates["form"] * 0.45
        - candidates["avg_next3_fdr"].fillna(3.0) * 0.35
    )
    best_transfers = (
        candidates.sort_values("transfer_score", ascending=False)
        .loc[:, ["name", "team_name", "position", "cost", "form", "points_per_game", "selected_by_percent", "avg_next3_fdr"]]
        .head(12)
        .reset_index(drop=True)
    )
    best_transfers.insert(0, "Rank", best_transfers.index + 1)

    st.markdown("### 🔥 Best Transfers In (ranked by Transfer Score)")
    st.caption("Higher points/form and kinder next 3 FDR → higher rank")
    st.dataframe(
        best_transfers.style.format({
            "cost": "{:.1f}",
            "form": "{:.1f}",
            "points_per_game": "{:.1f}",
            "selected_by_percent": "{:.1f}",
            "avg_next3_fdr": "{:.1f}",
        }),
        use_container_width=True
    )

    # ---------- Captaincy ----------
    work["captain_score"] = (
        work["points_per_game"] * 0.5
        + work["form"] * 0.5
        - work["avg_next3_fdr"].fillna(3.0) * 0.3
    )

    # Top 10 overall with ranks (show 10 so you can see ordering clearly)
    top_overall = (
        work.sort_values("captain_score", ascending=False)
        .loc[:, ["name", "team_name", "position", "form", "points_per_game", "selected_by_percent", "avg_next3_fdr", "captain_score"]]
        .head(10)
        .reset_index(drop=True)
    )
    top_overall.insert(0, "Rank", top_overall.index + 1)

    st.markdown("### 🏆 Captaincy — Overall (Top 10, ranked)")
    st.dataframe(
        top_overall.style.format({
            "form": "{:.1f}",
            "points_per_game": "{:.1f}",
            "selected_by_percent": "{:.1f}",
            "avg_next3_fdr": "{:.1f}",
            "captain_score": "{:.2f}",
        }),
        use_container_width=True
    )

    # Top from your squad (ranked)
    in_squad = work[work["name"].isin(current_players)].copy()
    if not in_squad.empty:
        top_squad = (
            in_squad.sort_values("captain_score", ascending=False)
            .loc[:, ["name", "team_name", "position", "form", "points_per_game", "selected_by_percent", "avg_next3_fdr", "captain_score"]]
            .head(10)
            .reset_index(drop=True)
        )
        top_squad.insert(0, "Rank", top_squad.index + 1)
        st.markdown("### 🧢 Captaincy — From Your Squad (ranked)")
        st.dataframe(
            top_squad.style.format({
                "form": "{:.1f}",
                "points_per_game": "{:.1f}",
                "selected_by_percent": "{:.1f}",
                "avg_next3_fdr": "{:.1f}",
                "captain_score": "{:.2f}",
            }),
            use_container_width=True
        )
    else:
        st.info("No players selected in your squad yet.")

    # ---------- Differentials ----------
    diffs = work[~work["name"].isin(current_players)].copy()
    diffs = diffs[
        (diffs["selected_by_percent"] <= 15.0) &
        ((diffs["points_per_game"] >= 3.0) | (diffs["form"] >= 4.0)) &
        (diffs["total_points"] > 0)
    ].copy()
    diffs = diffs[diffs["avg_next3_fdr"].fillna(3.0) <= 3.0]
    diffs["diff_score"] = (
        diffs["points_per_game"] * 0.6
        + diffs["form"] * 0.4
        - diffs["avg_next3_fdr"].fillna(3.0) * 0.5
    )
    diffs = (
        diffs.sort_values("diff_score", ascending=False)
        .loc[:, ["name", "team_name", "position", "selected_by_percent", "form", "points_per_game", "avg_next3_fdr", "diff_score"]]
        .head(10)
        .reset_index(drop=True)
    )
    diffs.insert(0, "Rank", diffs.index + 1)

    st.markdown("### 🎯 Differentials (≤15% selected, good form/PPG, kind fixtures)")
    st.dataframe(
        diffs.style.format({
            "selected_by_percent": "{:.1f}",
            "form": "{:.1f}",
            "points_per_game": "{:.1f}",
            "avg_next3_fdr": "{:.1f}",
            "diff_score": "{:.2f}",
        }),
        use_container_width=True
    )

    # ---------- Budget Enablers ----------
    budget = work[(~work["name"].isin(current_players)) & (work["cost"] <= 5.0)].copy()
    budget["budget_score"] = budget["points_per_game"] * 0.6 + budget["form"] * 0.4
    budget = (
        budget.sort_values("budget_score", ascending=False)
        .loc[:, ["name", "team_name", "position", "cost", "points_per_game", "form", "avg_next3_fdr", "budget_score"]]
        .head(12)
        .reset_index(drop=True)
    )
    budget.insert(0, "Rank", budget.index + 1)

    st.markdown("### 💰 Budget Enablers (≤ £5.0)")
    st.dataframe(
        budget.style.format({
            "cost": "{:.1f}",
            "points_per_game": "{:.1f}",
            "form": "{:.1f}",
            "avg_next3_fdr": "{:.1f}",
            "budget_score": "{:.2f}",
        }),
        use_container_width=True
    )
