import pandas as pd
import requests
import streamlit as st
import json
import os
import re

# ----------------------------
# Load FPL Data
# ----------------------------
@st.cache_data
def load_data():
    # Bootstrap data
    base = "https://fantasy.premierleague.com/api"
    data = requests.get(f"{base}/bootstrap-static/").json()
    elements = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])
    element_types = pd.DataFrame(data["element_types"])

    # Fixtures
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

    # Keep only fixtures with a GW number, sorted by GW
    fix = fixtures_df[fixtures_df["event"].notna()].copy()
    fix = fix.sort_values(["event", "kickoff_time"], na_position="last")

    # For each team, collect next N opponents with correct difficulty
    from collections import defaultdict
    team_next = defaultdict(list)  # team_id -> list of (opp_name, fdr_int)

    for _, row in fix.iterrows():
        th, ta = row["team_h"], row["team_a"]
        dh, da = row["team_h_difficulty"], row["team_a_difficulty"]

        team_next[th].append((team_map.get(ta, ""), int(dh) if pd.notna(dh) else None))
        team_next[ta].append((team_map.get(th, ""), int(da) if pd.notna(da) else None))

    # Fill each player's next 5 based on their team
    for idx, r in df.iterrows():
        seq = team_next.get(r["team"], [])
        for i in range(5):
            opp, fdr = (seq[i] if i < len(seq) else ("", None))
            df.at[idx, f"Next{i+1}_FDR"] = f"{opp} ({fdr})" if opp and fdr is not None else ""
            df.at[idx, f"Next{i+1}_FDR_num"] = fdr if fdr is not None else None

    return df, fdr_text_cols, fdr_num_cols

# ----------------------------
# Styling helpers
# ----------------------------
def fdr_color_from_text(val: str):
    """Colour cells based on '(FDR)' at the end of text like 'Opp (3)'. Empty returns no style."""
    return ""  # Disabled color styling, keep clean

# ----------------------------
# App
# ----------------------------
st.set_page_config(page_title="FPL Dashboard", layout="wide")
st.title("⚽ FPL Winning Tool")

# ----------------------------
# User login / username
# ----------------------------
username = st.text_input("Enter your username to save/load your squad:", "")
if username:
    squad_file = f"my_squad_{username}.json"
else:
    st.warning("Please enter a username to enable saving your squad.")
    squad_file = None

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
current_players = [p for pos in pos_order for p in (my_squad.get(pos, []) if isinstance(my_squad.get(pos, []), list) else []) if p]

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
        .loc[:, ["name", "team_name", "position", "cost", "form", "points_per_game", "selected_by_percent"]]
        .head(12)
    )

    st.markdown("### 🔥 Best Transfers In (next 3 GWs weighted by form & PPG)")
    st.dataframe(
        best_transfers.style.format({
            "cost": "{:.1f}",
            "form": "{:.1f}",
            "points_per_game": "{:.1f}",
            "selected_by_percent": "{:.1f}",
        }),
        use_container_width=True
    )

    # ---------- Captaincy ----------
    work["captain_score"] = (
        work["points_per_game"] * 0.5
        + work["form"] * 0.5
        - work["avg_next3_fdr"].fillna(3.0) * 0.3
    )

    # Top 3 overall
    top_overall = work.sort_values("captain_score", ascending=False).head(3)[
        ["name", "team_name", "position", "form", "points_per_game", "selected_by_percent"]
    ]

    # Top 3 from current squad
    in_squad = work[work["name"].isin(current_players)].copy()
    top_squad = in_squad.sort_values("captain_score", ascending=False).head(3)[
        ["name", "team_name", "position", "form", "points_per_game", "selected_by_percent"]
    ] if not in_squad.empty else pd.DataFrame()

    # Top 3 most selected
    top_selected = work.sort_values("selected_by_percent", ascending=False).head(3)[
        ["name", "team_name", "position", "form", "points_per_game", "selected_by_percent"]
    ]

    st.markdown("### 🏆 Captaincy Recommendations")
    st.write("**Top 3 Overall**")
    st.dataframe(top_overall.style.format({
        "form": "{:.1f}", "points_per_game": "{:.1f}", "selected_by_percent": "{:.1f}"
    }), use_container_width=True)

    st.write("**Top 3 From Your Squad**")
    if not top_squad.empty:
        st.dataframe(top_squad.style.format({
            "form": "{:.1f}", "points_per_game": "{:.1f}", "selected_by_percent": "{:.1f}"
        }), use_container_width=True)
    else:
        st.info("No players in your squad yet.")

    st.write("**Top 3 Most Selected by Managers**")
    st.dataframe(top_selected.style.format({
        "form": "{:.1f}", "points_per_game": "{:.1f}", "selected_by_percent": "{:.1f}"
    }), use_container_width=True)

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
    diffs = diffs.sort_values("diff_score", ascending=False).loc[
        :, ["name", "team_name", "position", "selected_by_percent", "form", "points_per_game"]
    ].head(10)

    st.markdown("### 🎯 Differential Picks (≤15% selected, good form/PPG, kind fixtures)")
    st.dataframe(
        diffs.style.format({
            "selected_by_percent": "{:.1f}",
            "form": "{:.1f}",
            "points_per_game": "{:.1f}",
        }),
        use_container_width=True
    )

    # ---------- Budget Enablers ----------
    budget = work[(~work["name"].isin(current_players)) & (work["cost"] <= 5.0)].copy()
    budget["budget_score"] = budget["points_per_game"] * 0.6 + budget["form"] * 0.4
    budget = budget.sort_values("budget_score", ascending=False).loc[
        :, ["name", "team_name", "position", "cost", "points_per_game", "form"]
    ].head(12)

    st.markdown("### 💰 Budget Enablers (≤ £5.0)")
    st.dataframe(
        budget.style.format({
            "cost": "{:.1f}",
            "points_per_game": "{:.1f}",
            "form": "{:.1f}",
        }),
        use_container_width=True
    )
