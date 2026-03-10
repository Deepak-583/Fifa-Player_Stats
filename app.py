"""
FIFA Player Stats - Spider Web (Radar) Chart Comparator
Compare strengths and weaknesses of football players using radar charts and ML predictions.
"""

import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from pathlib import Path
from models import train_models, predict_value, predict_potential, get_similar_players

# ============ CONFIG ============
_BASE = Path(__file__).resolve().parent
# Try project folder first (works on Streamlit Cloud), then local paths
DATA_PATHS = [
    _BASE / "updated_trending_football_players.xlsx",
    _BASE / "trending_football_players.xlsx",
]
if sys.platform == "win32":
    DATA_PATHS.extend([
        Path(r"c:\Users\CHEYAT COMPUTERS\Downloads\archive\updated_trending_football_players.xlsx"),
        Path(r"c:\Users\CHEYAT COMPUTERS\Downloads\archive\trending_football_players.xlsx"),
        Path.home() / "Downloads" / "archive" / "updated_trending_football_players.xlsx",
        Path.home() / "Downloads" / "archive" / "trending_football_players.xlsx",
    ])

STAT_CATEGORIES = ["Speed", "Power", "Passing", "Defense", "Shooting"]


def load_data() -> pd.DataFrame:
    """Load FIFA players data from Excel files."""
    for path in DATA_PATHS:
        if path.exists():
            try:
                df = pd.read_excel(path, sheet_name=0)
                return df
            except Exception as e:
                st.warning(f"Could not load {path.name}: {e}")
                continue
    st.error(
        "No data file found. Add `updated_trending_football_players.xlsx` or "
        "`trending_football_players.xlsx` to the project folder (or Downloads/archive locally)."
    )
    return pd.DataFrame()


def derive_player_stats(row: pd.Series) -> dict:
    """
    Derive Speed, Power, Passing, Defense, Shooting from available columns.
    Uses Overall, Total_stats, Positions, and Age for realistic distributions.
    """
    overall = float(row.get("Overall", 70))
    potential = float(row.get("Potential_overall", overall))
    total_stats = float(row.get("Total_stats", 1500))
    age = int(row.get("Age", 25))
    positions = str(row.get("Positions", "")).upper()

    # Base stats scale 0-99, centered around overall
    base = min(99, max(1, overall))
    # Use total_stats for variance (higher total = more balanced)
    variance = min(15, max(0, (total_stats - 1200) / 100)) if total_stats else 5
    # Age factor: peak physical 24-28
    age_factor = 1 - 0.02 * abs(age - 26) if 18 <= age <= 40 else 0.8

    # Position-based boosts (additive, then clamp)
    boosts = {"Speed": 0, "Power": 0, "Passing": 0, "Defense": 0, "Shooting": 0}

    if any(p in positions for p in ["CB", "RB", "LB", "LWB", "RWB", "SW"]):
        boosts["Defense"] += 8
        boosts["Power"] += 4
    if any(p in positions for p in ["CM", "CDM", "CAM", "LM", "RM"]):
        boosts["Passing"] += 8
        boosts["Speed"] += 2
    if any(p in positions for p in ["ST", "CF", "LW", "RW"]):
        boosts["Shooting"] += 8
        boosts["Speed"] += 4

    # Build stats with randomness seeded by player name for consistency
    seed = hash(str(row.get("Player_name", ""))) % 10000
    np.random.seed(seed)

    stats = {}
    for cat in STAT_CATEGORIES:
        noise = np.random.uniform(-variance, variance)
        val = base + boosts[cat] + noise
        if cat == "Power":
            val *= age_factor
        stats[cat] = int(min(99, max(1, round(val))))

    return stats


def create_radar_chart(players_data: list[dict]) -> go.Figure:
    """Create an overlapping radar chart for one or more players."""
    fig = go.Figure()

    colors = px.colors.qualitative.Set1
    for i, p in enumerate(players_data):
        name = p["name"]
        stats = p["stats"]
        theta = STAT_CATEGORIES + [STAT_CATEGORIES[0]]  # Close the shape
        r = [stats[s] for s in STAT_CATEGORIES] + [stats[STAT_CATEGORIES[0]]]
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatterpolar(
                r=r,
                theta=theta,
                name=name,
                line=dict(color=color, width=2.5),
                fill="toself",
                opacity=0.35,
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickvals=[20, 40, 60, 80, 100]),
            angularaxis=dict(tickfont=dict(size=14)),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        title=dict(text="Player Skills Radar Chart", font=dict(size=22)),
        margin=dict(t=80, b=60, l=80, r=80),
        height=500,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(245,245,245,0.8)",
    )
    return fig


def main():
    st.set_page_config(
        page_title="FIFA Player Stats | Radar Chart",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Custom styling
    st.markdown(
        """
        <style>
        .main { padding-top: 1rem; }
        h1 { color: #1a472a; font-weight: 700; }
        .stSelectbox > div { font-size: 1.1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("⚽ FIFA Player Stats – Spider Web Comparator")
    st.caption("Compare strengths and weaknesses of football players using radar charts")

    df = load_data()
    if df.empty:
        return

    # Ensure columns exist for model training (handle different Excel formats)
    if "Contract_start" not in df.columns and "Current_contract" in df.columns:
        df["Contract_start"] = None
    if "Potential_overall" not in df.columns:
        df["Potential_overall"] = df.get("Overall", 70)
    if "Value" not in df.columns:
        df["Value"] = 1_000_000
    if "Wage" not in df.columns:
        df["Wage"] = 5000

    # Train ML models (cached in session)
    if "ml_models" not in st.session_state:
        with st.spinner("Training models (Value, Potential, Similar Players)..."):
            try:
                st.session_state["ml_models"] = train_models(df, derive_player_stats)
            except Exception as e:
                st.warning(f"Model training skipped: {e}. Predictions will be unavailable.")
                st.session_state["ml_models"] = None

    ml_models = st.session_state.get("ml_models")

    # Build player stats cache
    if "player_stats_cache" not in st.session_state:
        cache = {}
        for _, row in df.iterrows():
            name = row.get("Player_name", "")
            if pd.notna(name) and str(name).strip():
                cache[str(name).strip()] = {
                    "row": row,
                    "stats": derive_player_stats(row),
                }
        st.session_state["player_stats_cache"] = cache

    cache = st.session_state["player_stats_cache"]
    player_names = sorted(cache.keys(), key=str.lower)

    # Top bar: Dropdowns
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        idx1 = player_names.index("L. Messi") if "L. Messi" in player_names else 0
        player1 = st.selectbox(
            "**Select Player 1**",
            options=player_names,
            index=idx1,
            key="player1",
        )
    with col2:
        player2 = st.selectbox(
            "**Select Player 2 (optional – overlap on chart)**",
            options=["— None —"] + player_names,
            index=0,
            key="player2",
        )
    with col3:
        st.write("")  # Spacer for layout

    # Build chart data
    players_data = [{"name": player1, "stats": cache[player1]["stats"]}]
    if player2 and player2 != "— None —":
        players_data.append({"name": player2, "stats": cache[player2]["stats"]})

    # Main: Radar chart
    fig = create_radar_chart(players_data)
    st.plotly_chart(fig, use_container_width=True)

    # Stats table below chart
    with st.expander("📊 View detailed stats", expanded=False):
        rows = []
        for p in players_data:
            r = {"Player": p["name"], **{k: p["stats"][k] for k in STAT_CATEGORIES}}
            rows.append(r)
        tbl = pd.DataFrame(rows)
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    # Player info cards
    st.markdown("---")
    for pname in [player1] + ([player2] if player2 and player2 != "— None —" else []):
        info = cache[pname]["row"]
        club = info.get("Current_club", "—")
        nation = info.get("National_team", "—")
        pos = info.get("Positions", "—")
        overall = info.get("Overall", "—")
        st.info(f"**{pname}** | Club: {club} | Nation: {nation} | Position: {pos} | Overall: {overall}")

    # Predictions section
    st.markdown("---")
    st.subheader("Predictions")
    if ml_models:
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        row1 = cache[player1]["row"]
        with pred_col1:
            try:
                pred_val = predict_value(ml_models, row1)
                actual_val = row1.get("Value", 0)
                st.metric("Predicted Market Value", f"€{pred_val/1e6:.2f}M", f"Actual: €{actual_val/1e6:.2f}M" if actual_val else None)
            except Exception as e:
                st.metric("Predicted Market Value", "—", "N/A")
                st.caption(str(e)[:80])
        with pred_col2:
            try:
                pred_pot = predict_potential(ml_models, row1)
                actual_pot = row1.get("Potential_overall", "—")
                st.metric("Predicted Potential", f"{pred_pot:.0f}", f"Actual: {actual_pot}" if pd.notna(actual_pot) else None)
            except Exception as e:
                st.metric("Predicted Potential", "—", "N/A")
        with pred_col3:
            st.metric("Model R² (Value)", f"{ml_models['metrics']['value_r2']:.2f}", "Training score")
        st.caption("Predictions from Random Forest models trained on the dataset.")

        # Similar players
        st.markdown("#### Similar Players")
        try:
            similar = get_similar_players(ml_models, player1, k=6)
            if similar:
                sim_df = pd.DataFrame(similar, columns=["Player", "Similarity (lower = more similar)"])
                sim_df["Similarity (lower = more similar)"] = sim_df["Similarity (lower = more similar)"].round(2)
                st.dataframe(sim_df, use_container_width=True, hide_index=True)
            else:
                st.caption(f"No similar players found for {player1} (player may not be in training set).")
        except Exception as e:
            st.caption(f"Similar players: {str(e)[:100]}")
    else:
        st.caption("Models could not be trained. Check data columns (Value, Wage, Overall, etc.).")


if __name__ == "__main__":
    main()
