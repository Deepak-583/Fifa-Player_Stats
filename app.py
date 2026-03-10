"""
FIFA Player Stats - Spider Web (Radar) Chart Comparator
Compare strengths and weaknesses of football players using radar charts and ML predictions.
"""

import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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
    """Create radar chart with vibrant multi-color theme."""
    fig = go.Figure()
    # Bold, distinct colors: cyan, gold, coral, lime, magenta
    TEAM_COLORS = ["#00d4ff", "#ffd700", "#ff6b35", "#00ff88", "#ff1493"]
    for i, p in enumerate(players_data):
        name = p["name"]
        stats = p["stats"]
        theta = STAT_CATEGORIES + [STAT_CATEGORIES[0]]
        r = [stats[s] for s in STAT_CATEGORIES] + [stats[STAT_CATEGORIES[0]]]
        color = TEAM_COLORS[i % len(TEAM_COLORS)]
        fig.add_trace(
            go.Scatterpolar(
                r=r,
                theta=theta,
                name=name,
                line=dict(color=color, width=3.5),
                fill="toself",
                opacity=0.45,
            )
        )

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(15,23,42,0.6)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                gridcolor="rgba(0,212,255,0.35)",
                tickfont=dict(color="#94a3b8", size=12),
            ),
            angularaxis=dict(
                gridcolor="rgba(34,197,94,0.3)",
                tickfont=dict(color="#e2e8f0", size=13),
            ),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#e2e8f0", size=14),
            bgcolor="rgba(15,23,42,0.85)",
        ),
        title=dict(
            text="Player Skills Radar Chart",
            font=dict(size=24, color="#00d4ff"),
        ),
        margin=dict(t=80, b=60, l=80, r=80),
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
    )
    return fig


def main():
    st.set_page_config(
        page_title="FIFA Player Stats | Radar Chart",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Vibrant multi-theme styling with gradients
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Oswald:wght@400;600&family=Poppins:wght@400;600&display=swap');
        
        /* Dynamic gradient background: night stadium → pitch green → teal */
        .stApp {
            background: linear-gradient(135deg, 
                #0f172a 0%, 
                #1e3a5f 20%, 
                #0d2818 40%, 
                #1a4d2e 60%, 
                #0c4a6e 80%, 
                #0f172a 100%);
            background-attachment: fixed;
            background-size: 400% 400%;
            animation: bgShift 15s ease infinite;
        }
        @keyframes bgShift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        .stApp::before {
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: radial-gradient(ellipse at 50% 0%, rgba(0,212,255,0.08) 0%, transparent 50%),
                        radial-gradient(ellipse at 80% 80%, rgba(255,215,0,0.06) 0%, transparent 40%),
                        radial-gradient(ellipse at 20% 70%, rgba(34,197,94,0.05) 0%, transparent 40%);
            pointer-events: none;
            z-index: 0;
        }
        
        /* Main content - glass card with cyan-to-amber gradient border */
        .main .block-container {
            padding: 2rem 2rem 3rem;
            max-width: 1200px;
            background: linear-gradient(160deg, 
                rgba(15,23,42,0.9) 0%, 
                rgba(30,58,95,0.85) 30%,
                rgba(26,77,46,0.8) 70%,
                rgba(15,23,42,0.95) 100%);
            border-radius: 16px;
            border: 1px solid transparent;
            background-clip: padding-box;
            box-shadow: 0 0 0 2px rgba(0,212,255,0.3), 
                        0 0 40px rgba(0,212,255,0.1),
                        0 8px 32px rgba(0,0,0,0.4),
                        inset 0 1px 0 rgba(255,255,255,0.05);
            margin-top: 1rem;
        }
        
        /* Title - vibrant gradient text */
        h1 {
            font-family: 'Bebas Neue', sans-serif !important;
            background: linear-gradient(90deg, #00d4ff, #ffd700, #22c55e, #00d4ff);
            background-size: 200% auto;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            font-size: 2.8rem !important;
            filter: drop-shadow(0 0 20px rgba(0,212,255,0.4));
            letter-spacing: 3px;
            animation: gradientText 4s linear infinite;
        }
        @keyframes gradientText {
            to { background-position: 200% center; }
        }
        h2, h3 {
            font-family: 'Oswald', sans-serif !important;
            color: #e2e8f0 !important;
            font-weight: 600 !important;
            text-shadow: 0 0 20px rgba(0,212,255,0.3);
        }
        p, span, div[data-testid="stMarkdown"] {
            color: #f1f5f9 !important;
        }
        .stCaption {
            color: #94a3b8 !important;
        }
        
        /* Selectboxes - cyan/green accent theme */
        .stSelectbox > div {
            background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(15,23,42,0.98)) !important;
            border: 2px solid rgba(0,212,255,0.5) !important;
            border-radius: 10px !important;
            font-family: 'Poppins', sans-serif !important;
            font-weight: 600 !important;
            color: #f8fafc !important;
            box-shadow: 0 0 15px rgba(0,212,255,0.2), inset 0 1px 0 rgba(255,255,255,0.05);
        }
        
        /* Expander - cyan/teal theme */
        .streamlit-expanderHeader {
            background: linear-gradient(90deg, rgba(0,212,255,0.2), rgba(34,197,94,0.15)) !important;
            border: 1px solid rgba(0,212,255,0.4) !important;
            border-radius: 10px !important;
            color: #00d4ff !important;
        }
        
        /* Info boxes - gold/amber gradient theme */
        div[data-testid="stAlert"] {
            background: linear-gradient(135deg, 
                rgba(251,191,36,0.15) 0%, 
                rgba(34,197,94,0.12) 50%,
                rgba(0,212,255,0.1) 100%) !important;
            border: 1px solid rgba(251,191,36,0.5) !important;
            border-radius: 10px !important;
            box-shadow: 0 0 20px rgba(251,191,36,0.1);
        }
        div[data-testid="stAlert"] p, div[data-testid="stAlert"] strong {
            color: #fef9c3 !important;
        }
        
        /* Metrics - each column gets different accent */
        [data-testid="stMetricValue"] {
            font-family: 'Bebas Neue', sans-serif !important;
            font-size: 2rem !important;
            background: linear-gradient(135deg, #00d4ff, #22c55e) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
        }
        [data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
            font-family: 'Oswald', sans-serif !important;
        }
        
        /* Dataframes - subtle gradient border */
        .stDataFrame {
            border-radius: 10px !important;
            overflow: hidden !important;
            border: 1px solid rgba(0,212,255,0.3) !important;
            box-shadow: 0 0 20px rgba(0,0,0,0.2);
        }
        
        /* Divider - gradient line */
        hr {
            height: 2px !important;
            background: linear-gradient(90deg, transparent, #00d4ff, #ffd700, #22c55e, transparent) !important;
            border: none !important;
            margin: 1.5rem 0 !important;
            opacity: 0.6;
        }
        
        /* Hide Streamlit branding */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { 
            background: linear-gradient(90deg, rgba(15,23,42,0.95), rgba(30,58,95,0.9)) !important;
            border-bottom: 1px solid rgba(0,212,255,0.2) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("⚽ FIFA Player Stats – Spider Web Comparator")
    st.caption("🏟️ Compare strengths and weaknesses of football players • Select players to view radar charts")

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
