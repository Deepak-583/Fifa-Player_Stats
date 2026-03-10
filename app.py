"""
FIFA Player Stats - Spider Web (Radar) Chart Comparator
Compare strengths and weaknesses of football players using radar charts and ML predictions.
"""

import base64
import html
import sys
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

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
PLACEHOLDER_B64 = "data:image/svg+xml;base64," + base64.b64encode(
    b'<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120"><circle cx="60" cy="60" r="55" fill="#1e293b" stroke="#475569" stroke-width="2"/><text x="60" y="65" fill="#94a3b8" font-size="14" text-anchor="middle">?</text></svg>'
).decode()


def fetch_player_image(url: str) -> str | None:
    """Fetch image from URL server-side and return as base64 data URI. Bypasses hotlink protection."""
    if not url or not str(url).startswith("http"):
        return None
    try:
        r = requests.get(
            url,
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        if r.status_code == 200:
            ext = "png" if "png" in url else "jpeg" if "jpg" in url else "svg+xml" if "svg" in url else "png"
            b64 = base64.b64encode(r.content).decode()
            return f"data:image/{ext};base64,{b64}"
    except Exception:
        pass
    return None


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
        
        /* Detailed stats table - white gradient, centered numbers */
        .streamlit-expanderContent {
            background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.98) 100%) !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            border: 1px solid rgba(0,212,255,0.3) !important;
        }
        .streamlit-expanderContent .stDataFrame {
            background: transparent !important;
        }
        .streamlit-expanderContent [data-testid="stDataFrame"] table {
            font-family: 'Oswald', sans-serif !important;
        }
        .streamlit-expanderContent [data-testid="stDataFrame"] th {
            background: linear-gradient(90deg, #1e3a5f, #0c4a6e) !important;
            color: #f8fafc !important;
            text-align: center !important;
            font-weight: 600 !important;
        }
        .streamlit-expanderContent [data-testid="stDataFrame"] td {
            text-align: center !important;
            color: #0f172a !important;
            background: rgba(255,255,255,0.6) !important;
        }
        .streamlit-expanderContent [data-testid="stDataFrame"] td:first-child {
            font-weight: 600 !important;
            color: #0c4a6e !important;
        }
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
        
        /* VS Box styling */
        .vs-container {
            display: flex;
            align-items: stretch;
            justify-content: center;
            gap: 0;
            padding: 1.5rem;
            margin: 1.5rem 0;
            background: linear-gradient(135deg, rgba(15,23,42,0.6), rgba(30,58,95,0.5));
            border-radius: 16px;
            border: 2px solid rgba(0,212,255,0.4);
            box-shadow: 0 0 30px rgba(0,212,255,0.15), inset 0 1px 0 rgba(255,255,255,0.05);
        }
        .vs-player-box {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 1.5rem;
            border-radius: 12px;
        }
        .vs-player-box.left { background: linear-gradient(180deg, rgba(0,212,255,0.12), transparent); }
        .vs-player-box.right { background: linear-gradient(180deg, rgba(255,215,0,0.12), transparent); }
        .vs-divider {
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 80px;
            font-family: 'Bebas Neue', sans-serif;
            font-size: 2.5rem;
            color: #ffd700;
            text-shadow: 0 0 20px rgba(255,215,0,0.6);
        }
        .vs-img {
            width: 120px;
            height: 120px;
            object-fit: contain;
            border-radius: 50%;
            border: 3px solid rgba(255,255,255,0.3);
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .vs-name { 
            margin-top: 0.75rem; 
            font-weight: 600; 
            color: #e2e8f0;
            text-align: center;
        }
        .vs-overall {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 1.5rem;
            color: #00d4ff;
            margin-top: 0.25rem;
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

    # VS Box: Player 1 | VS | Player 2 with images (fetched server-side to bypass CDN hotlink block)
    if "image_cache" not in st.session_state:
        st.session_state["image_cache"] = {}

    def get_image_src(url: str) -> str:
        if not url or not str(url).startswith("http"):
            return PLACEHOLDER_B64
        url_str = str(url)
        if url_str not in st.session_state["image_cache"]:
            data_uri = fetch_player_image(url_str)
            st.session_state["image_cache"][url_str] = data_uri or PLACEHOLDER_B64
        return st.session_state["image_cache"][url_str]

    row1 = cache[player1]["row"]
    img1_url = row1.get("Images", "")
    overall1 = row1.get("Overall", "—")
    img1_src = get_image_src(img1_url)

    if player2 and player2 != "— None —":
        row2 = cache[player2]["row"]
        img2_url = row2.get("Images", "")
        overall2 = row2.get("Overall", "—")
        img2_src = get_image_src(img2_url)
        name2_esc = html.escape(player2)
        ovr2_html = f'OVR {overall2}'
    else:
        img2_src = PLACEHOLDER_B64
        name2_esc = "Select Player 2"
        ovr2_html = "—"

    st.markdown(
        f"""
        <div class="vs-container">
            <div class="vs-player-box left">
                <img src="{img1_src}" class="vs-img" alt="{html.escape(player1)}" />
                <div class="vs-name">{html.escape(player1)}</div>
                <div class="vs-overall">OVR {overall1}</div>
            </div>
            <div class="vs-divider">VS</div>
            <div class="vs-player-box right">
                <img src="{img2_src}" class="vs-img" alt="{name2_esc}" />
                <div class="vs-name">{name2_esc}</div>
                <div class="vs-overall">{ovr2_html}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main: Radar chart
    fig = create_radar_chart(players_data)
    st.plotly_chart(fig, use_container_width=True)

    # Stats table below chart (white gradient box, centered numbers)
    with st.expander("📊 View detailed stats", expanded=False):
        rows = []
        for p in players_data:
            r = {"Player": p["name"], **{k: p["stats"][k] for k in STAT_CATEGORIES}}
            rows.append(r)
        tbl = pd.DataFrame(rows)
        st.dataframe(tbl, use_container_width=True, hide_index=True, column_config={
            c: st.column_config.NumberColumn(c, format="%d") for c in STAT_CATEGORIES
        })

    # Key Stats - what football fans care about (Age, Value, Wage, etc.)
    st.markdown("---")
    st.subheader("Key Stats")
    key_col1, key_col2 = st.columns(2)
    for idx, pname in enumerate([player1] + ([player2] if player2 and player2 != "— None —" else [])):
        info = cache[pname]["row"]
        age = info.get("Age", "—")
        overall = info.get("Overall", "—")
        potential = info.get("Potential_overall", "—")
        pos = info.get("Positions", "—")
        value = info.get("Value", 0)
        wage = info.get("Wage", 0)
        club = info.get("Current_club", "—")
        nation = info.get("National_team", "—")
        value_str = f"€{value/1e6:.1f}M" if value and value > 0 else "—"
        wage_str = f"€{wage/1000:.0f}K" if wage and wage > 0 else "—"
        with key_col1 if idx == 0 else key_col2:
            st.markdown(f"""
            **{pname}**  
            Age: {age} • Overall: {overall} • Potential: {potential} • Position: {pos}  
            Value: {value_str} • Wage: {wage_str}  
            Club: {club} • Nation: {nation}
            """)


if __name__ == "__main__":
    main()
