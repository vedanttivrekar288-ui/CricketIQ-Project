import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="CricketIQ | Cricket Analytics", page_icon="CI", layout="wide", initial_sidebar_state="collapsed")
ROOT = Path(__file__).resolve().parent

st.markdown("""<style>
.stApp {background:#081426;color:#edf3fc}
.block-container {padding-top:1.5rem;max-width:1450px}
h1,h2,h3 {color:#f6f9ff!important}
[data-testid="stMetric"] {background:#142641;border:1px solid #284364;padding:16px;border-radius:14px}
[data-testid="stSidebar"] {background:#0d1b30}
div.stButton>button[kind="primary"] {background:#0879e8;border-color:#0879e8}
.hero {background:linear-gradient(120deg,#112c52,#102039);padding:35px;border:1px solid #29486b;border-radius:18px;margin:10px 0 24px}
.hero h1 {font-size:3rem;margin:0}
.hero p {color:#c3d2e5;font-size:1.1rem}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "ipl_batting_features.csv", parse_dates=["date"])

@st.cache_resource
def load_model():
    return joblib.load(ROOT / "model.joblib")

@st.cache_data
def load_metrics():
    return json.loads((ROOT / "model_metrics.json").read_text(encoding="utf-8"))

try:
    df = load_data()
    metrics = load_metrics()
except (FileNotFoundError, ValueError, KeyError) as exc:
    st.error(f"Could not load cricket data or metrics: {exc}")
    st.stop()

required = {"player", "date", "match_id", "team", "opponent", "venue", "runs", "balls", "strike_rate", "prior_matches"}
missing = required - set(df.columns)
if missing:
    st.error("Missing dataset columns: " + ", ".join(sorted(missing)))
    st.stop()

df = df.dropna(subset=["player", "date", "runs"]).sort_values(["date", "match_id"]).copy()
if df.empty:
    st.error("No valid batting records found.")
    st.stop()

st.markdown("## CricketIQ")
st.caption("AI-powered IPL batting analytics | Historical Cricsheet records | MSc Data Science project")
page = st.radio("Website sections", ["Home", "IPL Dashboard", "Player Explorer", "AI Prediction", "Model Evaluation"], horizontal=True, label_visibility="collapsed")
st.divider()

players = sorted(df["player"].astype(str).unique())
default_player = players.index("V Kohli") if "V Kohli" in players else 0

# Player selection comes before all player-specific information.
if page in ("Player Explorer", "AI Prediction"):
    selected_player = st.selectbox("Search and select a player", players, index=default_player)
    player_df = df.loc[df["player"] == selected_player].sort_values(["date", "match_id"]).copy()
    latest = player_df.iloc[-1]

if page == "Home":
    st.markdown("""<div class="hero"><h1>Explore cricket through data.</h1>
    <p>Analyze IPL batting performances, discover player trends and estimate runs using machine learning.</p>
    <p>One platform. Historical statistics. Interactive insights.</p></div>""", unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Players in dataset", f"{df.player.nunique():,}")
    b.metric("Recorded innings", f"{len(df):,}")
    c.metric("Runs in records", f"{df.runs.sum():,.0f}")
    d.metric("Teams in dataset", f"{df.team.nunique():,}")
    st.subheader("What you can explore")
    st.markdown("**IPL Dashboard:** Filter historical records and compare teams and players.\n\n**Player Explorer:** Study batting form and opponent performance.\n\n**AI Prediction:** Estimate runs for a hypothetical match using your trained model.\n\n**Model Evaluation:** Review testing results and model limitations.")
    st.info("The dataset contains processed historical records, not complete official career totals or live match information.")

elif page == "IPL Dashboard":
    st.header("IPL Batting Analytics Dashboard")
    f1, f2, f3 = st.columns([1, 2, 2])
    years = sorted(df.date.dt.year.dropna().astype(int).unique().tolist())
    with f1:
        year_choice = st.selectbox("Season", ["All seasons"] + years)
    with f2:
        teams = sorted(df.team.dropna().astype(str).unique())
        team_choice = st.multiselect("Batting team", teams, default=[])
    with f3:
        player_choice = st.multiselect("Players", players, default=[])
    filtered = df.copy()
    if year_choice != "All seasons":
        filtered = filtered[filtered.date.dt.year == year_choice]
    if team_choice:
        filtered = filtered[filtered.team.isin(team_choice)]
