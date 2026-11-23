import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="CricketIQ | Cricket Analytics", layout="wide", initial_sidebar_state="collapsed")
ROOT = Path(__file__).resolve().parent

st.markdown("""<style>
.stApp {background:#081426;color:#edf3fc}
.block-container {padding-top:1.5rem;max-width:1450px}
h1,h2,h3 {color:#f6f9ff!important}
[data-testid="stMetric"] {background:#142641;border:1px solid #284364;padding:16px;border-radius:14px}
[data-testid="stSidebar"] {background:#0d1b30}
.hero {background:linear-gradient(120deg,#112c52,#102039);padding:32px;border:1px solid #29486b;border-radius:18px;margin:10px 0 24px}
.hero h1 {font-size:2.7rem;margin:0}
.hero p {color:#c3d2e5;font-size:1.1rem}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    data = pd.read_csv(ROOT / "ipl_batting_features.csv", low_memory=False)
    needed = {"player", "date", "team", "opponent", "venue", "runs", "balls", "strike_rate"}
    missing = needed - set(data.columns)
    if missing:
        raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing)))
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for col in ("runs", "balls", "strike_rate"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["player", "date", "runs"]).copy()
    for col in ("player", "team", "opponent", "venue"):
        data[col] = data[col].fillna("Unknown").astype(str).str.strip()
    sort_cols = ["date"] + (["match_id"] if "match_id" in data else [])
    return data.sort_values(sort_cols).reset_index(drop=True)

@st.cache_data
def load_metrics():
    with (ROOT / "model_metrics.json").open(encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_model():
    return joblib.load(ROOT / "model.joblib")

try:
    df = load_data()
    if df.empty:
        st.error("The batting dataset has no usable records.")
        st.stop()
except (OSError, ValueError, pd.errors.ParserError) as exc:
    st.error(f"Unable to load IPL data: {exc}")
    st.stop()

st.title("CricketIQ")
st.caption("AI-powered IPL batting analytics | Historical Cricsheet records | MSc Data Science project")
page = st.radio("Website sections", ["Home", "IPL Dashboard", "Player Explorer", "AI Prediction", "Model Evaluation"], horizontal=True, label_visibility="collapsed")
st.divider()
players = sorted(df["player"].unique().tolist())

if page == "Home":
    st.markdown("""<div class="hero"><h1>Explore cricket through data.</h1>
    <p>Analyze IPL batting performances, compare players and estimate runs with machine learning.</p>
    <p>Historical statistics. Interactive insights. One platform.</p></div>""", unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Players in dataset", f"{df.player.nunique():,}")
    b.metric("Recorded innings", f"{len(df):,}")
    c.metric("Runs in records", f"{df.runs.sum():,.0f}")
    d.metric("Teams in dataset", f"{df.team.nunique():,}")
    st.subheader("Explore the platform")
    st.markdown("**IPL Dashboard:** Season and team analysis with interactive charts.\n\n**Player Explorer:** Recent batting form, opponent comparisons and scorecards.\n\n**AI Prediction:** Estimate runs in a hypothetical match.\n\n**Model Evaluation:** Review the trained model's test results.")
    st.info("This project uses processed historical records, not live IPL data or complete official career totals.")

elif page == "IPL Dashboard":
    st.header("IPL Batting Analytics Dashboard")
    f1, f2, f3 = st.columns([1, 2, 2])
    seasons = sorted(df.date.dt.year.unique().astype(int).tolist(), reverse=True)
    with f1:
        season = st.selectbox("Season", ["All seasons"] + seasons)
    seasonal = df if season == "All seasons" else df[df.date.dt.year == season]
    with f2:
        team_options = sorted(seasonal.team.unique().tolist())
        teams = st.multiselect("Batting team", team_options)
    team_filtered = seasonal if not teams else seasonal[seasonal.team.isin(teams)]
    with f3:
        player_options = sorted(team_filtered.player.unique().tolist())
        chosen_players = st.multiselect("Players", player_options)
    filtered = team_filtered if not chosen_players else team_filtered[team_filtered.player.isin(chosen_players)]
    if filtered.empty:
        st.warning("No records match these filters. Try All seasons or clear the team and player filters.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Total Runs", f"{filtered.runs.sum():,.0f}")
