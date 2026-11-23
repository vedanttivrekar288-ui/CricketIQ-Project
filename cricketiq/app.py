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
page = st.sidebar.radio("Navigation", ["Home", "Player Analytics", "IPL Dashboard", "Model Evaluation"], index=0)
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
    st.markdown("**IPL Dashboard:** Season and team analysis with interactive charts.\n\n**Player Analytics:** Select a player once to view their batting profile, recent form, opponent comparisons, and AI run prediction.\n\n**Model Evaluation:** Review the trained model's test results.")
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
        b.metric("Recorded Innings", f"{len(filtered):,}")
        c.metric("Average Strike Rate", f"{filtered.strike_rate.mean():.2f}" if filtered.strike_rate.notna().any() else "N/A")
        d.metric("Players", f"{filtered.player.nunique():,}")
        stats = filtered.groupby("player", as_index=False).agg(total_runs=("runs", "sum"), innings=("runs", "size"), avg_strike_rate=("strike_rate", "mean"))
        top = stats.nlargest(10, "total_runs").sort_values("total_runs")
        st.subheader("Top run scorers")
        fig = px.bar(top, x="total_runs", y="player", orientation="h", color="total_runs", color_continuous_scale="Blues", template="plotly_dark", labels={"total_runs":"Runs", "player":"Player"})
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        left, right = st.columns(2)
        with left:
            st.subheader("Runs vs strike rate")
            scatter = stats.dropna(subset=["avg_strike_rate"])
            if scatter.empty:
                st.info("Strike rate data is unavailable for these records.")
            else:
                fig = px.scatter(scatter, x="total_runs", y="avg_strike_rate", size="innings", hover_name="player", template="plotly_dark", labels={"total_runs":"Runs", "avg_strike_rate":"Average strike rate"})
                st.plotly_chart(fig, use_container_width=True)
        with right:
            st.subheader("Runs by team")
            team_stats = filtered.groupby("team", as_index=False)["runs"].sum()
            team_stats = team_stats[team_stats.runs > 0]
            if team_stats.empty:
                st.info("No positive run totals to chart.")
            else:
                fig = px.pie(team_stats, names="team", values="runs", hole=0.55, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
        st.subheader("Player statistics")
        st.dataframe(stats.sort_values("total_runs", ascending=False), use_container_width=True, hide_index=True)
        st.download_button("Download filtered records", filtered.to_csv(index=False).encode("utf-8"), "cricketiq_filtered.csv", "text/csv")

elif page == "Player Analytics":
    st.header("Player Analytics and AI Prediction")
    default = players.index("V Kohli") if "V Kohli" in players else 0
    selected = st.selectbox("Search and select a player", players, index=default)
    p = df[df.player == selected].copy()
    st.subheader(f"Batting profile: {selected}")
    st.caption(f"Latest recorded team: {p.iloc[-1].team}")
    a, b, c, d = st.columns(4)
    a.metric("Recorded Innings", f"{len(p):,}")
    b.metric("Total Runs", f"{p.runs.sum():,.0f}")
    c.metric("Runs per Innings", f"{p.runs.mean():.1f}")
    d.metric("Highest Score", f"{p.runs.max():.0f}")
    st.caption("Summaries cover the available feature records, not official complete career totals.")
    recent = p.tail(15).copy()
    recent["match_label"] = recent.date.dt.strftime("%d %b %Y") + " | " + recent.opponent
    st.subheader("Recent batting form")
    fig = px.bar(recent, x="match_label", y="runs", color="runs", color_continuous_scale="Blues", template="plotly_dark", labels={"match_label":"Match", "runs":"Runs"})
    fig.update_layout(xaxis_tickangle=-45, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Opponent analysis")
    opp = p.groupby("opponent", as_index=False).agg(innings=("runs", "size"), average_runs=("runs", "mean"))
    fig = px.bar(opp.sort_values("average_runs", ascending=False).head(12), x="opponent", y="average_runs", hover_data=["innings"], template="plotly_dark", labels={"average_runs":"Average runs"})
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Recent scorecards")
    columns = [c for c in ["date", "team", "opponent", "venue", "runs", "balls", "strike_rate"] if c in p]
    st.dataframe(p[columns].tail(10).sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    st.download_button("Download player statistics", p.to_csv(index=False).encode("utf-8"), "player_statistics.csv", "text/csv")

    st.divider()
    st.header("AI Player Runs Prediction")
    latest = p.iloc[-1]
    st.caption("Select a hypothetical match scenario. Predictions are estimates, not guaranteed scores or verified upcoming fixtures.")
    left, right = st.columns(2)
    with left:
        opponent = st.selectbox("Opposition", sorted(df.opponent.unique().tolist()))
    with right:
        venue = st.selectbox("Venue", sorted(df.venue.unique().tolist()))
    if st.button("Predict player runs", type="primary"):
        try:
            metrics = load_metrics()
            model = load_model()
            features = metrics.get("features")
            if not isinstance(features, list) or not features:
                raise ValueError("model_metrics.json must contain a nonempty 'features' list")
            missing = [col for col in features if col not in latest.index]
            if missing:
                raise ValueError("Prediction features missing from CSV: " + ", ".join(missing))
            row = latest[features].to_dict()
            if "opponent" in features:
                row["opponent"] = opponent
            if "venue" in features:
                row["venue"] = venue
            for n in (3, 5, 10):
                if f"avg_runs_last_{n}" in features:
                    row[f"avg_runs_last_{n}"] = float(p.runs.tail(n).mean())
                if f"avg_sr_last_{n}" in features:
                    row[f"avg_sr_last_{n}"] = float(p.strike_rate.tail(n).mean())
            if "career_avg_before" in features:
                row["career_avg_before"] = float(p.runs.mean())
            if "career_sr_before" in features:
                row["career_sr_before"] = float(p.strike_rate.mean())
            if "prior_matches" in features:
                row["prior_matches"] = int(pd.to_numeric(latest["prior_matches"], errors="coerce")) + 1
            estimate = float(model.predict(pd.DataFrame([row], columns=features))[0])
            if not np.isfinite(estimate):
                raise ValueError("Model returned a non-finite prediction")
            st.metric("Estimated runs", f"{max(0, estimate):.0f}")
            st.write(f"{selected} vs {opponent} at {venue}")
            st.caption("Single-point prediction from historical data; this is not a confidence interval.")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            st.error(f"Prediction unavailable: {exc}")

elif page == "Model Evaluation":
    st.header("Model Evaluation")
    try:
        metrics = load_metrics()
        st.write("**Selected model:**", metrics.get("model", "Not specified"))
        st.write("**Chronological test start:**", metrics.get("test_start", "Not specified"))
        st.write("**Data through:**", metrics.get("data_end", "Not specified"))
        results = metrics.get("results", {})
        if isinstance(results, dict) and results:
            st.subheader("Model comparison")
            st.dataframe(pd.DataFrame.from_dict(results, orient="index"), use_container_width=True)
        else:
            st.info("No model comparison results are stored in model_metrics.json.")
        st.caption("MAE and RMSE are measured in runs; lower is better. Individual innings predictions are inherently uncertain.")
    except (OSError, ValueError, TypeError) as exc:
        st.error(f"Unable to read model evaluation: {exc}")

st.divider()
st.caption("CricketIQ | Historical Cricsheet IPL data | Educational machine-learning project")
