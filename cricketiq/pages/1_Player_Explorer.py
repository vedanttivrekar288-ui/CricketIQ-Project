import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="CricketIQ | Player Explorer",
    page_icon="🏏",
    layout="wide"
)

st.markdown("""
<style>
.stApp {background-color: #0b1220;}
h1, h2, h3 {color: #f1f5f9;}
[data-testid="stMetric"] {
    background: #172338;
    padding: 18px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("🏏 CricketIQ | Player Explorer")
st.caption("Explore historical IPL batting performances")

DATA_PATH = Path(__file__).resolve().parents[1] / "ipl_batting_features.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# Identify columns from the existing dataset
def find_column(options):
    for name in options:
        if name in df.columns:
            return name
    return None

player_col = find_column([
    "player", "batter", "batsman", "player_name"
])

runs_col = find_column([
    "runs", "batter_runs", "total_runs", "runs_scored"
])

date_col = find_column([
    "date", "match_date", "start_date"
])

if not player_col or not runs_col:
    st.error(
        "Player or runs column not found. "
        "Available columns: " + ", ".join(df.columns)
    )
    st.stop()

df[runs_col] = pd.to_numeric(
    df[runs_col], errors="coerce"
)

st.sidebar.header("🔎 Search Player")

players = sorted(
    df[player_col].dropna().astype(str).unique()
)

selected_player = st.sidebar.selectbox(
    "Select a cricket player",
    players
)

player_data = df[
    df[player_col].astype(str) == selected_player
].copy()

if date_col:
    player_data[date_col] = pd.to_datetime(
        player_data[date_col], errors="coerce"
    )
    player_data = player_data.sort_values(date_col)

st.header(f"🏏 {selected_player}")

total_runs = player_data[runs_col].sum()
innings = len(player_data)
runs_per_innings = (
    total_runs / innings if innings else 0
)
highest_score = player_data[runs_col].max()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Runs", f"{total_runs:,.0f}")
c2.metric("Recorded Innings", innings)
c3.metric("Runs / Innings", f"{runs_per_innings:.2f}")
c4.metric("Highest Score", f"{highest_score:.0f}")

st.divider()

st.subheader("📈 Batting Performance")

plot_data = player_data.reset_index(drop=True)
plot_data["Innings Number"] = range(
    1, len(plot_data) + 1
)

fig = px.line(
    plot_data,
    x="Innings Number",
    y=runs_col,
    markers=True,
    title="Runs Scored Across Innings",
    labels={runs_col: "Runs"}
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("🔥 Recent Form")

recent = player_data.tail(10).copy()
recent["Innings"] = range(1, len(recent) + 1)

fig2 = px.bar(
    recent,
    x="Innings",
    y=runs_col,
    color=runs_col,
    title="Last 10 Recorded Innings",
    labels={runs_col: "Runs"},
    color_continuous_scale="Blues"
)

st.plotly_chart(fig2, use_container_width=True)

st.subheader("📋 Historical Batting Records")
st.dataframe(
    player_data.tail(20).iloc[::-1],
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "Download Player Statistics",
    data=player_data.to_csv(index=False),
    file_name="player_statistics.csv",
    mime="text/csv"
)
