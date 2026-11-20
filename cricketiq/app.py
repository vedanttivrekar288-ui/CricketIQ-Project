
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ==================================================
# 1. WEBSITE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="CricketIQ | AI Cricket Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent


# ==================================================
# 2. WEBSITE DESIGN
# ==================================================

st.markdown("""
<style>
.stApp {
    background-color: #091322;
    color: #edf3fc;
}

[data-testid="stMetric"] {
    background-color: #14243a;
    border: 1px solid #263d57;
    padding: 15px;
    border-radius: 14px;
}

.block-container {
    padding-top: 1.7rem;
}

h1, h2, h3 {
    color: #f5f8ff !important;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. LOAD DATA AND TRAINED AI MODEL
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv(
        ROOT / "ipl_batting_features.csv",
        parse_dates=["date"]
    )


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "model.joblib")


@st.cache_data
def load_metrics():
    with open(ROOT / "model_metrics.json") as file:
        return json.load(file)


try:
    df = load_data()
    model = load_model()
    metrics = load_metrics()

except FileNotFoundError as error:
    st.error(f"Required project file missing: {error}")
    st.stop()

except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


# ==================================================
# 4. WEBSITE HEADER
# ==================================================

st.title("🏏 CricketIQ")
st.caption(
    "AI-Powered IPL Player Performance Prediction "
    "| Historical Cricket Analytics"
)

st.warning(
    "Predictions are experimental estimates based "
    "on historical data, not guaranteed match scores."
)


# ==================================================
# 5. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("🏏 CricketIQ")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔍 Player Explorer",
        "🎯 AI Prediction",
        "🧠 Model Evaluation"
    ]
)

players = sorted(df["player"].dropna().unique())

default_player = (
    players.index("V Kohli")
    if "V Kohli" in players
    else 0
)

selected_player = st.sidebar.selectbox(
    "🔎 Search Player",
    players,
    index=default_player
)

st.sidebar.caption(
    "Historical player names follow Cricsheet notation."
)


# ==================================================
# 6. FILTER SELECTED PLAYER
# ==================================================

player_df = (
    df[df["player"] == selected_player]
    .sort_values(["date", "match_id"])
    .copy()
)

if player_df.empty:
    st.error("No records available for this player.")
    st.stop()

latest = player_df.iloc[-1]


# ==================================================
# 7. DASHBOARD PAGE
# ==================================================

if page == "🏠 Dashboard":

    st.header("📊 CricketIQ Dashboard")

    st.subheader(f"👤 {selected_player}")

    st.write(
        f"Latest recorded IPL team: **{latest['team']}**"
    )

    st.caption(
        f"Historical data through: "
        f"{metrics.get('data_end', 'Not specified')}"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Recorded Innings",
        len(player_df)
    )

    col2.metric(
        "Total Runs in Records",
        f"{player_df['runs'].sum():,.0f}"
    )

    col3.metric(
        "Runs Per Innings",
        f"{player_df['runs'].mean():.2f}"
    )

    col4.metric(
        "Highest Score",
        int(player_df["runs"].max())
    )

    st.caption(
        "These statistics are calculated from the "
        "processed feature dataset, not complete "
        "official career records."
    )

    st.subheader("📈 Recent Batting Performance")

    recent = player_df.tail(15).copy()

    recent["match_label"] = (
        recent["date"].dt.strftime("%d %b %Y")
        + " vs "
        + recent["opponent"].astype(str)
    )

    fig = px.bar(
        recent,
        x="match_label",
        y="runs",
        color="runs",
        color_continuous_scale="Teal",
        title="Runs in Last 15 Recorded Innings",
        labels={
            "match_label": "Match",
            "runs": "Runs"
        }
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_tickangle=-45,
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==================================================
# 8. PLAYER EXPLORER PAGE
# ==================================================

elif page == "🔍 Player Explorer":

    st.header("🔍 Player Explorer")

    st.subheader(f"Player Profile: {selected_player}")

    st.write(
        f"**Latest Recorded Team:** {latest['team']}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Recorded Innings",
        len(player_df)
    )

    col2.metric(
        "Total Runs",
        f"{player_df['runs'].sum():,.0f}"
    )

    col3.metric(
        "Average Strike Rate",
        f"{player_df['strike_rate'].mean():.2f}"
    )

    st.subheader("🏟️ Performance Against Opponents")

    opponent_stats = (
        player_df.groupby("opponent")
        .agg(
            innings=("runs", "size"),
            average_runs=("runs", "mean")
        )
        .reset_index()
    )

    opponent_stats = opponent_stats[
        opponent_stats["innings"] >= 3
    ]

    fig = px.bar(
        opponent_stats,
        x="opponent",
        y="average_runs",
        color="average_runs",
        title="Average Runs Against IPL Teams",
        hover_data=["innings"]
    )

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("📈 Batting Form Over Time")

    form = player_df.tail(20).copy()

    fig2 = px.line(
        form,
        x="date",
        y="runs",
        markers=True,
        title="Recent Batting Performance"
    )

    fig2.update_layout(template="plotly_dark")

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.subheader("📋 Historical Match Records")

    display_columns = [
        "date",
        "team",
        "opponent",
        "venue",
        "runs",
        "balls",
        "strike_rate"
    ]

    st.dataframe(
        player_df[display_columns]
        .sort_values("date", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "📥 Download Player Statistics",
        data=player_df.to_csv(index=False),
        file_name="player_statistics.csv",
        mime="text/csv"
    )


# ==================================================
# 9. AI PERFORMANCE PREDICTION PAGE
# ==================================================

elif page == "🎯 AI Prediction":

    st.header("🎯 AI Player Performance Prediction")

    st.write(
        f"Predicting batting performance for "
        f"**{selected_player}**"
    )

    st.info(
        "Select a hypothetical IPL match scenario. "
        "This is not a verified upcoming fixture."
    )

    col1, col2 = st.columns(2)

    with col1:
        opponent = st.selectbox(
            "Select Opposition Team",
            sorted(df["opponent"].dropna().unique())
        )

    with col2:
        venue = st.selectbox(
            "Select Match Venue",
            sorted(df["venue"].dropna().unique())
        )

    st.subheader("📊 Recent Player Form")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Last 3 Innings Average",
        f"{player_df['runs'].tail(3).mean():.1f}"
    )

    col2.metric(
        "Last 5 Innings Average",
        f"{player_df['runs'].tail(5).mean():.1f}"
    )

    col3.metric(
        "Last 10 Innings Average",
        f"{player_df['runs'].tail(10).mean():.1f}"
    )

    if st.button(
        "✨ Predict Player Runs",
        type="primary"
    ):

        feature_cols = metrics["features"]

        # Start with the most recent feature record
        prediction_row = latest[feature_cols].to_dict()

        # Update hypothetical match conditions
        prediction_row["opponent"] = opponent
        prediction_row["venue"] = venue

        # Calculate current batting form
        for n in [3, 5, 10]:

            prediction_row[
                f"avg_runs_last_{n}"
            ] = float(
                player_df["runs"].tail(n).mean()
            )

            prediction_row[
                f"avg_sr_last_{n}"
            ] = float(
                player_df["strike_rate"].tail(n).mean()
            )

        prediction_row["career_avg_before"] = float(
            player_df["runs"].mean()
        )

        prediction_row["career_sr_before"] = float(
            player_df["strike_rate"].mean()
        )

        prediction_row["prior_matches"] = (
            int(latest["prior_matches"]) + 1
        )

        # Convert input into model-compatible DataFrame
        prediction_input = pd.DataFrame(
            [prediction_row],
            columns=feature_cols
        )

        # AI prediction
        predicted_runs = float(
            model.predict(prediction_input)[0]
        )

        predicted_runs = max(0, predicted_runs)

        st.success("Prediction Completed Successfully!")

        st.metric(
            "🏏 Estimated Runs",
            f"{predicted_runs:.0f}"
        )

        st.write(
            f"**Player:** {selected_player}"
        )

        st.write(
            f"**Opposition:** {opponent}"
        )

        st.write(
            f"**Venue:** {venue}"
        )

        st.caption(
            "This is a single-point estimate, "
            "not a confidence interval."
        )


# ==================================================
# 10. MACHINE LEARNING MODEL EVALUATION
# ==================================================

elif page == "🧠 Model Evaluation":

    st.header("🧠 Machine Learning Model Evaluation")

    st.write(
        f"**Selected Model:** "
        f"{metrics.get('model', 'Not specified')}"
    )

    st.write(
        f"**Test Period Begins:** "
        f"{metrics.get('test_start', 'Not specified')}"
    )

    st.subheader("📊 Model Comparison")

    results = pd.DataFrame(
        metrics["results"]
    ).T

    st.dataframe(
        results,
        use_container_width=True
    )

    st.subheader("📉 Evaluation Metrics")

    st.write(
        "**MAE:** Mean Absolute Error measures "
        "average prediction error in runs."
    )

    st.write(
        "**RMSE:** Root Mean Squared Error "
        "penalizes larger prediction errors."
    )

    st.write(
        "**R² Score:** Measures how much variation "
        "the model explains compared with "
        "a mean-value baseline."
    )

    st.info(
        "Cricket innings are affected by many "
        "unpredictable factors, so model accuracy "
        "is limited."
    )


# ==================================================
# 11. FOOTER
# ==================================================

st.divider()

st.caption(
    "🏏 CricketIQ | MSc Data Science Project | "
    "Historical IPL data: Cricsheet"
)
