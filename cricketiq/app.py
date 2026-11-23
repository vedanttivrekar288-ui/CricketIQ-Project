    if player_choice:
        filtered = filtered[filtered.player.isin(player_choice)]
    if filtered.empty:
        st.warning("No records match these filters. Change the selections above.")
        st.stop()
    x1, x2, x3, x4 = st.columns(4)
    x1.metric("Total runs", f"{filtered.runs.sum():,.0f}")
    x2.metric("Recorded innings", f"{len(filtered):,}")
    x3.metric("Runs per innings", f"{filtered.runs.mean():.1f}")
    x4.metric("Average strike rate", f"{filtered.strike_rate.mean():.1f}")
    st.caption("All metrics use the filtered processed batting dataset. Strike rate is the unweighted mean of innings strike rates.")
    ranking = (filtered.groupby("player", as_index=False).agg(runs=("runs", "sum"), innings=("runs", "size"), avg_strike_rate=("strike_rate", "mean"))
               .sort_values("runs", ascending=False))
    team_runs = filtered.groupby("team", as_index=False).agg(runs=("runs", "sum")).sort_values("runs", ascending=False)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(ranking.head(12).sort_values("runs"), x="runs", y="player", orientation="h", title="Top batters by recorded runs", color="runs", color_continuous_scale="Blues", template="plotly_dark"), use_container_width=True)
    with right:
        st.plotly_chart(px.scatter(ranking, x="runs", y="avg_strike_rate", size="innings", hover_name="player", title="Runs vs average strike rate", template="plotly_dark"), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(team_runs, x="team", y="runs", color="team", title="Recorded runs by batting team", template="plotly_dark"), use_container_width=True)
    with right:
        st.plotly_chart(px.pie(team_runs, names="team", values="runs", hole=0.52, title="Team share of recorded runs", template="plotly_dark"), use_container_width=True)
    st.subheader("Player rankings")
    st.dataframe(ranking, use_container_width=True, hide_index=True)
    st.download_button("Download filtered batting data", filtered.to_csv(index=False), "cricketiq_filtered_data.csv", "text/csv")
    st.caption("Boundary counts and 50/100 milestones are not displayed because the existing processed dataset has not been verified to contain those fields.")

elif page == "Player Explorer":
    st.header(f"Player Explorer: {selected_player}")
    st.write(f"Latest recorded team: **{latest['team']}** | Historical data through **{metrics.get('data_end', 'not specified')}**")
    a, b, c, d = st.columns(4)
    a.metric("Recorded innings", len(player_df))
    b.metric("Total runs in records", f"{player_df.runs.sum():,.0f}")
    c.metric("Runs per innings", f"{player_df.runs.mean():.1f}")
    d.metric("Highest innings", f"{player_df.runs.max():.0f}")
    st.caption("These are processed dataset statistics, not official complete career totals.")
    recent = player_df.tail(15).copy()
    recent["match_label"] = recent.date.dt.strftime("%d %b %Y") + " vs " + recent.opponent.astype(str)
    fig = px.bar(recent, x="match_label", y="runs", color="runs", title="Runs in last 15 recorded innings", color_continuous_scale="Teal", template="plotly_dark")
    fig.update_layout(xaxis_tickangle=-45, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    left, right = st.columns(2)
    with left:
        opp = player_df.groupby("opponent", as_index=False).agg(innings=("runs", "size"), average_runs=("runs", "mean"))
        opp = opp[opp.innings >= 3].sort_values("average_runs", ascending=False).head(12)
        if not opp.empty:
            st.plotly_chart(px.bar(opp, x="opponent", y="average_runs", hover_data=["innings"], title="Average runs by opponent (min 3 innings)", template="plotly_dark"), use_container_width=True)
        else:
            st.info("Not enough opponent records for this chart.")
    with right:
        st.plotly_chart(px.line(player_df.tail(20), x="date", y="runs", markers=True, title="Recent batting form over time", template="plotly_dark"), use_container_width=True)
    st.subheader("Historical match records")
    st.dataframe(player_df[["date", "team", "opponent", "venue", "runs", "balls", "strike_rate"]].sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    st.download_button("Download player statistics", player_df.to_csv(index=False), "player_statistics.csv", "text/csv")

elif page == "AI Prediction":
    st.header("AI Player Performance Prediction")
    st.write(f"Selected player: **{selected_player}**")
    st.info("Choose a hypothetical match scenario. This is not a verified upcoming IPL fixture.")
    a, b = st.columns(2)
    with a:
        opponent = st.selectbox("Opposition", sorted(df.opponent.dropna().astype(str).unique()))
    with b:
        venue = st.selectbox("Venue", sorted(df.venue.dropna().astype(str).unique()))
    x, y, z = st.columns(3)
    x.metric("Last 3 innings average", f"{player_df.runs.tail(3).mean():.1f}")
    y.metric("Last 5 innings average", f"{player_df.runs.tail(5).mean():.1f}")
    z.metric("Last 10 innings average", f"{player_df.runs.tail(10).mean():.1f}")
    if st.button("Predict player runs", type="primary"):
        try:
            model = load_model()
            feature_cols = metrics["features"]
            row = latest[feature_cols].to_dict()
            row["opponent"] = opponent
            row["venue"] = venue
            for n in (3, 5, 10):
                row[f"avg_runs_last_{n}"] = float(player_df.runs.tail(n).mean())
                row[f"avg_sr_last_{n}"] = float(player_df.strike_rate.tail(n).mean())
            row["career_avg_before"] = float(player_df.runs.mean())
            row["career_sr_before"] = float(player_df.strike_rate.mean())
            row["prior_matches"] = int(latest.prior_matches) + 1
            prediction_input = pd.DataFrame([row], columns=feature_cols)
            estimated_runs = max(0.0, float(model.predict(prediction_input)[0]))
            st.metric("Estimated runs", f"{estimated_runs:.0f}")
            st.write(f"{selected_player} vs {opponent} at {venue}")
            st.caption("Experimental point estimate based on historical records; not a confidence interval or guaranteed score.")
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")

elif page == "Model Evaluation":
    st.header("Machine Learning Model Evaluation")
    st.write(f"**Selected model:** {metrics.get('model', 'Not specified')}")
    st.write(f"**Chronological test begins:** {metrics.get('test_start', 'Not specified')}")
    if "results" in metrics:
        results = pd.DataFrame(metrics["results"]).T
        st.dataframe(results, use_container_width=True)
    st.markdown("**MAE:** Average absolute prediction error in runs.\n\n**RMSE:** Penalizes large prediction errors more heavily.\n\n**R²:** Performance relative to a mean-value baseline; may be negative.")
    st.info("Individual cricket innings are difficult to predict accurately due to unpredictable match conditions.")

st.divider()
st.caption("CricketIQ | MSc Data Science Project | Historical data: Cricsheet | Educational use only")
