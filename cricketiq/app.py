import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='CricketIQ | AI Player Analytics',page_icon='🏏',layout='wide')
st.markdown('''<style>
.stApp {background:#091322;color:#edf3fc}
[data-testid="stMetric"] {background:#14243a;border:1px solid #263d57;padding:15px;border-radius:14px}
.block-container {padding-top:1.7rem}
h1,h2,h3 {color:#f5f8ff!important}
</style>''',unsafe_allow_html=True)

@st.cache_resource
def load_model(): return joblib.load(ROOT/'model.joblib')
@st.cache_data
def load_data(): return pd.read_csv(ROOT/'ipl_batting_features.csv',parse_dates=['date'])
try:
    df=load_data(); model=load_model(); metrics=json.loads((ROOT/'model_metrics.json').read_text())
except FileNotFoundError:
    st.error('Missing generated data/model files. Run prepare_data.py and train_model.py in Colab, then upload the outputs to GitHub.')
    st.stop()

st.title('🏏 CricketIQ')
st.caption('AI-powered IPL batting analytics • Historical data: Cricsheet • Educational project')
st.warning('Predictions are experimental estimates, not guaranteed scores. This app does not fetch live fixtures or current player biographies.')

players=sorted(df.player.unique())
selected=st.sidebar.selectbox('🔎 Search player',players,index=players.index('V Kohli') if 'V Kohli' in players else 0)
st.sidebar.caption('Names follow Cricsheet scorecard notation (for example, V Kohli).')
player_df=df[df.player==selected].sort_values(['date','match_id'])
latest=player_df.iloc[-1]
st.subheader(f'👤 {selected}')
st.write(f'Latest recorded IPL team: **{latest.team}**  |  Historical data through **{metrics["data_end"]}**')
st.caption('Player summaries below are computed from the filtered feature dataset (appearances after the first three), not official complete career totals.')
a,b,c,d=st.columns(4)
a.metric('Recorded innings',f'{len(player_df):,}')
b.metric('Total runs in records',f'{player_df.runs.sum():,}')
c.metric('Runs per innings',f'{player_df.runs.mean():.1f}')
d.metric('Highest innings',str(int(player_df.runs.max())))

st.subheader('📈 Recent batting form')
last=player_df.tail(15).copy();last['match_label']=last.date.dt.strftime('%d %b %Y')+' • '+last.opponent
fig=px.bar(last,x='match_label',y='runs',color='runs',color_continuous_scale='Teal',
           title='Runs in last 15 recorded batting innings',labels={'match_label':'Match','runs':'Runs'})
fig.update_layout(xaxis_tickangle=-55,template='plotly_dark',coloraxis_showscale=False)
st.plotly_chart(fig,use_container_width=True)

left,right=st.columns(2)
with left:
    st.subheader('🏟️ Opponent analysis')
    agg=player_df.groupby('opponent').agg(innings=('runs','size'),avg_runs=('runs','mean')).reset_index()
    agg=agg[agg.innings>=3].sort_values('avg_runs',ascending=False).head(12)
    st.plotly_chart(px.bar(agg,x='opponent',y='avg_runs',hover_data=['innings'],template='plotly_dark',title='Average runs vs opponents (min 3 innings)'),use_container_width=True)
with right:
    st.subheader('📋 Recent scorecards')
    st.dataframe(player_df[['date','team','opponent','venue','runs','balls','strike_rate']].tail(10).sort_values('date',ascending=False),hide_index=True,use_container_width=True)

st.subheader('🎯 Predict runs for a future match scenario')
st.caption('This is a user-selected hypothetical fixture, NOT a verified upcoming match.')
c1,c2,c3=st.columns(3)
with c1: opponent=st.selectbox('Opposition',sorted(df.opponent.unique()))
with c2: venue=st.selectbox('Venue',sorted(df.venue.unique()))
with c3: st.write('**Format**');st.info('IPL / T20')
if st.button('✨ Predict player runs',type='primary'):
    feature_cols=metrics['features']
    row=latest[feature_cols].to_dict()
    row['opponent']=opponent;row['venue']=venue
    # Recompute current form using all available innings (latest row has pre-match lagged values).
    for n in (3,5,10):
        row[f'avg_runs_last_{n}']=float(player_df.runs.tail(n).mean())
        row[f'avg_sr_last_{n}']=float(player_df.strike_rate.tail(n).mean())
    row['career_avg_before']=float(player_df.runs.mean())
    row['career_sr_before']=float(player_df.strike_rate.mean())
    row['prior_matches']=int(latest.prior_matches)+1
    pred=max(0,float(model.predict(pd.DataFrame([row]))[0]))
    st.metric('Estimated runs',f'{pred:.0f}')
    st.write(f'**{selected} vs {opponent} at {venue}**')
    st.caption('Single-point estimate based on past IPL records; not a probability or confidence interval.')

st.divider()
st.subheader('🧠 Model evaluation')
st.write(f'**Selected model:** {metrics["model"]} | **Chronological test begins:** {metrics["test_start"]}')
st.dataframe(pd.DataFrame(metrics['results']).T,use_container_width=True)
st.caption('MAE/RMSE in runs (lower is better). R² may be low or negative because individual innings are difficult to predict.')
st.caption('Source: Cricsheet IPL ball-by-ball data. No live schedule or verified biographical information is provided.')
