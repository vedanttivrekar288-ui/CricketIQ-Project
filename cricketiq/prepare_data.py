import io, json, zipfile, requests
from pathlib import Path
import numpy as np
import pandas as pd

URL = 'https://cricsheet.org/downloads/ipl_json.zip'
OUT = Path(__file__).resolve().parent
print('Downloading IPL ball-by-ball JSON from Cricsheet...')
r = requests.get(URL, timeout=120, headers={'User-Agent':'CricketIQ educational project'}); r.raise_for_status()
archive = zipfile.ZipFile(io.BytesIO(r.content))
records=[]
for filename in archive.namelist():
    if not filename.endswith('.json') or filename.startswith('__MACOSX'): continue
    try: match=json.loads(archive.read(filename))
    except (ValueError, UnicodeDecodeError): continue
    info=match.get('info',{})
    dates=info.get('dates',[])
    if not dates: continue
    date=str(dates[0]); venue=str(info.get('venue','Unknown'))
    teams=info.get('teams',[])
    if len(teams)<2: continue
    for innings in match.get('innings',[]):
        batting_team=innings.get('team')
        if batting_team not in teams: continue
        opponent=teams[1] if batting_team==teams[0] else teams[0]
        batter_runs={}; balls={}; fours={}; sixes={}
        for over in innings.get('overs',[]):
            for d in over.get('deliveries',[]):
                batter=d.get('batter')
                if not batter: continue
                runs=d.get('runs',{}).get('batter',0)
                batter_runs[batter]=batter_runs.get(batter,0)+runs
                extras=d.get('extras',{})
                # Wides do not count as balls faced; no-balls do.
                balls[batter]=balls.get(batter,0)+(0 if 'wides' in extras else 1)
                fours[batter]=fours.get(batter,0)+(1 if runs==4 else 0)
                sixes[batter]=sixes.get(batter,0)+(1 if runs==6 else 0)
        for player,runs in batter_runs.items():
            records.append(dict(date=date,match_id=Path(filename).stem,player=player,
                                team=batting_team,opponent=opponent,venue=venue,
                                runs=runs,balls=balls[player],fours=fours[player],sixes=sixes[player]))

if not records: raise RuntimeError('No match records parsed; check source format or internet connection.')
df=pd.DataFrame(records)
df['date']=pd.to_datetime(df['date'],errors='coerce')
df=df.dropna(subset=['date']).sort_values(['date','match_id','player']).reset_index(drop=True)
df['strike_rate']=np.where(df.balls>0,df.runs/df.balls*100,0)
# Lagged features: only earlier matches for the same player, never the target match.
g=df.groupby('player',sort=False)
df['prior_matches']=g.cumcount()
for n in [3,5,10]:
    df[f'avg_runs_last_{n}']=g['runs'].transform(lambda s:s.shift(1).rolling(n,min_periods=1).mean())
    df[f'avg_sr_last_{n}']=g['strike_rate'].transform(lambda s:s.shift(1).rolling(n,min_periods=1).mean())
df['career_avg_before']=g['runs'].transform(lambda s:s.shift(1).expanding().mean())
df['career_sr_before']=g['strike_rate'].transform(lambda s:s.shift(1).expanding().mean())
# Remove first 3 appearances: insufficient prior history for prediction.
df=df[df.prior_matches>=3].copy()
for c in ['avg_runs_last_3','avg_runs_last_5','avg_runs_last_10','avg_sr_last_3','avg_sr_last_5','avg_sr_last_10','career_avg_before','career_sr_before']:
    df[c]=df[c].fillna(0)
df['date']=df['date'].dt.strftime('%Y-%m-%d')
df.to_csv(OUT/'ipl_batting_features.csv',index=False)
print(f'Saved {len(df):,} historical player-innings records, {df.player.nunique()} players')
