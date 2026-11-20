from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUT=Path(__file__).resolve().parent
df=pd.read_csv(OUT/'ipl_batting_features.csv',parse_dates=['date']).sort_values(['date','match_id','player'])
CAT=['player','opponent','venue']
NUM=['prior_matches','avg_runs_last_3','avg_runs_last_5','avg_runs_last_10',
     'avg_sr_last_3','avg_sr_last_5','avg_sr_last_10','career_avg_before','career_sr_before']
FEATURES=CAT+NUM
# Chronological split on distinct dates to avoid same-day leakage.
dates=sorted(df.date.unique())
cutoff=dates[int(len(dates)*0.8)]
train=df[df.date<cutoff]; test=df[df.date>=cutoff]
if len(train)<100 or len(test)<30: raise RuntimeError('Not enough historical data for train/test split')

def pipeline(estimator):
    pre=ColumnTransformer([
        ('cat',OneHotEncoder(handle_unknown='ignore'),CAT),
        ('num',SimpleImputer(strategy='median'),NUM)
    ])
    return Pipeline([('preprocess',pre),('model',estimator)])
models={'Ridge':pipeline(Ridge(alpha=20)),
        'Random Forest':pipeline(RandomForestRegressor(n_estimators=100,min_samples_leaf=5,max_features=0.8,n_jobs=-1,random_state=42))}
results={}
for name,m in models.items():
    m.fit(train[FEATURES],train.runs)
    p=np.maximum(0,m.predict(test[FEATURES]))
    results[name]={'MAE':round(float(mean_absolute_error(test.runs,p)),2),
                   'RMSE':round(float(np.sqrt(mean_squared_error(test.runs,p))),2),
                   'R2':round(float(r2_score(test.runs,p)),3)}
    print(name,results[name])
    models[name]=m
best=min(results,key=lambda n:results[n]['MAE'])
joblib.dump(models[best],OUT/'model.joblib')
meta={'model':best,'results':results,'test_start':str(pd.Timestamp(cutoff).date()),
      'training_rows':len(train),'test_rows':len(test),'features':FEATURES,
      'data_end':str(df.date.max().date()),'source':'Cricsheet IPL JSON'}
(OUT/'model_metrics.json').write_text(json.dumps(meta,indent=2))
print('Saved model.joblib and model_metrics.json. Selected:',best)
