# CricketIQ

Educational Streamlit website for IPL batting analytics and experimental runs prediction.

## Colab workflow
1. Upload `prepare_data.py` and `train_model.py` into Colab (or run the included notebook).
2. Run `!pip install -r requirements.txt` and then `!python prepare_data.py` and `!python train_model.py`.
3. Download `ipl_batting_features.csv`, `model.joblib`, `model_metrics.json`.
4. Put all files in the same GitHub repository as `app.py` and `requirements.txt`.
5. Deploy on Streamlit Community Cloud with entrypoint `app.py`.

Data: https://cricsheet.org/downloads/ (IPL JSON). Real historical match records, not live fixtures. This project does not include official player biographies or photos.

Note: Source records may include name variations, match-data corrections, and incomplete coverage. Batting totals in the dashboard exclude each player's first three appearances to enable lagged features. These are not official career totals. Model performance must be assessed using held-out chronological matches.
