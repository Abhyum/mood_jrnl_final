from prophet import Prophet
import pandas as pd
import os

def forecast_mood(data_path='data/mood_logs.csv'):
    if not os.path.exists(data_path) or os.stat(data_path).st_size == 0:
        return None
    df = pd.read_csv(data_path)
    if len(df) < 2:
        return None
    df['ds'] = pd.to_datetime(df['timestamp'])
    df['y'] = df['score']
    model = Prophet()
    model.fit(df[['ds', 'y']])
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]