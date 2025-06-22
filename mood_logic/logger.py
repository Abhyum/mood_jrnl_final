import os
import pandas as pd

DATA_PATH = 'data/mood_logs.csv'
os.makedirs('data', exist_ok=True)

def log_mood(emotion, score, timestamp):
    df = pd.DataFrame([{'timestamp': timestamp, 'emotion': emotion, 'score': score}])
    df.to_csv(DATA_PATH, mode='a', index=False, header=not os.path.exists(DATA_PATH))

def get_history():
    if not os.path.exists(DATA_PATH) or os.stat(DATA_PATH).st_size == 0:
        return pd.DataFrame(columns=['timestamp', 'emotion', 'score'])
    return pd.read_csv(DATA_PATH)