from transformers import pipeline
from datetime import datetime

def get_pipes():
    emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    tox_pipe = pipeline("text-classification", model="unitary/toxic-bert")
    return emotion_pipe, tox_pipe

def process_input(text, emotion_pipe, tox_pipe):
    emotion = emotion_pipe(text)[0]["label"]
    tox_score = tox_pipe(text)[0]["score"]
    timestamp = datetime.now().isoformat()
    return {"text": text, "emotion": emotion, "toxicity_score": tox_score, "timestamp": timestamp}