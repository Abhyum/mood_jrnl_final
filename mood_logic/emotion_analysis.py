from transformers import pipeline
import torch
from datetime import datetime  # Required for timestamp

def get_pipes():
    # Load emotion classifier
    emotion_pipe = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        device=-1,  # Force CPU
        framework="pt"  # Explicitly use PyTorch
    )

    # Load toxicity classifier
    tox_pipe = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        device=-1,  # Force CPU
        framework="pt"
    )

    return emotion_pipe, tox_pipe

def process_text(emotion_pipe, tox_pipe, text):
    # Analyze emotion
    emotion_result = emotion_pipe(text)[0]

    # Analyze toxicity
    toxicity_result = tox_pipe(text)[0]

    return {
        "emotion_label": emotion_result["label"],
        "emotion_score": emotion_result["score"],
        "toxicity_label": toxicity_result["label"],
        "toxicity_score": toxicity_result["score"],
        "timestamp": datetime.now().isoformat()  # Include timestamp
    }
