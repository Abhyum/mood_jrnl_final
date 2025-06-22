from transformers import pipeline
import torch

def get_pipes():
    # Force CPU and PyTorch framework explicitly
    emotion_pipe = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        device=-1,
        framework="pt"  # explicitly state PyTorch backend
    )

    tox_pipe = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        device=-1,
        framework="pt"
    )

    return emotion_pipe, tox_pipe

def process_text(emotion_pipe, tox_pipe, text):
    emotion_result = emotion_pipe(text)[0]
    toxicity_result = tox_pipe(text)[0]

    return {
        "emotion_label": emotion_result["label"],
        "emotion_score": emotion_result["score"],
        "toxicity_label": toxicity_result["label"],
        "toxicity_score": toxicity_result["score"]
    }
