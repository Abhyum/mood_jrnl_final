from transformers import pipeline

# Load models on CPU explicitly to avoid device errors
def get_pipes():
    emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=-1)
    tox_pipe = pipeline("text-classification", model="unitary/toxic-bert", device=-1)
    return emotion_pipe, tox_pipe

# Function to process text for both emotion and toxicity
def process_text(text, emotion_pipe, tox_pipe):
    emotion_result = emotion_pipe(text)[0]  # Single text input
    toxicity_result = tox_pipe(text)[0]
    return {
        "emotion_label": emotion_result["label"],
        "emotion_score": emotion_result["score"],
        "toxicity_label": toxicity_result["label"],
        "toxicity_score": toxicity_result["score"]
    }
