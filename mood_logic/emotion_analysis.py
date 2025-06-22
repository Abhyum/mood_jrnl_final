from transformers import pipeline

# Initialize the emotion and toxicity pipelines
def get_pipes():
    emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=-1)
    tox_pipe = pipeline("text-classification", model="martin-ha/toxic-comment-model", device=-1)
    return emotion_pipe, tox_pipe

# Process input text to get emotion and toxicity predictions
def process_text(text, emotion_pipe, tox_pipe):
    emotion_result = emotion_pipe(text)
    toxicity_result = tox_pipe(text)

    top_emotion = emotion_result[0] if emotion_result else {"label": "Unknown", "score": 0}
    top_toxicity = toxicity_result[0] if toxicity_result else {"label": "Unknown", "score": 0}

    return {
        "emotion_label": top_emotion["label"],
        "emotion_score": round(top_emotion["score"], 2),
        "toxicity_label": top_toxicity["label"],
        "toxicity_score": round(top_toxicity["score"], 2)
    }
