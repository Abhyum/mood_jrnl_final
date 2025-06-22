import os
import streamlit as st
import pandas as pd
from datetime import datetime

from mood_logic.emotion_analysis import get_pipes, process_input
from mood_logic.logger import log_mood, get_history
from mood_logic.graph import build_graph, get_strategies_from_graph
from mood_logic.forecast import forecast_mood
from llm.gemini import configure_gemini, get_llm_suggestions

# --- Configure Gemini ---
try:
    genai = configure_gemini()
    GEMINI_ENABLED = True
except Exception as e:
    GEMINI_ENABLED = False
    st.error(f"❌ Gemini setup failed: {e}")

# --- Load ML Pipelines ---
emotion_pipe, tox_pipe = get_pipes()

# --- Load Strategy Graph ---
graph = build_graph()

# --- Streamlit App Setup ---
st.set_page_config(page_title="🌱 Mood & Emotion Journal", layout="centered")
st.title("🌱 Mood & Emotion Journal")

# --- Mood Input Form ---
with st.form(key="journal_form"):
    user_text = st.text_area("How are you feeling today?", height=100)
    submitted = st.form_submit_button(label="Analyze & Log")
    if submitted and user_text.strip():
        result = process_input(user_text, emotion_pipe, tox_pipe)
        score = 1.0 - result['toxicity_score']
        log_mood(result['emotion'], score, result['timestamp'])
        st.session_state["last_entry"] = result

# --- Show Analysis Result ---
if "last_entry" in st.session_state:
    result = st.session_state["last_entry"]
    st.subheader("Your Analysis:")
    st.write(f"**Detected Emotion:** :blue[{result['emotion'].capitalize()}]")
    st.write(f"**Toxicity Score:** {result['toxicity_score']:.2f}")

    strategies = get_strategies_from_graph(result["emotion"], graph)
    if strategies:
        st.write(f"**Coping Strategies:** {', '.join(strategies)}")
    else:
        st.write("**Coping Strategies:** No direct strategies found.")

    if GEMINI_ENABLED:
        suggestions = get_llm_suggestions(genai, result["emotion"], strategies, result["text"])
        st.info(f"💡 Suggestions: {suggestions}")
    else:
        st.warning("⚠️ Gemini LLM not enabled. No suggestions available.")

# --- Show Mood Logs ---
st.subheader("📖 Past Mood Logs")
history = get_history()
if len(history) > 0:
    st.dataframe(history.sort_values("timestamp", ascending=False), use_container_width=True)
else:
    st.write("*No logs yet. Your first mood entry will appear here.*")

# --- Forecast Mood if Enough Data ---
if len(history) >= 2:
    st.subheader("📊 Mood Score Over Time")
    history['timestamp'] = pd.to_datetime(history['timestamp'])
    st.line_chart(history.set_index('timestamp')['score'])
    forecast = forecast_mood()
    if forecast is not None:
        st.write("🔮 *Mood Forecast (next 7 days)*")
        st.line_chart(forecast.set_index('ds')['yhat'])
else:
    st.write("*(Log at least two moods to see trends!)*")

st.caption("🛡️ Your data is stored locally and privately in your device.")
