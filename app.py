import os
from datetime import datetime
import streamlit as st
import pandas as pd

from database.db import create_tables, register_user, login_user, log_user_prompt, get_user_logs
from mood_logic.emotion_analysis import get_pipes, process_text
from mood_logic.graph import build_graph, get_strategies_from_graph
from mood_logic.forecast import forecast_mood
from llm.gemini import configure_gemini, get_llm_suggestions

# Setup
create_tables()
configure_gemini()
graph = build_graph()
emotion_pipe, tox_pipe = get_pipes()

# Debug login session
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
    st.write("🔄 Session state 'user_id' initialized.")

# TEMPORARY DEBUG BYPASS — remove this line in production
# st.session_state["user_id"] = 1

# LOGIN / REGISTER
if st.session_state["user_id"] is None:
    st.set_page_config(page_title="🔐 Login or Register", layout="centered")
    st.title("🔐 Login or Register")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            st.write("🔍 Checking credentials...")
            user_id = login_user(email, password)
            if user_id:
                st.success("✅ Login successful!")
                st.session_state["user_id"] = user_id
                st.experimental_rerun()
            else:
                st.error("❌ Invalid credentials.")

    with tab2:
        new_email = st.text_input("New Email")
        new_password = st.text_input("New Password", type="password")
        if st.button("Register"):
            if register_user(new_email, new_password):
                st.success("🎉 Registered! Please login now.")
            else:
                st.error("🚫 Email already registered.")
    st.stop()

# MAIN APP
st.set_page_config(page_title="🌱 Mood & Emotion Journal", layout="centered")
st.title("🌱 Mood & Emotion Journal")

with st.form(key="journal_form"):
    user_text = st.text_area("How are you feeling today?", height=100)
    submitted = st.form_submit_button(label="Analyze & Log")

    if submitted and user_text.strip():
        try:
            result = process_text(emotion_pipe, tox_pipe, user_text)
            emotion = result["emotion_label"]
            toxicity_score = result["toxicity_score"]
            timestamp = result["timestamp"]

            strategies = get_strategies_from_graph(emotion, graph)
            suggestion = get_llm_suggestions(emotion, strategies, user_text)

            log_user_prompt(
                st.session_state["user_id"],
                user_text,
                emotion,
                suggestion,
                toxicity_score,
                timestamp
            )

            result["suggestion"] = suggestion
            st.session_state["last_entry"] = result

        except Exception as e:
            st.error(f"🚨 Error while processing: {e}")
            st.stop()

# SHOW ANALYSIS
if "last_entry" in st.session_state:
    result = st.session_state["last_entry"]
    st.subheader("Your Analysis:")
    st.write(f"**Detected Emotion:** :blue[{result['emotion_label'].capitalize()}]")
    st.write(f"**Toxicity Score:** {result['toxicity_score']:.2f}")

    strategies = get_strategies_from_graph(result['emotion_label'], graph)
    if strategies:
        st.write(f"**Coping Strategies:** {', '.join(strategies)}")
    else:
        st.write("**Coping Strategies:** No direct strategies found.")

    st.info(f"💡 Suggestions: {result['suggestion']}")

# SHOW MOOD HISTORY
st.subheader("📖 Your Mood History")
history = get_user_logs(st.session_state["user_id"])
if not history.empty:
    st.dataframe(history[['timestamp', 'prompt', 'emotion', 'toxicity_score']])
else:
    st.write("*No entries found.*")
