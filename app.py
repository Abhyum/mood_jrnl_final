import streamlit as st
st.set_page_config(page_title="🌱 Mood & Emotion Journal", layout="centered")

# Initialize session state keys early
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

import os
from datetime import datetime
import pandas as pd

from database.db import (
    create_tables,
    register_user,
    login_user,
    log_user_prompt,
    get_user_logs
)
from mood_logic.emotion_analysis import get_pipes, process_text
from mood_logic.graph import build_graph, get_strategies_from_graph
from mood_logic.forecast import forecast_mood
from llm.gemini import configure_gemini, get_llm_suggestions

# Setup
create_tables()
configure_gemini()
graph = build_graph()
emotion_pipe, tox_pipe = get_pipes()

# 🔧 TEMPORARY LOGIN BYPASS — uncomment during development
# st.session_state["user_id"] = 1
