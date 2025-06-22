import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file")

def configure_gemini():
    genai.configure(api_key=api_key)
    return genai

def get_llm_suggestions(genai, emotion, strategies, user_text):
    try:
        model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")
        if strategies:
            strat_txt = "\n".join(f"- {s}" for s in strategies)
            prompt = (
                f"The user is feeling {emotion}.\n"
                f"Recommended strategies:\n{strat_txt}\n"
                f"User said: \"{user_text}\"\n"
                f"Give 2 personalized and simple suggestions to help them."
            )
        else:
            prompt = (
                f"The user is feeling {emotion}.\n"
                f"User said: \"{user_text}\"\n"
                f"Suggest 2 supportive and practical things they can do."
            )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini API failed: {e}"
