# if you dont use pipenv uncomment the following:
from dotenv import load_dotenv
load_dotenv()

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

#Step1: Setup UI with streamlit (model provider, model, system prompt, web_search, query)
import streamlit as st

st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
st.title("AI Chatbot Agents")
st.write("Create and Interact with the AI Agents!")

system_prompt = st.text_area("Define your AI Agent: ", height=70, placeholder="Type your system prompt here...")

MODEL_NAMES_GEMINI = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]

provider = st.radio("Select Provider:", ("Gemini",))

# Since we're only using Gemini now
selected_model = st.selectbox("Select Gemini Model:", MODEL_NAMES_GEMINI)

allow_web_search = st.checkbox("Allow Web Search")

user_query = st.text_area("Enter your query: ", height=150, placeholder="Ask Anything!")

API_URL = "https://agent-u60w.onrender.com/chat"


if st.button("Ask Agent!"):
    if user_query.strip():
        import requests

        payload = {
            "model_name": selected_model,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": allow_web_search
        }

        try:
            response = requests.post(API_URL, json=payload, timeout=60)
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")
            st.stop()

        if response.status_code == 200:
            try:
                response_data = response.json()
            except ValueError:
                st.error(f"Expected JSON, but got: {response.text}")
                st.stop()

            # Safely check for "error" in response
            if isinstance(response_data, dict) and "error" in response_data:
                st.error(response_data["error"])
            elif isinstance(response_data, dict) and "answer" in response_data:
                st.subheader("Agent Response")
                st.markdown(response_data["answer"])
            else:
                st.error(f"Unexpected response format: {response_data}")
        else:
            st.error(f"Request failed with status code {response.status_code}")
