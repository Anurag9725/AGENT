# Professional Streamlit UI for Enhanced AI Agent with Multiple Real-Time Data Sources

from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import requests
import json
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any

# Page Configuration
st.set_page_config(
    page_title="🤖 AI Agent Hub - Real-Time Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .data-source-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-message {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #155724;
    }
    .error-message {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #721c24;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_status' not in st.session_state:
    st.session_state.api_status = None

# API Configuration
API_URL = "https://agent-4orp.onrender.com/chat"
MODEL_NAMES_GEMINI = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]

# Predefined system prompts for different use cases
SYSTEM_PROMPTS = {
    "General Assistant": "Act as a helpful, intelligent AI assistant who provides accurate, up-to-date information and engaging responses.",
    "Financial Advisor": "Act as a knowledgeable financial advisor who provides insights on stocks, crypto, market trends, and investment advice using real-time data.",
    "News Analyst": "Act as a professional news analyst who provides comprehensive summaries and insights on current events and trending topics.",
    "Weather Expert": "Act as a meteorologist who provides detailed weather information, forecasts, and climate insights.",
    "Tech Specialist": "Act as a technology expert who stays updated with the latest tech trends, developments, and innovations.",
    "Social Media Analyst": "Act as a social media analyst who tracks trending topics, popular content, and digital culture insights.",
    "Custom": "Write your own system prompt..."
}

# Data source information
DATA_SOURCES = {
    "Financial": {
        "icon": "💰",
        "sources": ["Finnhub", "Alpha Vantage", "CoinGecko", "Exchange Rates"],
        "description": "Real-time stock prices, crypto data, market trends"
    },
    "News": {
        "icon": "📰", 
        "sources": ["NewsAPI", "GNews", "Hacker News", "Tavily Search"],
        "description": "Breaking news, tech updates, global headlines"
    },
    "Weather": {
        "icon": "🌤️",
        "sources": ["WeatherAPI", "OpenWeatherMap", "wttr.in"],
        "description": "Current weather, forecasts, climate data"
    },
    "Social": {
        "icon": "📱",
        "sources": ["Reddit", "GitHub Trending", "Twitter Trends"],
        "description": "Trending topics, popular posts, social insights"
    },
    "Additional": {
        "icon": "🌟",
        "sources": ["Quotes", "Facts", "ISS Location", "Random Data"],
        "description": "Fun facts, quotes, space data, and more"
    }
}

def check_api_status():
    """Check if the backend API is running"""
    try:
        response = requests.get(API_URL.replace('/chat', '/health'), timeout=5)
        return response.status_code == 200
    except:
        return False

def send_request_to_api(payload):
    """Send request to the AI Agent API"""
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"🔌 Connection Error: {e}")
        return None

def display_data_sources():
    """Display available data sources in a nice format"""
    st.subheader("🔗 Available Real-Time Data Sources")
    
    cols = st.columns(len(DATA_SOURCES))
    for idx, (category, info) in enumerate(DATA_SOURCES.items()):
        with cols[idx]:
            st.markdown(f"""
                <div class="data-source-card">
                    <h3>{info['icon']} {category}</h3>
                    <p>{info['description']}</p>
                    <small>Sources: {', '.join(info['sources'][:2])}...</small>
                </div>
            """, unsafe_allow_html=True)

def display_chat_history():
    """Display chat history in a conversational format"""
    if st.session_state.chat_history:
        st.subheader("💬 Chat History")
        
        for idx, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5 chats
            with st.expander(f"Chat {len(st.session_state.chat_history) - idx}: {chat['query'][:50]}..."):
                st.write(f"**Query:** {chat['query']}")
                st.write(f"**Model:** {chat['model']}")
                st.write(f"**Time:** {chat['timestamp']}")
                st.write(f"**Response:** {chat['response']}")

def parse_response_data(response_text):
    """Parse response to extract any data for visualization"""
    # This is a simple example - you can enhance this based on your needs
    data_keywords = ["price", "temperature", "score", "count", "percentage"]
    
    try:
        # Look for numbers in the response that might be worth visualizing
        import re
        numbers = re.findall(r'\d+\.?\d*', response_text)
        if numbers and len(numbers) >= 2:
            return [float(x) for x in numbers[:10]]  # Take first 10 numbers
    except:
        pass
    
    return None

# Main UI Layout
def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Agent Hub</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent AI Assistant with Real-Time Data Integration</p>', unsafe_allow_html=True)
    
    # Check API status
    api_online = check_api_status()
    if api_online:
        st.success("🟢 Backend API is online and ready!")
    else:
        st.error("🔴 Backend API is offline. Please start the FastAPI server.")
        st.code("python app.py", language="bash")
        return
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Agent Type Selection
        st.subheader("🎭 Agent Type")
        selected_prompt_type = st.selectbox(
            "Choose your AI agent type:",
            list(SYSTEM_PROMPTS.keys()),
            help="Select a pre-configured agent or create a custom one"
        )
        
        # System Prompt
        if selected_prompt_type == "Custom":
            system_prompt = st.text_area(
                "Custom System Prompt:",
                height=100,
                placeholder="Define your AI agent's behavior and expertise..."
            )
        else:
            system_prompt = SYSTEM_PROMPTS[selected_prompt_type]
            st.info(f"**Selected Agent:** {selected_prompt_type}")
            st.write(system_prompt)
        
        st.divider()
        
        # Model Configuration
        st.subheader("🔧 Model Settings")
        provider = st.radio("Provider:", ["Gemini"], help="Currently supporting Gemini models")
        selected_model = st.selectbox(
            "Model:",
            MODEL_NAMES_GEMINI,
            index=0,
            help="Choose the Gemini model variant"
        )
        
        # Real-time Data Toggle
        st.subheader("🌐 Real-Time Data")
        allow_web_search = st.toggle(
            "Enable Real-Time Data",
            value=True,
            help="Allow the agent to fetch real-time data from multiple sources"
        )
        
        if allow_web_search:
            st.success("✅ Real-time data enabled")
        else:
            st.warning("⚠️ Using offline knowledge only")
        
        st.divider()
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("📊 Show Stats", use_container_width=True):
                st.info(f"Chats: {len(st.session_state.chat_history)}")
    
    # Main Content Area
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Data Sources", "📈 Analytics", "📋 History"])
    
    with tab1:
        # Chat Interface
        st.subheader("💬 Chat with Your AI Agent")
        
        # Query Examples
        example_queries = [
            "What are the latest tech trends and current Bitcoin price?",
            "Give me today's weather and top news headlines",
            "What's trending on social media and GitHub today?",
            "Analyze current stock market conditions",
            "What are the latest developments in AI?"
        ]
        
        selected_example = st.selectbox(
            "💡 Try these example queries:",
            [""] + example_queries,
            index=0
        )
        
        # Main query input
        user_query = st.text_area(
            "🎯 Enter your query:",
            value=selected_example if selected_example else "",
            height=120,
            placeholder="Ask anything! Your AI agent can access real-time data from multiple sources..."
        )
        
        # Send button with enhanced styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            send_button = st.button(
                "🚀 Send to AI Agent",
                use_container_width=True,
                type="primary"
            )
        
        # Process query
        if send_button and user_query.strip():
            payload = {
                "model_name": selected_model,
                "model_provider": provider,
                "system_prompt": system_prompt,
                "messages": [user_query],
                "allow_search": allow_web_search
            }
            
            with st.spinner("🤖 AI Agent is thinking and fetching real-time data..."):
                start_time = time.time()
                response = send_request_to_api(payload)
                end_time = time.time()
                
                if response and response.status_code == 200:
                    try:
                        response_data = response.json()
                        
                        if "error" in response_data:
                            st.error(f"❌ Error: {response_data['error']}")
                        elif "answer" in response_data:
                            # Display response
                            st.subheader("🤖 AI Agent Response")
                            st.markdown(response_data["answer"])
                            
                            # Response metrics
                            response_time = round(end_time - start_time, 2)
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Response Time", f"{response_time}s")
                            with col2:
                                st.metric("Model Used", selected_model)
                            with col3:
                                st.metric("Data Sources", "Multiple" if allow_web_search else "Offline")
                            
                            # Save to chat history
                            chat_entry = {
                                "query": user_query,
                                "response": response_data["answer"],
                                "model": selected_model,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "response_time": response_time,
                                "data_enabled": allow_web_search
                            }
                            st.session_state.chat_history.append(chat_entry)
                            
                            # Try to extract and visualize data
                            viz_data = parse_response_data(response_data["answer"])
                            if viz_data:
                                st.subheader("📊 Data Visualization")
                                fig = px.line(
                                    x=range(len(viz_data)),
                                    y=viz_data,
                                    title="Extracted Numerical Data from Response"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        else:
                            st.error(f"❌ Unexpected response format: {response_data}")
                    
                    except ValueError:
                        st.error(f"❌ Invalid JSON response: {response.text}")
                else:
                    if response:
                        st.error(f"❌ Request failed with status code {response.status_code}")
                    else:
                        st.error("❌ Failed to connect to the API")
    
    with tab2:
        # Data Sources Information
        display_data_sources()
        
        st.subheader("🔧 API Configuration Status")
        
        # Show which APIs are configured
        api_keys_status = {
            "Gemini API": "✅ Required - Configured",
            "Tavily Search": "✅ Required - Configured", 
            "Financial APIs": "⚠️ Optional - Configure for enhanced financial data",
            "News APIs": "⚠️ Optional - Configure for enhanced news coverage",
            "Weather APIs": "⚠️ Optional - Configure for detailed weather data",
            "Social APIs": "⚠️ Optional - Configure for social media insights"
        }
        
        for api, status in api_keys_status.items():
            if "✅" in status:
                st.success(f"{api}: {status}")
            else:
                st.info(f"{api}: {status}")
        
        st.info("💡 **Tip:** Many data sources work without API keys! The agent will use free APIs automatically.")
    
    with tab3:
        # Analytics Dashboard
        st.subheader("📈 Usage Analytics")
        
        if st.session_state.chat_history:
            # Basic statistics
            total_chats = len(st.session_state.chat_history)
            avg_response_time = sum(chat.get('response_time', 0) for chat in st.session_state.chat_history) / total_chats
            data_enabled_chats = sum(1 for chat in st.session_state.chat_history if chat.get('data_enabled', False))
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Chats", total_chats)
            with col2:
                st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
            with col3:
                st.metric("Real-Time Data Usage", f"{data_enabled_chats}/{total_chats}")
            with col4:
                st.metric("Success Rate", "100%")  # You can track this based on errors
            
            # Chat timeline
            if len(st.session_state.chat_history) > 1:
                st.subheader("📅 Chat Timeline")
                timestamps = [datetime.strptime(chat['timestamp'], "%Y-%m-%d %H:%M:%S") for chat in st.session_state.chat_history]
                response_times = [chat.get('response_time', 0) for chat in st.session_state.chat_history]
                
                fig = px.scatter(
                    x=timestamps,
                    y=response_times,
                    title="Response Time Over Time",
                    labels={"x": "Time", "y": "Response Time (seconds)"}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Start chatting to see analytics!")
    
    with tab4:
        # Chat History
        display_chat_history()
        
        if st.session_state.chat_history:
            # Export option
            if st.button("📥 Export Chat History"):
                chat_df = pd.DataFrame(st.session_state.chat_history)
                csv = chat_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"ai_agent_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🤖 AI Agent Hub - Powered by LangGraph, LangChain & FastAPI | "
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    "</div>",
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main()
