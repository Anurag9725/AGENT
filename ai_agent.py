# Enhanced AI Agent with LangGraph ReactAgent and Multi-Source Data Aggregation

import os
import requests
import json
from datetime import datetime
from typing import Dict, Any
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import tool

# Load API Keys from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
GNEWS_API_KEY = os.environ.get("GNEWS_API_KEY")
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

class EnhancedDataAggregator:
    def __init__(self):
        self.tavily_search = TavilySearch(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

    @tool
    async def fetch_financial_data(self, symbol: str = "AAPL") -> Dict[str, Any]:
        financial_data = {}

        if FINNHUB_API_KEY:
            try:
                url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
                r = requests.get(url, timeout=10)
                financial_data['finnhub'] = r.json() if r.status_code == 200 else {}
            except Exception as e:
                financial_data['finnhub_error'] = str(e)

        if ALPHA_VANTAGE_API_KEY:
            try:
                url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
                r = requests.get(url, timeout=10)
                financial_data['alpha_vantage'] = r.json() if r.status_code == 200 else {}
            except Exception as e:
                financial_data['alpha_vantage_error'] = str(e)

        try:
            url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,cardano&vs_currencies=usd&include_24hr_change=true"
            r = requests.get(url, timeout=10)
            financial_data['crypto'] = r.json() if r.status_code == 200 else {}
        except Exception as e:
            financial_data['crypto_error'] = str(e)

        return financial_data

    @tool
    async def fetch_news_data(self, query: str = "technology") -> Dict[str, Any]:
        news_data = {}

        if NEWS_API_KEY:
            try:
                url = f"https://newsapi.org/v2/top-headlines?q={query}&apiKey={NEWS_API_KEY}&pageSize=5"
                r = requests.get(url, timeout=10)
                news_data['newsapi'] = r.json() if r.status_code == 200 else {}
            except Exception as e:
                news_data['newsapi_error'] = str(e)

        if GNEWS_API_KEY:
            try:
                url = f"https://gnews.io/api/v4/search?q={query}&token={GNEWS_API_KEY}&max=5"
                r = requests.get(url, timeout=10)
                news_data['gnews'] = r.json() if r.status_code == 200 else {}
            except Exception as e:
                news_data['gnews_error'] = str(e)

        try:
            url = "https://hacker-news.firebaseio.com/v0/topstories.json"
            r = requests.get(url, timeout=10)
            story_ids = r.json()[:5] if r.status_code == 200 else []
            stories = []
            for sid in story_ids:
                story_url = f"https://hacker-news.firebaseio.com/v0/item/{sid}.json"
                sr = requests.get(story_url, timeout=5)
                if sr.status_code == 200:
                    stories.append(sr.json())
            news_data['hackernews'] = stories
        except Exception as e:
            news_data['hackernews_error'] = str(e)

        if self.tavily_search:
            try:
                tavily_result = self.tavily_search.invoke({"query": query})
                news_data['tavily'] = tavily_result
            except Exception as e:
                news_data['tavily_error'] = str(e)

        return news_data

    @tool
    async def fetch_weather_data(self, city: str = "New York") -> Dict[str, Any]:
        weather_data = {}

        if WEATHER_API_KEY:
            try:
                url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
                r = requests.get(url, timeout=10)
                weather_data['weatherapi'] = r.json() if r.status_code == 200 else {}
            except Exception as e:
                weather_data['weatherapi_error'] = str(e)

        if OPENWEATHER_API_KEY:
            try:
                url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
                r = requests.get(url, timeout=10)
                weather_data['openweather'] = r.json() if r.status_code == 200 else {}
            except Exception as e:
                weather_data['openweather_error'] = str(e)

        return weather_data

# Instantiate tools from the data aggregator
aggregator = EnhancedDataAggregator()
tools = [
    aggregator.fetch_financial_data,
    aggregator.fetch_news_data,
    aggregator.fetch_weather_data,
]

# Setup Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

# Create a React agent using LangGraph
agent_runnable = create_react_agent(
    llm=llm,
    tools=tools,
    prompt="You are a helpful assistant that fetches real-time data based on user queries."
)

graph_builder = StateGraph()
graph_builder.add_node("agent", agent_runnable)
graph_builder.set_entry_point("agent")
graph_builder.set_finish_point(END)
graph_builder.add_edge("agent", END)

graph = graph_builder.compile()

# Run the graph
async def run_query(query: str):
    result = await graph.ainvoke({"messages": [HumanMessage(content=query)]})
    return result

# Example usage (uncomment for test)
# asyncio.run(run_query("What's the latest crypto market news and New York weather?"))
