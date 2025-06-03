# app.py - FastAPI Backend with Multiple Real-Time Data Sources

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional
import os
import json
from datetime import datetime
import time
import logging
import requests
import asyncio

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  # This loads your .env file automatically

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Agent API",
    description="Enhanced AI Agent with Multiple Real-Time Data Sources",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys from .env file
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# Financial Data APIs
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")

# News APIs
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
GNEWS_API_KEY = os.environ.get("GNEWS_API_KEY")

# Weather APIs
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

# Other APIs
EXCHANGERATE_API_KEY = os.environ.get("EXCHANGERATE_API_KEY")
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")

logger.info(f"GEMINI_API_KEY configured: {bool(GEMINI_API_KEY)}")
logger.info(f"TAVILY_API_KEY configured: {bool(TAVILY_API_KEY)}")
logger.info(f"NEWS_API_KEY configured: {bool(NEWS_API_KEY)}")
logger.info(f"WEATHER_API_KEY configured: {bool(WEATHER_API_KEY)}")
logger.info(f"FINNHUB_API_KEY configured: {bool(FINNHUB_API_KEY)}")

# Pydantic Models
class ChatRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to use")
    model_provider: str = Field(default="Gemini", description="Model provider")
    system_prompt: str = Field(..., description="System prompt for the AI agent")
    messages: List[str] = Field(..., description="List of user messages")
    allow_search: bool = Field(default=True, description="Allow real-time data search")

    @validator('messages')
    def messages_must_not_be_empty(cls, v):
        if not v or all(not msg.strip() for msg in v):
            raise ValueError('Messages cannot be empty')
        return v

class ChatResponse(BaseModel):
    answer: str = Field(..., description="AI agent response")
    model_used: str = Field(..., description="Model that generated the response")
    timestamp: str = Field(..., description="Response timestamp")
    response_time: float = Field(..., description="Response time in seconds")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Health check timestamp")
    api_keys: Dict[str, bool] = Field(..., description="API key status")

# Try to import LangChain components
LANGCHAIN_AVAILABLE = False
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
    from langchain_tavily import TavilySearch
    LANGCHAIN_AVAILABLE = True
    logger.info("LangChain components loaded successfully")
except ImportError as e:
    logger.error(f"Failed to import LangChain components: {e}")

class EnhancedDataAggregator:
    """Enhanced data aggregator for multiple real-time data sources"""
    
    def __init__(self):
        self.tavily_search = None
        if TAVILY_API_KEY and LANGCHAIN_AVAILABLE:
            try:
                self.tavily_search = TavilySearch(
                    api_key=TAVILY_API_KEY,
                    max_results=5,
                    topic="news",
                    search_depth="advanced"
                )
            except Exception as e:
                logger.error(f"Failed to initialize Tavily search: {e}")
        
    async def fetch_financial_data(self, symbol: str = "AAPL") -> Dict[str, Any]:
        """Fetch real-time financial data"""
        financial_data = {}
        logger.info("Fetching financial data...")
        
        # CoinGecko - Crypto data (free API)
        try:
            url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,cardano&vs_currencies=usd&include_24hr_change=true"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                financial_data['crypto_prices'] = response.json()
                logger.info("✓ CoinGecko crypto data fetched")
        except Exception as e:
            logger.error(f"CoinGecko API error: {e}")
        
        # Exchange rates (free API)
        try:
            url = "https://api.exchangerate-api.com/v4/latest/USD"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                financial_data['exchange_rates'] = response.json()
                logger.info("✓ Exchange rates fetched")
        except Exception as e:
            logger.error(f"Exchange Rate API error: {e}")
        
        # Finnhub - Stock quotes (if API key available)
        if FINNHUB_API_KEY:
            try:
                url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    financial_data['stock_quote'] = response.json()
                    logger.info("✓ Finnhub stock data fetched")
            except Exception as e:
                logger.error(f"Finnhub API error: {e}")
        
        return financial_data
    
    async def fetch_news_data(self, query: str = "technology") -> Dict[str, Any]:
        """Fetch real-time news data"""
        news_data = {}
        logger.info("Fetching news data...")
        
        # Hacker News (free API)
        try:
            url = "https://hacker-news.firebaseio.com/v0/topstories.json"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                story_ids = response.json()[:3]  # Get top 3 stories
                stories = []
                for story_id in story_ids:
                    story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                    story_response = requests.get(story_url, timeout=5)
                    if story_response.status_code == 200:
                        stories.append(story_response.json())
                news_data['hackernews'] = stories
                logger.info("✓ Hacker News data fetched")
        except Exception as e:
            logger.error(f"Hacker News API error: {e}")
        
        # NewsAPI (if API key available)
        if NEWS_API_KEY:
            try:
                url = f"https://newsapi.org/v2/top-headlines?q={query}&apiKey={NEWS_API_KEY}&pageSize=3"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    news_data['newsapi'] = response.json()
                    logger.info("✓ NewsAPI data fetched")
            except Exception as e:
                logger.error(f"NewsAPI error: {e}")
        
        # Tavily Search
        if self.tavily_search:
            try:
                tavily_result = self.tavily_search.invoke({"query": query})
                news_data['tavily_search'] = tavily_result
                logger.info(f"✓ Tavily search completed: {tavily_result}")
            except Exception as e:
                logger.error(f"Tavily Search error: {e}")
        
        return news_data
    
    async def fetch_weather_data(self, city: str = "New York") -> Dict[str, Any]:
        """Fetch real-time weather data"""
        weather_data = {}
        logger.info("Fetching weather data...")
        
        # wttr.in (free weather API)
        try:
            url = f"http://wttr.in/{city}?format=j1"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                weather_data['wttr'] = response.json()
                logger.info("✓ wttr.in weather data fetched")
        except Exception as e:
            logger.error(f"wttr.in API error: {e}")
        
        # WeatherAPI (if API key available)
        if WEATHER_API_KEY:
            try:
                url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    weather_data['weatherapi'] = response.json()
                    logger.info("✓ WeatherAPI data fetched")
            except Exception as e:
                logger.error(f"WeatherAPI error: {e}")
        
        return weather_data
    
    async def fetch_social_data(self) -> Dict[str, Any]:
        """Fetch social media and trending data"""
        social_data = {}
        logger.info("Fetching social data...")
        
        # Reddit (free API)
        try:
            url = "https://www.reddit.com/r/all/hot.json?limit=3"
            headers = {'User-Agent': 'AI-Agent/1.0'}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                social_data['reddit_hot'] = response.json()
                logger.info("✓ Reddit data fetched")
        except Exception as e:
            logger.error(f"Reddit API error: {e}")
        
        # GitHub trending (free API)
        try:
            url = "https://api.github.com/search/repositories?q=created:>2024-01-01&sort=stars&order=desc&per_page=3"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                social_data['github_trending'] = response.json()
                logger.info("✓ GitHub trending data fetched")
        except Exception as e:
            logger.error(f"GitHub API error: {e}")
        
        return social_data
    
    async def fetch_additional_data(self) -> Dict[str, Any]:
        """Fetch additional real-time data"""
        additional_data = {}
        logger.info("Fetching additional data...")
        
        # Quote of the day
        try:
            url = "https://api.quotable.io/random"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                additional_data['quote_of_day'] = response.json()
                logger.info("✓ Quote fetched")
        except Exception as e:
            logger.error(f"Quotable API error: {e}")
        
        # ISS location
        try:
            url = "http://api.open-notify.org/iss-now.json"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                additional_data['iss_location'] = response.json()
                logger.info("✓ ISS location fetched")
        except Exception as e:
            logger.error(f"ISS API error: {e}")
        
        return additional_data
    
    async def aggregate_all_data(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Aggregate data from all sources based on query context"""
        logger.info(f"Aggregating data for query: {query[:50]}...")
        
        aggregated_data = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'data_sources_used': []
        }
        
        query_lower = query.lower()
        
        # Determine what data to fetch based on query
        fetch_financial = any(word in query_lower for word in ['stock', 'market', 'crypto', 'bitcoin', 'trading', 'finance', 'money', 'price'])
        fetch_news = any(word in query_lower for word in ['news', 'current', 'latest', 'today', 'happening', 'events'])
        fetch_weather = any(word in query_lower for word in ['weather', 'temperature', 'rain', 'sunny', 'climate'])
        fetch_social = any(word in query_lower for word in ['trending', 'popular', 'reddit', 'github', 'social'])
        
        # If no specific category detected, fetch news and additional data
        if not any([fetch_financial, fetch_news, fetch_weather, fetch_social]):
            fetch_news = True
        
        try:
            # Fetch relevant data
            if fetch_financial:
                aggregated_data['financial_data'] = await self.fetch_financial_data()
                aggregated_data['data_sources_used'].extend(['CoinGecko', 'ExchangeRate'])
                if FINNHUB_API_KEY:
                    aggregated_data['data_sources_used'].append('Finnhub')
            
            if fetch_news:
                aggregated_data['news_data'] = await self.fetch_news_data(query)
                aggregated_data['data_sources_used'].extend(['HackerNews'])
                if NEWS_API_KEY:
                    aggregated_data['data_sources_used'].append('NewsAPI')
                if TAVILY_API_KEY:
                    aggregated_data['data_sources_used'].append('Tavily')
            
            if fetch_weather:
                # Try to extract city from query, fallback to "New York"
                city = "New York"
                for word in query_lower.split():
                    if word in ["hyderabad", "delhi", "mumbai", "bangalore", "chennai", "kolkata"]:
                        city = word.capitalize()
                aggregated_data['weather_data'] = await self.fetch_weather_data(city)
                aggregated_data['data_sources_used'].append('wttr.in')
                if WEATHER_API_KEY:
                    aggregated_data['data_sources_used'].append('WeatherAPI')
            
            if fetch_social:
                aggregated_data['social_data'] = await self.fetch_social_data()
                aggregated_data['data_sources_used'].extend(['Reddit', 'GitHub'])
            
            # Always fetch some additional context
            aggregated_data['additional_data'] = await self.fetch_additional_data()
            aggregated_data['data_sources_used'].extend(['Quotable', 'ISS-API'])
            
            logger.info(f"✓ Data aggregation complete. Sources used: {aggregated_data['data_sources_used']}")
            
        except Exception as e:
            logger.error(f"Error aggregating data: {e}")
            aggregated_data['error'] = str(e)
        
        return aggregated_data

# Initialize the enhanced data aggregator
data_aggregator = EnhancedDataAggregator()

async def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    """Enhanced AI Agent with multiple real-time data sources"""
    try:
        logger.info(f"Processing request - Model: {llm_id}, Allow Search: {allow_search}")
        
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not configured")
        
        if not LANGCHAIN_AVAILABLE:
            raise ValueError("LangChain not available")
        
        llm = ChatGoogleGenerativeAI(
            model=llm_id, 
            google_api_key=GEMINI_API_KEY,
            temperature=0.7
        )
        
        user_query = query if isinstance(query, str) else " ".join(query)
        
        if allow_search:
            try:
                logger.info("🔍 Fetching real-time data from multiple sources...")
                realtime_data = await data_aggregator.aggregate_all_data(user_query)
                logger.info(f"✅ Real-time data fetched from: {realtime_data.get('data_sources_used', [])}")

                # Optionally, extract weather/news highlights for prompt
                weather_data = realtime_data.get("weather_data", {})
                news_data = realtime_data.get("news_data", {})
                tavily_data = news_data.get("tavily_search") if news_data else None

                weather_summary = ""
                if "wttr" in weather_data:
                    current = weather_data["wttr"].get("current_condition", [{}])[0]
                    weather_summary = (
                        f"Current weather:\n"
                        f"Temperature: {current.get('temp_C', 'N/A')}°C\n"
                        f"Feels Like: {current.get('FeelsLikeC', 'N/A')}°C\n"
                        f"Humidity: {current.get('humidity', 'N/A')}%\n"
                        f"Wind: {current.get('windspeedKmph', 'N/A')} km/h\n"
                        f"Sky: {current.get('weatherDesc', [{'value': 'N/A'}])[0]['value']}\n"
                    )

                tavily_summary = ""
                if tavily_data:
                    tavily_summary = f"Tavily News Results:\n{json.dumps(tavily_data, indent=2, default=str)}\n"

                prompt = (
                    f"{system_prompt}\n\n"
                    f"{weather_summary}"
                    f"{tavily_summary}"
                    f"Here's comprehensive real-time data to help answer the user's question:\n"
                    f"{json.dumps(realtime_data, indent=2, default=str)}\n\n"
                    f"User's question: {user_query}\n\n"
                    f"Please provide a helpful, informative response using the most relevant data from above. "
                    f"Focus on accuracy and cite specific data points when appropriate."
                )
            except Exception as e:
                logger.error(f"Error fetching real-time data: {e}")
                prompt = f"{system_prompt}\n\nUser's question: {user_query}"
        else:
            prompt = f"{system_prompt}\n\nUser's question: {user_query}"
        
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        
        result = response.content if hasattr(response, "content") else str(response)
        logger.info("✅ AI response generated successfully")
        
        return result
        
    except Exception as e:
        logger.error(f"AI Agent error: {e}")
        raise e

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        api_keys={
            "gemini": bool(GEMINI_API_KEY),
            "tavily": bool(TAVILY_API_KEY),
            "news_api": bool(NEWS_API_KEY),
            "weather_api": bool(WEATHER_API_KEY),
            "finnhub": bool(FINNHUB_API_KEY),
            "langchain_available": LANGCHAIN_AVAILABLE
        }
    )

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    start_time = time.time()
    
    try:
        logger.info(f"🚀 Chat request received: {request.model_name}")
        
        response = await get_response_from_ai_agent(
            llm_id=request.model_name,
            query=request.messages,
            allow_search=request.allow_search,
            system_prompt=request.system_prompt,
            provider=request.model_provider
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        logger.info(f"✅ Request completed in {response_time:.2f}s")
        
        return ChatResponse(
            answer=response,
            model_used=request.model_name,
            timestamp=datetime.now().isoformat(),
            response_time=response_time
        )
        
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api-status")
async def get_api_status():
    """Get detailed API configuration status"""
    return {
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "free_apis": {
            "coingecko": "✅ Available",
            "exchange_rates": "✅ Available", 
            "hacker_news": "✅ Available",
            "wttr_weather": "✅ Available",
            "reddit": "✅ Available",
            "github": "✅ Available",
            "quotable": "✅ Available",
            "iss_location": "✅ Available"
        },
        "premium_apis": {
            "gemini": "✅ Required" if GEMINI_API_KEY else "❌ Missing",
            "tavily": "✅ Configured" if TAVILY_API_KEY else "⚠️ Optional", 
            "news_api": "✅ Configured" if NEWS_API_KEY else "⚠️ Optional",
            "weather_api": "✅ Configured" if WEATHER_API_KEY else "⚠️ Optional",
            "finnhub": "✅ Configured" if FINNHUB_API_KEY else "⚠️ Optional"
        }
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("=== 🤖 AI Agent API Starting ===")
    logger.info(f"📁 Environment file loaded: .env")
    logger.info(f"🔑 GEMINI_API_KEY: {'✅ Configured' if GEMINI_API_KEY else '❌ Missing'}")
    logger.info(f"🔍 TAVILY_API_KEY: {'✅ Configured' if TAVILY_API_KEY else '⚠️  Optional'}")
    logger.info(f"📰 NEWS_API_KEY: {'✅ Configured' if NEWS_API_KEY else '⚠️  Optional'}")
    logger.info(f"🌤️  WEATHER_API_KEY: {'✅ Configured' if WEATHER_API_KEY else '⚠️  Optional'}")
    logger.info(f"💰 FINNHUB_API_KEY: {'✅ Configured' if FINNHUB_API_KEY else '⚠️  Optional'}")
    logger.info("🆓 Free APIs: CoinGecko, HackerNews, Reddit, GitHub, wttr.in, ISS, Quotes")
    logger.info("=== ✅ Startup Complete ===")

if __name__ == "__main__":
    import uvicorn
    
    if not GEMINI_API_KEY:
        print("❌ ERROR: GEMINI_API_KEY is required!")
        print("Add it to your .env file: GEMINI_API_KEY=your_api_key")
        exit(1)
    
    print("🚀 Starting AI Agent API Server...")
    print("📁 Loading from .env file...")
    print(f"🔑 Gemini API: {'✅' if GEMINI_API_KEY else '❌'}")
    print(f"🔍 Tavily API: {'✅' if TAVILY_API_KEY else '⚠️'}")
    print(f"📰 News API: {'✅' if NEWS_API_KEY else '⚠️'}")
    print(f"🌤️ Weather API: {'✅' if WEATHER_API_KEY else '⚠️'}")
    print("🆓 Free APIs: Always available!")
    
    uvicorn.run("app:app", host="127.0.0.1", port=9999, reload=True, log_level="info")
