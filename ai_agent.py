# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

# Step1: Setup API Keys for Gemini and Tavily
import os

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# Step2: Setup LLM & Tools
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage

gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
search_tool = TavilySearch(max_results=2, topic="general")

system_prompt = "Act as an AI chatbot who is smart and friendly"

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    llm = ChatGoogleGenerativeAI(model=llm_id, google_api_key=GEMINI_API_KEY)
    # Explicitly invoke Tavily tool if allowed
    if allow_search:
        tavily_result = search_tool.invoke({"query": query if isinstance(query, str) else query[0]})
        print("Tavily Search Output:\n", tavily_result)
        # Pass Tavily result as context to Gemini
        prompt = (
            f"{system_prompt}\n\n"
            f"Here are some recent search results you must use to answer the user's question:\n"
            f"{tavily_result}\n\n"
            f"User's question: {query if isinstance(query, str) else query[0]}"
        )
    else:
        prompt = f"{system_prompt}\n\nUser's question: {query if isinstance(query, str) else query[0]}"
    # Send prompt to Gemini
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    print("\nGemini LLM Response:\n", response.content if hasattr(response, "content") else response)
    return response.content if hasattr(response, "content") else response

# Example usage
if __name__ == "__main__":
    question = "What are the top shares in 2025?"
    answer = get_response_from_ai_agent(
        llm_id="gemini-1.5-flash",
        query=question,
        allow_search=True,
        system_prompt=system_prompt,
        provider="Gemini"
    )
