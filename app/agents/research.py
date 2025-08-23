from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

# Load TavilySearch environment variables from .env file
load_dotenv()

def create_research_agent(llm: BaseChatModel) -> CompiledStateGraph: 
    web_search = TavilySearch(max_results=3)
    research_agent = create_react_agent(
        model=llm,
        tools=[web_search],
        prompt=(
            "You are a research agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with research-related tasks, DO NOT do any math\n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
            "- Use only tool 'research_agent' if necessary. Do NOT use other tool."
        ),
        name="research_agent",
    )
    return research_agent
