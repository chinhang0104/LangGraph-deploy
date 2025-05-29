import os
from dotenv import load_dotenv
import getpass
import json
from pydantic import BaseModel
from typing import Annotated
from typing_extensions import TypedDict
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Load environment variables from .env file
load_dotenv()

# Check the environment variables
env_vars = ['TAVILY_API_KEY', 'GROQ_API_KEY']
for env_var in env_vars:
    if env_var not in os.environ:
        raise RuntimeError(f"The required environment variable '{env_var}' is not set.")

app = FastAPI()

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# tool
# Loading up search engine 
search = TavilySearchResults(max_results=2)
tools = [search]

# llm
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

# tell the LLM which tools it can call
llm_with_tools = llm.bind_tools(tools)

# The class is a dict to store messages returned by llm
# The key "messages" has a value of list
# The update rule of "messages" value is appending the list instead of overwritting it
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        # dict of tools
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        # get the most recent message in the state dict
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        # calls tools if the message contains tool_calls
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

tool_node = BasicToolNode(tools=[search])
graph_builder.add_node("tools", tool_node)

# Define the conditional_edges
def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)

# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# MemorySaver helps remember chat history
# Use SqliteSaver or PostgresSaver and connect a database for persistent store. 
memory = MemorySaver() # In-Memory Saver, for demo only
graph = graph_builder.compile(checkpointer=memory)

# Set up system message, set None to disable
system_message = '''You are a helpful customer support assistant.
                    Whenever you do not know the answer or need up-to-date information, 
                    perform a web search to find accurate and relevant results before responding.'''

# Define Pydantic model for request body
class QuestionRequest(BaseModel):
    question: str
    user_id: int
    thread_id: int

@app.post("/generate")
async def stream_graph_updates(request: QuestionRequest, system_message: str = system_message):          
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": request.question}]        
    config={"configurable": {"user_id": request.user_id, "thread_id": request.thread_id}}    
    try:
        outputs = []
        for output in graph.stream({"messages": messages}, config):
            print(output)
            for key, value in output.items():
                outputs.append({key: value})
        # print('outputs', outputs)
        return {"result": outputs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)