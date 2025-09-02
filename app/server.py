import os
import uuid
import asyncio
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, WebSocket
from fastapi.responses import RedirectResponse, StreamingResponse
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.checkpoint.memory import MemorySaver
from agents.research import create_research_agent
from agents.retriever import get_retriever_tool
from agents.sql import *
from agents.supervisor import create_task_description_handoff_tool, create_supervisor_agent_with_description
from tools.chinook_db import get_sql_db_tool
from tools.lilianweng_vectorstore import get_vectorstore
#from tools.postgres_chat_message_history import init_chat_history_manager

app = FastAPI()

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# Load Groq environment variables from .env file
load_dotenv()

# Init llm model
llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")

# RAG
vectorstore = get_vectorstore()
retriever_tool = get_retriever_tool(vectorstore, "retrieve_blog_posts", "Search and return information about Lilian Weng blog posts.")

# Web search
research_agent = create_research_agent(llm)

# SQL
db_tools, db = get_sql_db_tool(llm)
set_llm(llm)
set_tools(db_tools)
set_db(db)
get_schema_node = get_get_schema_node()
run_query_node = get_run_query_node()

# Handoffs tools
assign_to_research_agent_with_description = create_task_description_handoff_tool(
    agent_name="research_agent",
    description="Assign task to a researcher agent.",
)

assign_to_retriever_agent_with_description = create_task_description_handoff_tool(
    agent_name="retriever",
    description="Assign task to a rag agent.",
)

assign_to_sql_agent_with_description = create_task_description_handoff_tool(
    agent_name="sql_agent",
    description="Assign task to a rag agent.",
)
supervisor_handoffs_tools = [assign_to_research_agent_with_description,
                             assign_to_retriever_agent_with_description,
                             assign_to_sql_agent_with_description]

# Supervisor agent
supervisor_agent_with_description = create_supervisor_agent_with_description(llm, supervisor_handoffs_tools) # Change tools here

# Define the graph
builder = StateGraph(MessagesState)
builder.add_node(
    supervisor_agent_with_description, destinations=("research_agent", "retrieve", "sql_agent", END)
)
# Add nodes otherthan SQL
builder.add_node(research_agent)
builder.add_node("retrieve", ToolNode([retriever_tool]))

# Add edges otherthan SQL
builder.add_edge(START, "supervisor")
builder.add_edge("research_agent", "supervisor")
builder.add_edge("retrieve", "supervisor")

# SQL agent
builder.add_node(sql_agent)
builder.add_node(list_tables)
builder.add_node(call_get_schema)
builder.add_node(get_schema_node, "get_schema")
builder.add_node(generate_query)
builder.add_node(check_query)
builder.add_node(run_query_node, "run_query")

builder.add_edge("sql_agent", "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_conditional_edges(
    "generate_query",
    should_continue,
)
builder.add_edge("check_query", "run_query")
builder.add_edge("run_query", "generate_query")

# MemorySaver helps remember chat history
# Use SqliteSaver or PostgresSaver and connect a database for persistent store. 
memory = MemorySaver() # In-Memory Saver, for demo only

agent = builder.compile(checkpointer=memory)

# Define Pydantic model for request body
class QuestionRequest(BaseModel):
    question: str
    user_id: str
    thread_id: str

# Invoke the graph, return last answer from chatbot
@app.post("/generate")
async def stream_graph_updates(request: QuestionRequest):
    try:
        steps = []
        for step in agent.stream(
            {"messages": [{"role": "user", "content": request.question}]}, 
            config= {"configurable": {"user_id": request.user_id, "thread_id": request.thread_id}},        
            stream_mode="values",  #Use stream_mode "values" for real application. Use stream_mode "debug" for debug. 
        ):
            # if stream_mode == "debug":
            print(step) 
            steps.append(step)
        # if stream_mode == "debug":
        #     return steps
        # Sometimes supervisor will silence if toolnodes answer the user question.
        # In this case, we return the message from toolnodes
        if steps[-1]['messages'][-1].content:
            return {"result": steps[-1]['messages'][-1].content} #supervisor message
        else:
            return {"result": steps[-1]['messages'][-2].content} #toolnodes message
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Invoke the graph, return every message from chatbot
@app.websocket("/ws/generate")
async def websocket_generator(ws: WebSocket):
    await ws.accept()
    data = await ws.receive_json()
    req = QuestionRequest(**data)

    for step in agent.stream(
        {"messages":[{"role":"user","content":req.question}]},
        config={"configurable":{"user_id":req.user_id,"thread_id":req.thread_id}},
        stream_mode="values",
    ):        
        msg = step["messages"][-1].content
        print("message:" + msg)
        await ws.send_json({"message": msg})

    await ws.close()

# # Set up chat history store
# if stream_mode == "values":
#     DATABASE_URL = os.getenv('DATABASE_URL')
#     session_id = str(uuid.uuid4())
#     chat_history = init_chat_history_manager(DATABASE_URL, session_id)
#     # Add messages to database
#     chat_history.add_messages(steps[-1]['messages'])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
