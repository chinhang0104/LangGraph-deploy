import os
import uuid
from dotenv import load_dotenv
import argparse
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

def main() -> List[dict]:
    parser = argparse.ArgumentParser(description='I am a Chatbot.')
    parser.add_argument('question', type=str, help='User question')
    parser.add_argument('user_id', type=str, help='User id number')
    parser.add_argument('thread_id', type=str, help='Thread id of the conversation')
    args = parser.parse_args()

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

    # Invoke the graph
    question = args.question #"Which genre on average has the longest tracks in the database?"
    steps = []
    #Use stream_mode "value" for real application. Use stream_mode "debug" for debug. 
    stream_mode="debug"
    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config = {"configurable": {"user_id": args.user_id, "thread_id": args.thread_id}}, 
        stream_mode=stream_mode, 
    ):
        if stream_mode == "debug":
            print(step) 
        steps.append(step)

    # Set up chat history store
    # if stream_mode == "value":
    #     DATABASE_URL = os.getenv('DATABASE_URL')
    #     session_id = str(uuid.uuid4())
    #     chat_history = init_chat_history_manager(DATABASE_URL, session_id)
    #     # Add messages to database
    #     chat_history.add_messages(steps[-1]['messages'])

    return steps

if __name__ == "__main__":
    main()