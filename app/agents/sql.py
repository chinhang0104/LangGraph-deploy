from typing import Literal, List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools.base import BaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState

# Global avr
llm = None
tools = None
db = None

# Get Set Global vars
def get_llm() -> BaseChatModel:
    return llm

def set_llm(value: BaseChatModel):
    global llm
    llm = value

def get_tools() -> List[BaseTool]:
    return tools

def set_tools(value: List[BaseTool]):
    global tools
    tools = value

def get_db() -> SQLDatabase:
    return db

def set_db(value: SQLDatabase):
    global db
    db = value

# Get table schema
# Use set_tools to set tool before running this function
def get_get_schema_node() -> ToolNode:
    get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
    return ToolNode([get_schema_tool], name="get_schema")

# Run SQL query
# Use set_tools to set tool before running this function
def get_run_query_node() -> ToolNode:
    run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
    return ToolNode([run_query_tool], name="run_query")

# Entry point of SQL
def sql_agent(state: MessagesState):
    return state

def list_tables(state: MessagesState):
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "abc123",
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])

    list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
    tool_message = list_tables_tool.invoke(tool_call)
    response = AIMessage(f"Available tables: {tool_message.content}")

    return {"messages": [tool_call_message, tool_message, response]}


# Example: force a model to create a tool call
def call_get_schema(state: MessagesState):
    # Note that LangChain enforces that all models accept `tool_choice="any"`
    # as well as `tool_choice=<string name of tool>`.
    get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
    llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}


def generate_query(state: MessagesState):
    generate_query_system_prompt = """
    You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {dialect} query to run,
    then look at the results of the query and return the answer. Unless the user
    specifies a specific number of examples they wish to obtain, always limit your
    query to at most {top_k} results.

    You can order the results by a relevant column to return the most interesting
    examples in the database. Never query for all the columns from a specific table,
    only ask for the relevant columns given the question.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
    """.format(
        dialect=db.dialect,
        top_k=5,
    )
    
    system_message = {
        "role": "system",
        "content": generate_query_system_prompt,
    }
    # We do not force a tool call here, to allow the model to
    # respond naturally when it obtains the solution.
    run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
    llm_with_tools = llm.bind_tools([run_query_tool])
    response = llm_with_tools.invoke([system_message] + state["messages"])

    return {"messages": [response]}


def check_query(state: MessagesState):
    check_query_system_prompt = """
    You are a SQL expert with a strong attention to detail.
    Double check the {dialect} query for common mistakes, including:
    - Using NOT IN with NULL values
    - Using UNION when UNION ALL should have been used
    - Using BETWEEN for exclusive ranges
    - Data type mismatch in predicates
    - Properly quoting identifiers
    - Using the correct number of arguments for functions
    - Casting to the correct data type
    - Using the proper columns for joins

    If there are any of the above mistakes, rewrite the query. If there are no mistakes,
    just reproduce the original query.

    You will call the appropriate tool to execute the query after running this check.
    """.format(dialect=db.dialect)

    system_message = {
            "role": "system",
            "content": check_query_system_prompt,
        }

    # Generate an artificial user message to check
    tool_call = state["messages"][-1].tool_calls[0]
    user_message = {"role": "user", "content": tool_call["args"]["query"]}
    run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
    llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    response.id = state["messages"][-1].id

    return {"messages": [response]}

def should_continue(state: MessagesState) -> Literal["supervisor", "check_query"]:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "supervisor"
    else:
        return "check_query"