
from typing import Annotated, List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from langchain_core.tools.base import BaseTool
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import Command, Send

def create_task_description_handoff_tool(
    *, agent_name: str, description: str | None = None
):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        # this is populated by the supervisor LLM
        task_description: Annotated[
            str,
            "Description of what the next agent should do, including all of the relevant context.",
        ],
        # these parameters are ignored by the LLM
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        task_description_message = {"role": "user", "content": task_description}
        agent_input = {**state, "messages": [task_description_message]}
        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )
    return handoff_tool

def create_supervisor_agent_with_description(llm: BaseChatModel, tools: List[BaseTool]) -> CompiledStateGraph:
    # Supervisor node
    supervisor_agent_with_description = create_react_agent(
        model=llm,
        tools=tools,
        prompt=(
            "You are a chatbot master supervisoring agent:\n"
            "- a research agent. The research agent search latest information on internet. Assign research-related tasks to this assistant.\n"
            "Use transfer_to_research_agent if neccessary. Do Not use other agents.\n"
            "Assign work to one agent at a time, do not call agents in parallel.\n"
            "The user should only see the final response. When no further tool use is needed, finalize your answer to the user, including relevant references such as web links, document IDs, or database table names."
        ),
        name="supervisor",
    )
    return supervisor_agent_with_description
