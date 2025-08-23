import os
from typing import List, Tuple
import requests
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.tools.base import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel

def get_sql_db_tool(llm: BaseChatModel) -> Tuple[List[BaseTool], SQLDatabase]:
    # Download db if not exist
    if not os.path.exists("Chinook.db"):
        url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"

        response = requests.get(url)

        if response.status_code == 200:
            # Open a local file in binary write mode
            with open("Chinook.db", "wb") as file:
                # Write the content of the response (the file) to the local file
                file.write(response.content)
            print("File downloaded and saved as Chinook.db")
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")   
    # Load up db         
    db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    return tools, db