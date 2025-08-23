# from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
# from langchain_postgres import PostgresChatMessageHistory
# import psycopg

# def init_chat_history_manager(conn_info: str, session_id: str) -> PostgresChatMessageHistory:
#     # Replace username, password, port and table with your setting
#     #conn_info=r"postgresql://username:password@localhost:port/table"

#     # Establish a synchronous connection to the database
#     # (or use psycopg.AsyncConnection for async)
#     sync_connection = psycopg.connect(conn_info)

#     # Create the table schema (only needs to be done once)
#     table_name = "chat_history"
#     PostgresChatMessageHistory.create_tables(sync_connection, table_name)

#     # Initialize the chat history manager
#     chat_history = PostgresChatMessageHistory(
#         table_name,
#         session_id,
#         sync_connection=sync_connection
#     )
#     return 