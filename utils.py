import os
from langchain_openai import ChatOpenAI
import requests
import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from langchain_core.messages import convert_to_messages


# this function is used to initialize the llm
def initialize_llm(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.7,
    llm_provider: str = "openrouter",
):
    if llm_provider == "openrouter":
        return ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
        )
    elif llm_provider == "openai":
        return ChatOpenAI(model_name=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")


def initialize_sql_database():
    # Download and populate Chinook database
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text

    # Create in-memory SQLite database
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)

    # Create SQLAlchemy engine
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")