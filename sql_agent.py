# this agent is used to query the database and return the results according to the question passed from the supervisor agent
import os
from typing import List, Dict, Optional, Literal, TypedDict, Annotated, Any
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from utils import initialize_llm, initialize_sql_database, pretty_print_messages


def create_sql_agent():
    llm = initialize_llm(model_name="gpt-4o-mini")
    engine = initialize_sql_database()
    db = SQLDatabase(engine)
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return create_react_agent(
        model=llm,
        tools=sql_toolkit.get_tools(),
        prompt="""
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
            dialect=db.dialect, top_k=5
        ),
        name="sql_agent",
    )
