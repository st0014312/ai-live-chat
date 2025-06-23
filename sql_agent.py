"""
SQL Agent Module

This module provides an agent specialized in querying SQL databases and returning
results based on natural language questions. It uses LangChain's SQLDatabaseToolkit
for intelligent database interaction.
"""

import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase

from utils import initialize_llm, initialize_sql_database, pretty_print_messages

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_K = 5


class SQLAgentError(Exception):
    """Exception raised when SQL agent operations fail."""

    pass


def create_sql_agent(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    database_url: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
) -> Any:
    """
    Create a SQL agent for database querying.

    Args:
        model_name: Name of the model to use
        temperature: Temperature for response generation
        database_url: Custom database URL (if None, uses default Chinook database)
        top_k: Maximum number of results to return

    Returns:
        Configured SQL agent

    Raises:
        SQLAgentError: If agent creation fails
    """
    try:
        # Load environment variables
        load_dotenv()

        # Set defaults
        model_name = model_name or os.getenv("SQL_AGENT_MODEL_NAME", DEFAULT_MODEL_NAME)
        temperature = temperature or float(
            os.getenv("SQL_AGENT_TEMPERATURE", DEFAULT_TEMPERATURE)
        )

        logger.info(
            f"Creating SQL agent with model: {model_name}, temperature: {temperature}"
        )

        # Initialize LLM
        llm = initialize_llm(
            model_name=model_name,
            temperature=temperature,
            llm_provider="openrouter",  # SQL agent typically uses openrouter for cost efficiency
        )

        # Initialize database
        engine = initialize_sql_database(database_url=database_url)
        db = SQLDatabase(engine)

        # Create SQL toolkit
        sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)

        # Create system prompt
        system_prompt = _create_sql_system_prompt(db.dialect, top_k)

        # Create agent
        agent = create_react_agent(
            model=llm,
            tools=sql_toolkit.get_tools(),
            prompt=system_prompt,
            name="sql_agent",
        )

        logger.info("SQL agent created successfully")
        return agent

    except Exception as e:
        logger.error(f"Failed to create SQL agent: {str(e)}")
        raise SQLAgentError(f"SQL agent creation failed: {str(e)}") from e


def _create_sql_system_prompt(dialect: str, top_k: int) -> str:
    """
    Create the system prompt for the SQL agent.

    Args:
        dialect: SQL dialect (e.g., 'sqlite', 'postgresql')
        top_k: Maximum number of results to return

    Returns:
        Formatted system prompt
    """
    return f"""You are an expert SQL agent designed to interact with a {dialect} database.

**Your Responsibilities:**
- Convert natural language questions into syntactically correct {dialect} queries
- Execute queries and return meaningful results
- Always limit results to at most {top_k} rows unless specifically requested otherwise
- Order results by relevant columns to show the most interesting examples
- Only query relevant columns based on the question
- Double-check queries before execution and retry if errors occur

**Database Guidelines:**
- NEVER execute DML statements (INSERT, UPDATE, DELETE, DROP, etc.)
- ALWAYS start by exploring available tables and their schemas
- Query the schema of relevant tables before writing complex queries
- Use appropriate JOINs when querying related data
- Apply proper filtering and ordering for meaningful results

**Response Format:**
- Provide clear, concise answers based on query results
- Include relevant context and explanations
- Format results in a readable manner
- If no results are found, explain why and suggest alternatives

**Error Handling:**
- If a query fails, analyze the error and rewrite the query
- Provide helpful error messages to users
- Suggest alternative approaches when appropriate

**Example Queries You Can Handle:**
- "How many customers do we have?"
- "What are the top 3 best selling artists?"
- "Show me the most expensive tracks"
- "Which country has the most customers?"
- "What is the total revenue from all sales?"

Remember: Always prioritize accuracy, security, and user-friendly responses."""
