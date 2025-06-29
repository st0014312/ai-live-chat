"""
Utility functions for the AI Live Chat Agent.

This module provides utility functions for LLM initialization, database setup,
and message formatting.
"""

import os
import logging
import requests
import sqlite3
from typing import Optional, Union, Dict, Any
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import SQLAlchemyError
from langchain_core.messages import convert_to_messages, BaseMessage

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CHINOOK_URL = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TEMPERATURE = 0.7


class LLMInitializationError(Exception):
    """Exception raised when LLM initialization fails."""
    pass


class DatabaseInitializationError(Exception):
    """Exception raised when database initialization fails."""
    pass


def initialize_llm(
    model_name: str = "gpt-4o-mini",
    temperature: float = DEFAULT_TEMPERATURE,
    llm_provider: str = "openrouter",
    max_tokens: Optional[int] = None,
    request_timeout: Optional[int] = None,
) -> ChatOpenAI:
    """
    Initialize a language model with the specified configuration.
    
    Args:
        model_name: Name of the model to use
        temperature: Temperature for response generation (0.0 to 1.0)
        llm_provider: Provider to use ("openai" or "openrouter")
        max_tokens: Maximum number of tokens in response
        request_timeout: Request timeout in seconds
        
    Returns:
        Configured ChatOpenAI instance
        
    Raises:
        LLMInitializationError: If initialization fails
        ValueError: If provider is not supported
    """
    try:
        if llm_provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            base_url = os.getenv("OPENROUTER_BASE_URL")
            
            if not api_key:
                raise LLMInitializationError("OPENROUTER_API_KEY environment variable not set")
            
            if not base_url:
                raise LLMInitializationError("OPENROUTER_BASE_URL environment variable not set")
            
            llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                openai_api_key=api_key,
                openai_api_base=base_url,
            )
            
        elif llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise LLMInitializationError("OPENAI_API_KEY environment variable not set")
            
            llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                openai_api_key=api_key,
            )
            
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        
        # Set optional parameters
        if max_tokens:
            llm.max_tokens = max_tokens
        
        if request_timeout:
            llm.request_timeout = request_timeout
        
        logger.debug(f"LLM initialized: {model_name} (provider: {llm_provider}, temperature: {temperature})")
        return llm
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        if isinstance(e, (LLMInitializationError, ValueError)):
            raise
        raise LLMInitializationError(f"LLM initialization failed: {str(e)}") from e


def initialize_sql_database(
    database_url: Optional[str] = None,
    chinook_url: str = DEFAULT_CHINOOK_URL,
    timeout: int = 30,
) -> Any:
    """
    Initialize a SQL database with sample data.
    
    Args:
        database_url: Custom database URL (if None, uses in-memory SQLite)
        chinook_url: URL to download Chinook database schema
        timeout: Request timeout in seconds
        
    Returns:
        SQLAlchemy engine instance
        
    Raises:
        DatabaseInitializationError: If database initialization fails
    """
    try:
        if database_url:
            # Use custom database URL
            logger.info(f"Connecting to custom database: {database_url}")
            return create_engine(database_url)
        
        # Download and populate Chinook database
        logger.info("Downloading Chinook database schema...")
        response = requests.get(chinook_url, timeout=timeout)
        response.raise_for_status()
        sql_script = response.text
        
        # Create in-memory SQLite database
        logger.info("Creating in-memory SQLite database...")
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        connection.executescript(sql_script)
        
        # Create SQLAlchemy engine
        engine = create_engine(
            "sqlite://",
            creator=lambda: connection,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        
        logger.info("SQL database initialized successfully")
        return engine
        
    except requests.RequestException as e:
        logger.error(f"Failed to download database schema: {str(e)}")
        raise DatabaseInitializationError(f"Failed to download database schema: {str(e)}") from e
    except SQLAlchemyError as e:
        logger.error(f"Failed to create database engine: {str(e)}")
        raise DatabaseInitializationError(f"Failed to create database engine: {str(e)}") from e
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise DatabaseInitializationError(f"Database initialization failed: {str(e)}") from e


def pretty_print_message(message: BaseMessage, indent: bool = False) -> None:
    """
    Pretty print a single message.
    
    Args:
        message: The message to print
        indent: Whether to indent the output
    """
    try:
        pretty_message = message.pretty_repr(html=True)
        if not indent:
            print(pretty_message)
            return
        
        indented = "\n".join("\t" + line for line in pretty_message.split("\n"))
        print(indented)
        
    except Exception as e:
        logger.error(f"Failed to pretty print message: {str(e)}")
        # Fallback to simple printing
        print(f"Message: {str(message)}")


def pretty_print_messages(update: Union[Dict[str, Any], tuple], last_message: bool = False) -> None:
    """
    Pretty print messages from a graph update.
    
    Args:
        update: Update from the graph execution
        last_message: Whether to show only the last message
    """
    try:
        is_subgraph = False
        
        if isinstance(update, tuple):
            ns, update = update
            # Skip parent graph updates in the printouts
            if len(ns) == 0:
                return
            
            graph_id = ns[-1].split(":")[0]
            print(f"Update from subgraph {graph_id}:")
            print()
            is_subgraph = True
        
        for node_name, node_update in update.items():
            update_label = f"Update from node {node_name}:"
            if is_subgraph:
                update_label = "\t" + update_label
            
            print(update_label)
            print()
            
            if "messages" not in node_update:
                logger.warning(f"No messages found in node update: {node_name}")
                continue
            
            messages = convert_to_messages(node_update["messages"])
            if last_message:
                messages = messages[-1:]
            
            for message in messages:
                pretty_print_message(message, indent=is_subgraph)
            print()
            
    except Exception as e:
        logger.error(f"Failed to pretty print messages: {str(e)}")
        print(f"Error printing messages: {str(e)}")


def validate_environment_variables() -> Dict[str, bool]:
    """
    Validate that required environment variables are set.
    
    Returns:
        Dictionary mapping variable names to their validation status
    """
    required_vars = {
        "OPENAI_API_KEY": False,
        "OPENROUTER_API_KEY": False,
        "OPENROUTER_BASE_URL": False,
    }
    
    for var_name in required_vars:
        if os.getenv(var_name):
            required_vars[var_name] = True
    
    return required_vars


def get_config_value(key: str, default: Any = None, required: bool = False) -> Any:
    """
    Get a configuration value from environment variables.
    
    Args:
        key: Configuration key
        default: Default value if not found
        required: Whether the value is required
        
    Returns:
        Configuration value
        
    Raises:
        ValueError: If required value is not found
    """
    value = os.getenv(key, default)
    
    if required and value is None:
        raise ValueError(f"Required configuration '{key}' not found")
    
    return value


def sanitize_input(text: str, max_length: int = 1000) -> str:
    """
    Sanitize user input text.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Remove leading/trailing whitespace
    sanitized = text.strip()
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."
        logger.warning(f"Input truncated to {max_length} characters")
    
    return sanitized


def format_error_response(error: Exception, context: str = "") -> Dict[str, str]:
    """
    Format an error response for the user.
    
    Args:
        error: The exception that occurred
        context: Additional context about the error
        
    Returns:
        Formatted error response
    """
    error_message = str(error)
    
    # Provide user-friendly error messages
    if "API key" in error_message.lower():
        return {
            "error": "Authentication error. Please check your API configuration.",
            "details": error_message
        }
    elif "timeout" in error_message.lower():
        return {
            "error": "Request timed out. Please try again.",
            "details": error_message
        }
    elif "rate limit" in error_message.lower():
        return {
            "error": "Rate limit exceeded. Please wait a moment and try again.",
            "details": error_message
        }
    else:
        return {
            "error": "An unexpected error occurred. Please try again.",
            "details": error_message
        }