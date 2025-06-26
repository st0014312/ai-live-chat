"""
AI Live Chat Agent

This module provides a comprehensive AI-powered customer service agent that can
handle live chat for online stores. The agent integrates knowledge base retrieval,
SQL database querying, and intelligent routing through a supervisor pattern.
"""

import os
import logging
from typing_extensions import TypedDict
from typing import List, Dict, Optional, Literal, Annotated, Any, Union
from dataclasses import dataclass
from dotenv import load_dotenv

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import HumanMessage, AIMessage

from utils import initialize_llm, pretty_print_messages
from langgraph_supervisor import create_supervisor
from sql_agent import create_sql_agent
from knowledge_base_agent import create_knowledge_base_agent
from recommendation_agent import create_recommendation_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# System prompt for the supervisor agent
SYSTEM_PROMPT = """You are a supervisor agent responsible for coordinating the following specialized agents:

1. **sql_agent**: Handles database queries and returns results for customer data, sales analytics, 
   inventory information, and business intelligence questions.

2. **knowledge_base_agent**: Answers questions about store policies, shipping information, 
   return policies, FAQs from the knowledge base.

3. **recommendation_agent**: Provides personalized product recommendations based on user preferences, 
   chat history, and available product data.

**Routing Guidelines:**
- Route database-related queries (customer data, sales, analytics, inventory, product information) to sql_agent
- Route policy and general information queries to knowledge_base_agent
- Route product recommendation requests to recommendation_agent
- For questions about conversation history, context, or personal information (like "who am i", "what did we talk about", etc.), answer directly using the conversation context
- For follow-up questions that reference previous conversation, answer directly using the conversation context
- Assign work to one agent at a time (no parallel processing)
- If the knowledge_base_agent cannot answer, route to sql_agent as fallback
- Only delegate to specialized agents for their specific domains

**Examples:**
- "How many customers do we have?" → sql_agent
- "What's your return policy?" → knowledge_base_agent
- "Show me top selling products" → sql_agent
- "How long does shipping take?" → knowledge_base_agent
- "Can you recommend some rock music albums?" → recommendation_agent
- "I'm looking for jazz music under $20" → recommendation_agent
- "What are the best-selling products in the classical genre?" → recommendation_agent
- "Show me products similar to what I bought before" → recommendation_agent
- "Recommend something based on my preferences" → recommendation_agent
- "Who am I?" → Answer directly using conversation context
- "What did we talk about?" → Answer directly using conversation context
- "Can you remember our previous conversation?" → Answer directly using conversation context

**Important:** Answer conversation history and context questions directly. Only route to specialized agents for their specific domains (database queries, store policies, product recommendations, etc.).
"""


class AgentState(TypedDict):
    """State definition for the agent workflow."""

    messages: Annotated[List[AnyMessage], add_messages]


@dataclass
class AgentConfig:
    """Configuration for the AI agent."""

    llm_provider: str = "openrouter"
    embedding_provider: str = "huggingface"
    model_name: Optional[str] = None
    temperature: float = 0.7
    knowledge_base_dir: str = "./knowledge_base"
    verbose: bool = False
    max_retries: int = 3


class AgentError(Exception):
    """Custom exception for agent-related errors."""

    pass


class Agent:
    """
    AI Live Chat Agent for customer service.

    This agent provides a comprehensive customer service solution that can:
    - Answer questions from a knowledge base
    - Query SQL databases for customer data and analytics
    - Route queries intelligently between specialized agents
    - Maintain conversation context across sessions
    """

    def __init__(
        self,
        llm_provider: str = "openrouter",
        embedding_provider: str = "huggingface",
        model_name: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        knowledge_base_dir: Optional[str] = "./knowledge_base",
        verbose: bool = False,
        max_retries: int = 3,
    ):
        """
        Initialize the AI agent.

        Args:
            llm_provider: LLM provider to use ("openai" or "openrouter")
            embedding_provider: Embedding provider to use ("openai" or "huggingface")
            model_name: Specific model name to use (overrides environment variable)
            temperature: Temperature for LLM responses (0.0 to 1.0)
            knowledge_base_dir: Directory containing knowledge base files
            verbose: Enable verbose logging
            max_retries: Maximum number of retries for failed operations

        Raises:
            AgentError: If initialization fails
        """
        try:
            # Load environment variables
            load_dotenv()

            # Configure logging
            if verbose:
                logging.getLogger().setLevel(logging.DEBUG)

            # Store configuration
            self.config = AgentConfig(
                llm_provider=llm_provider,
                embedding_provider=embedding_provider,
                model_name=model_name,
                temperature=temperature or 0.7,
                knowledge_base_dir=knowledge_base_dir or "./knowledge_base",
                verbose=verbose,
                max_retries=max_retries,
            )

            # Initialize components
            self._initialize_llm()
            self._initialize_agents()
            self._initialize_supervisor()

            logger.info(
                f"Agent initialized successfully with {llm_provider} LLM and {embedding_provider} embeddings"
            )

        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise AgentError(f"Agent initialization failed: {str(e)}") from e

    def _initialize_llm(self) -> None:
        """Initialize the language model."""
        try:
            # Determine model name
            if self.config.model_name:
                model_name = self.config.model_name
            else:
                env_key = f"{self.config.llm_provider.upper()}_MODEL_NAME"
                model_name = os.getenv(env_key)

                if not model_name:
                    # Default models
                    if self.config.llm_provider == "openai":
                        model_name = "gpt-4o-mini"
                    else:  # openrouter
                        model_name = "openai/gpt-4o-mini"

            self.llm = initialize_llm(
                model_name=model_name,
                temperature=self.config.temperature,
                llm_provider=self.config.llm_provider,
            )

            logger.debug(
                f"LLM initialized: {model_name} (provider: {self.config.llm_provider})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise AgentError(f"LLM initialization failed: {str(e)}") from e

    def _initialize_agents(self) -> None:
        """Initialize specialized agents."""
        try:
            # Initialize SQL agent first (needed by recommendation agent)
            self.sql_agent = create_sql_agent()
            logger.debug("SQL agent initialized")

            # Initialize knowledge base agent
            self.knowledge_base_agent = create_knowledge_base_agent(
                embeddings_provider=self.config.embedding_provider,
                knowledge_base_dir=self.config.knowledge_base_dir,
            )
            logger.debug("Knowledge base agent initialized")

            # Initialize recommendation agent (depends on SQL agent)
            self.recommendation_agent = create_recommendation_agent(
                sql_agent=self.sql_agent
            )
            logger.debug("Recommendation agent initialized")

        except Exception as e:
            logger.error(f"Failed to initialize agents: {str(e)}")
            raise AgentError(f"Agent initialization failed: {str(e)}") from e

    def _initialize_supervisor(self) -> None:
        """Initialize the supervisor workflow."""
        try:
            self.supervisor = create_supervisor(
                model=self.llm,
                agents=[
                    self.sql_agent,
                    self.knowledge_base_agent,
                    self.recommendation_agent,
                ],
                prompt=SYSTEM_PROMPT,
                add_handoff_back_messages=True,
                output_mode="full_history",
            ).compile(checkpointer=MemorySaver())

            logger.debug("Supervisor workflow initialized")

        except Exception as e:
            logger.error(f"Failed to initialize supervisor: {str(e)}")
            raise AgentError(f"Supervisor initialization failed: {str(e)}") from e

    def chat(
        self,
        user_input: str,
        thread_id: Optional[str] = None,
        verbose: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Process user input and return a response.

        Args:
            user_input: The user's message
            thread_id: Optional thread ID for conversation continuity
            verbose: Override verbose setting for this call

        Returns:
            Dictionary containing the response and metadata

        Raises:
            AgentError: If processing fails
        """
        if not user_input or not user_input.strip():
            return {
                "response": "I didn't receive any input. Please provide a question or message.",
                "error": "Empty input",
                "thread_id": thread_id,
            }

        # Use instance verbose setting if not overridden
        if verbose is None:
            verbose = self.config.verbose

        try:
            logger.debug(f"Processing user input: {user_input[:100]}...")

            # Create initial state
            state = {
                "messages": [HumanMessage(content=user_input.strip())],
            }

            # Configure the run with thread ID for memory
            config = (
                {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}
                if thread_id
                else {}
            )

            # Process through supervisor
            final_chunk = None
            for chunk in self.supervisor.stream(state, config=config):
                # if verbose:
                # pretty_print_messages(chunk, last_message=True)
                pretty_print_messages(chunk, last_message=True)

                final_chunk = chunk

            if not final_chunk:
                raise AgentError("No response generated from supervisor")

            # Extract response
            if "supervisor" in final_chunk and "messages" in final_chunk["supervisor"]:
                final_message_history = final_chunk["supervisor"]["messages"]
            else:
                # Fallback: look for any agent response
                for agent_name, agent_data in final_chunk.items():
                    if "messages" in agent_data:
                        final_message_history = agent_data["messages"]
                        break
                else:
                    raise AgentError("No valid response found in final chunk")

            # Format response
            response = self._format_response(final_message_history)

            logger.debug(f"Response generated successfully for thread: {thread_id}")

            return {
                "response": response,
                "thread_id": thread_id,
                "agent_used": self._determine_agent_used(final_chunk),
                "timestamp": self._get_timestamp(),
            }

        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}")
            return {
                "response": f"I apologize, but I encountered an error while processing your request: {str(e)}",
                "error": str(e),
                "thread_id": thread_id,
            }

    def _format_response(self, message_history: List[AnyMessage]) -> str:
        """Format the message history into a readable response."""
        if not message_history:
            return "I apologize, but I couldn't generate a response."

        # Get the last AI message
        for message in reversed(message_history):
            if isinstance(message, AIMessage):
                return message.content

        # Fallback: return the last message content
        last_message = message_history[-1]
        if hasattr(last_message, "content"):
            return last_message.content

        return "I apologize, but I couldn't format the response properly."

    def _determine_agent_used(self, final_chunk: Dict[str, Any]) -> str:
        """Determine which agent was used based on the final chunk."""
        # Look for the agent that has the most recent activity
        for agent_name in ["sql_agent", "knowledge_base_agent", "recommendation_agent"]:
            if agent_name in final_chunk:
                return agent_name
        return "supervisor"

    def _get_timestamp(self) -> str:
        """Get current timestamp for response metadata."""
        from datetime import datetime

        return datetime.now().isoformat()

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent."""
        return {
            "status": "ready",
            "llm_provider": self.config.llm_provider,
            "embedding_provider": self.config.embedding_provider,
            "model_name": self.config.model_name,
            "temperature": self.config.temperature,
            "verbose": self.config.verbose,
            "max_retries": self.config.max_retries,
        }

    def reset_conversation(self, thread_id: str) -> bool:
        """
        Reset conversation context for a specific thread.

        Args:
            thread_id: The thread ID to reset

        Returns:
            True if reset was successful
        """
        try:
            # This would need to be implemented based on the specific checkpoint system
            # For now, we'll just log the request
            logger.info(f"Conversation reset requested for thread: {thread_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to reset conversation: {str(e)}")
            return False
