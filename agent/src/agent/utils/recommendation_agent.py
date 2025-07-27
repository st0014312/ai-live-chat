"""
Recommendation Agent Module

This module provides an agent specialized in product recommendations based on:
- Chat history and user preferences
- Product data from the database
- User behavior patterns
- Similar product analysis
"""

import os
from typing_extensions import TypedDict
import logging
from typing import List, Dict, Optional, Literal, Annotated, Any, Union
from dotenv import load_dotenv

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from langgraph_supervisor import create_supervisor

from src.agent.utils.utils import initialize_llm

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_RECOMMENDATIONS = 5


class AgentState(TypedDict):
    """State definition for the agent workflow."""

    messages: Annotated[List[AnyMessage], add_messages]


def create_recommendation_agent(
    sql_agent: Any,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_recommendations: int = DEFAULT_MAX_RECOMMENDATIONS,
) -> Any:
    """
    Create a recommendation agent for product recommendations.

    Args:
        sql_agent: SQL agent instance to use for database queries
        model_name: Name of the model to use
        temperature: Temperature for response generation
        max_recommendations: Maximum number of recommendations to provide

    Returns:
        Configured recommendation agent

    Raises:
        Exception: If agent creation fails
    """
    try:
        # Load environment variables
        load_dotenv()

        # Set defaults
        model_name = model_name or os.getenv(
            "RECOMMENDATION_AGENT_MODEL_NAME", DEFAULT_MODEL_NAME
        )
        temperature = temperature or float(
            os.getenv("RECOMMENDATION_AGENT_TEMPERATURE", DEFAULT_TEMPERATURE)
        )

        logger.info(
            f"Creating recommendation agent with model: {model_name}, temperature: {temperature}"
        )

        # Initialize LLM
        llm = initialize_llm(
            model_name=model_name,
            temperature=temperature,
            llm_provider="openrouter",  # Recommendation agent typically uses openrouter
        )

        # Create system prompt
        system_prompt = _create_recommendation_system_prompt(max_recommendations)

        # Create agent using the same pattern as SQL agent
        agent = create_supervisor(
            model=llm,
            prompt=system_prompt,
            agents=[sql_agent],
            add_handoff_back_messages=False,
            output_mode="last_message",
        ).compile(name="recommendation_agent")

        logger.info("Recommendation agent created successfully")
        return agent

    except Exception as e:
        logger.error(f"Failed to create recommendation agent: {str(e)}")
        raise Exception(f"Recommendation agent creation failed: {str(e)}") from e


def _create_recommendation_system_prompt(max_recommendations: int) -> str:
    """
    Create the system prompt for the recommendation agent.

    Args:
        max_recommendations: Maximum number of recommendations to provide

    Returns:
        Formatted system prompt
    """
    return f"""You are a recommendation supervisor agent specializing in providing personalized product recommendations. You coordinate with the SQL agent to gather product data and create meaningful recommendations.

**Your Responsibilities:**
- Analyze user queries to understand what they're looking for
- Use conversation history to understand user preferences and interests
- Delegate database queries to the SQL agent to find relevant products
- Synthesize the SQL agent's results into personalized recommendations
- Provide clear explanations for why each product is recommended
- Consider user behavior patterns and preferences from chat history

**Coordination with SQL Agent:**
- When you need product data, delegate the query to the SQL agent
- Ask the SQL agent to search for products by category, price range, or features
- Request information about top-selling products, product details, and availability
- Use the SQL agent's results to build your recommendations

**Recommendation Process:**
1. **Analyze User Query**: Understand what the user is asking for
2. **Review Conversation History**: Extract user preferences and interests from previous messages
3. **Delegate to SQL Agent**: Ask the SQL agent to find relevant products based on criteria
4. **Synthesize Results**: Combine SQL agent data with user preferences
5. **Generate Recommendations**: Create personalized recommendations with explanations
6. **Provide Context**: Explain why each recommendation is relevant

**Recommendation Guidelines:**
- Provide {max_recommendations} recommendations maximum
- Include product names, prices, and brief descriptions
- Explain why each product is recommended based on user preferences
- Consider user's budget constraints mentioned in the conversation
- Suggest alternatives if exact matches aren't available
- Include complementary products when relevant
- Reference conversation history when explaining recommendations

**Response Format:**
- Start with a brief understanding of what the user is looking for
- List recommendations with clear explanations
- Include pricing information when available
- Reference previous conversation context when relevant
- End with a summary or next steps

**Example Queries You Can Handle:**
- "Can you recommend some rock music albums?"
- "I'm looking for jazz music under $20"
- "What are the best-selling products in the classical genre?"
- "Show me products similar to what I bought before"
- "Recommend something based on my preferences"
- "I need some country music recommendations"
- "What's popular in the pop music category?"

**SQL Agent Delegation Examples:**
- "Find rock music albums under $15" → delegate to SQL agent
- "Show me top-selling jazz albums" → delegate to SQL agent
- "What are the most expensive classical albums?" → delegate to SQL agent
- "Find albums by popular artists" → delegate to SQL agent

**Important Notes:**
- Always delegate database queries to the SQL agent
- Use conversation history to personalize recommendations
- Explain your reasoning for each recommendation
- Consider user's previous interactions and preferences
- Provide helpful, personalized recommendations with clear explanations

Remember: You are a recommendation specialist who coordinates with the SQL agent to provide the best possible product recommendations based on user preferences and available data.
"""
