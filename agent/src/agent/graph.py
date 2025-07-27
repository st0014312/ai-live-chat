"""
AI Live Chat Agent

This module provides a comprehensive AI-powered customer service agent that can
handle live chat for online stores. The agent integrates knowledge base retrieval,
SQL database querying, and intelligent routing through a supervisor pattern.
"""

import logging
import os
from dotenv import load_dotenv


from agent.utils.node import create_general_node
from src.agent.utils.utils import (
    initialize_llm,
    initialize_knowledge_base,
)
from langgraph_supervisor import create_supervisor
from src.agent.utils.sql_agent import create_sql_agent
from src.agent.utils.recommendation_agent import create_recommendation_agent
from src.agent.utils.state import AgentConfig, AgentState

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
"""

DEFAULT_TEMPERATURE = 0.7

load_dotenv()
llm = initialize_llm(model_name=os.getenv("OPENROUTER_MODEL_NAME"))
knowledge_base = initialize_knowledge_base()

general_agent = create_general_node(
    model=llm,
    vector_store=knowledge_base,
)
sql_agent = create_sql_agent()
recommendation_agent = create_recommendation_agent(
    sql_agent=sql_agent,
    model_name="gpt-4o-mini",
)

graph = create_supervisor(
    model=llm,
    agents=[general_agent, recommendation_agent],
    prompt=SYSTEM_PROMPT,
    tools=[],
    add_handoff_back_messages=True,
    output_mode="full_history",
    config_schema=AgentConfig,
    state_schema=AgentState,
).compile(name="ai-chat-agent")
