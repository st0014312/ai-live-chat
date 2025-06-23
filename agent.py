import os
from typing import List, Dict, Optional, Literal, TypedDict, Annotated, Any
from dotenv import load_dotenv

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import HumanMessage

from utils import initialize_llm, pretty_print_messages
from langgraph_supervisor import create_supervisor
from sql_agent import create_sql_agent
from knowledge_base_agent import create_knowledge_base_agent

SYSTEM_PROMPT = """ you are a supervisor agent that is responsible for the following agents:
        - sql_agent: a agent that is responsible for querying the database and returning the results
        - knowledge_base_agent: a agent that is responsible for answering questions about the knowledge base except the questions about products
        Assign work to one agent at a time, do not call agents in parallel
        Do not do anything by yourself, only assign work to the agents
        if the knowledge base agent is not able to answer the question, then assign the question to the next agent"""


class AgentState(TypedDict):
    """State for the agent."""

    messages: Annotated[list[AnyMessage], add_messages]


class Agent:
    def __init__(
        self,
        llm_provider: str = "openrouter",
        embedding_provider: str = "huggingface",
        model_name: Optional[str] = "gpt-4o-mini",
        temperature: Optional[float] = 0.7,
        knowledge_base_dir: Optional[str] = "./knowledge_base",
    ):
        """Initialize the agent.

        Args:
            llm_provider: LLM provider to use
            embedding_provider: Embedding provider to use
            model_name: Model name to use
            temperature: Temperature for LLM
            knowledge_base_dir: Directory containing knowledge base files
        """

        # Load environment variables
        load_dotenv()

        self.model_name = os.getenv(
            f"{llm_provider.upper()}_MODEL_NAME",
            "gpt-4o-mini" if llm_provider == "openai" else "openai/gpt-4o-mini",
        )

        self.embedding_provider = os.getenv(
            f"{embedding_provider.upper()}_EMBEDDING_PROVIDER",
            "huggingface" if embedding_provider == "huggingface" else "openai",
        )

        self.sql_agent = create_sql_agent()
        self.knowledge_base_agent = create_knowledge_base_agent(
            embeddings_provider=self.embedding_provider,
        )
        self.llm = initialize_llm(model_name=self.model_name, temperature=temperature)
        self.supervisor = create_supervisor(
            model=self.llm,
            agents=[self.sql_agent, self.knowledge_base_agent],
            prompt=SYSTEM_PROMPT,
            add_handoff_back_messages=True,
            output_mode="full_history",
        ).compile(checkpointer=MemorySaver())

    def chat(self, user_input: str, thread_id: Optional[str]) -> Dict:
        """Process user input and return a response."""
        try:
            # Create initial state
            state = {
                "messages": [HumanMessage(content=user_input)],
            }

            # Configure the run with thread ID for memory
            config = {"configurable": {"thread_id": thread_id}}

            for chunk in self.supervisor.stream(state, config=config):
                pretty_print_messages(chunk, last_message=True)

            final_message_history = chunk["supervisor"]["messages"]
            return final_message_history
        except Exception as e:
            return {"response": f"Error: {str(e)}"}
