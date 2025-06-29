from typing_extensions import TypedDict
from typing import List, Annotated, Optional
from langgraph.graph.message import AnyMessage, add_messages
from dataclasses import dataclass


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


class AgentState(TypedDict):
    """State definition for the agent workflow."""

    messages: Annotated[List[AnyMessage], add_messages]
    remaining_steps: any
