import os
from dotenv import load_dotenv
from langgraph_supervisor import create_supervisor

from src.agent.utils.utils import initialize_llm
from src.agent.utils.sql_agent import create_sql_agent
from src.agent.utils.state import AgentState
from langgraph.prebuilt import create_react_agent

load_dotenv()
llm = initialize_llm(model_name=os.getenv("OPENROUTER_MODEL_NAME"))

math_agent = create_react_agent(
    model=llm,
    tools=[],
    name="math-agent",
    prompt="you are a professional math tutor. You are given a question and you need to answer it.",
)

tech_agent = create_react_agent(
    model=llm,
    tools=[],
    name="tech-agent",
    prompt="you are a professional tech support agent. You are given a question and you need to answer it.",
)

graph = create_supervisor(
    model=llm,
    agents=[math_agent, tech_agent],
    prompt=(
        """
            you are a supervisor agent that can delegate tasks to the agent.
            for math questions, you will delegate it to the math agent.
            for tech questions, you will delegate it to the tech agent.
            you will not answer the question directly.
        """
    ),
    add_handoff_back_messages=True,
    output_mode="last_message",
).compile(name="test-agent")
