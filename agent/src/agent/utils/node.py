from typing import List, Any
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph
from langchain_core.vectorstores import VectorStore
from langchain.tools.retriever import create_retriever_tool


def create_general_node(
    model: Any,
    vector_store: VectorStore,
) -> CompiledStateGraph:

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 3,
            "fetch_k": 5,
        },
    )

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_knowledge_base",
        "Search and return relevant information from the knowledge base. Only return information that directly answers the user's question.",
    )

    return create_react_agent(
        model=model,
        tools=[retriever_tool],
        name="general_agent",
        prompt="""
        You are a helpful customer service assistant for an e-commerce company. Your role is to provide accurate, helpful, and friendly responses to customer inquiries about various aspects of the business.

        Your responsibilities include:
        - Answering questions about return and refund policies
        - Providing contact information and support details
        - Explaining delivery and shipping options
        - Handling general customer service inquiries
        - Directing customers to appropriate resources when needed
        """,
    )
