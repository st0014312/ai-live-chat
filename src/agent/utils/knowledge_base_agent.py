"""
Knowledge Base Agent Module

This module provides an agent specialized in answering questions from a knowledge base
using vector embeddings and retrieval. It supports multiple embedding providers and
can handle various document formats.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool

from src.agent.utils.utils import initialize_llm

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_K = 3
DEFAULT_FETCH_K = 5


class KnowledgeBaseAgentError(Exception):
    """Exception raised when knowledge base agent operations fail."""

    pass


def create_knowledge_base_agent(
    knowledge_base_dir: str = "./knowledge_base",
    embeddings_provider: str = "huggingface",
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    top_k: int = DEFAULT_TOP_K,
    fetch_k: int = DEFAULT_FETCH_K,
) -> Any:
    """
    Create a knowledge base agent for answering questions from documents.

    Args:
        knowledge_base_dir: Directory containing knowledge base files
        embeddings_provider: Provider to use for embeddings ("openai" or "huggingface")
        model_name: Name of the model to use
        temperature: Temperature for response generation
        chunk_size: Size of text chunks for splitting documents
        chunk_overlap: Overlap between text chunks
        top_k: Number of top results to retrieve
        fetch_k: Number of results to fetch before filtering

    Returns:
        Configured knowledge base agent

    Raises:
        KnowledgeBaseAgentError: If agent creation fails
    """
    try:
        # Set defaults
        model_name = model_name or os.getenv("KB_AGENT_MODEL_NAME", DEFAULT_MODEL_NAME)
        temperature = temperature or float(
            os.getenv("KB_AGENT_TEMPERATURE", DEFAULT_TEMPERATURE)
        )
        chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
        chunk_overlap = chunk_overlap or int(
            os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)
        )

        logger.info(
            f"Creating knowledge base agent with model: {model_name}, provider: {embeddings_provider}"
        )

        # Initialize LLM
        llm = initialize_llm(
            model_name=model_name,
            temperature=temperature,
            llm_provider="openrouter",  # Knowledge base agent typically uses openrouter
        )

        # Initialize knowledge base
        knowledge_base = _initialize_knowledge_base(
            knowledge_base_dir=knowledge_base_dir,
            embeddings_provider=embeddings_provider,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Create retriever
        retriever = knowledge_base.as_retriever(
            search_kwargs={
                "k": top_k,
                "fetch_k": fetch_k,
            }
        )

        # Create retriever tool
        retriever_tool = create_retriever_tool(
            retriever,
            "retrieve_knowledge_base",
            "Search and return relevant information from the knowledge base. Only return information that directly answers the user's question.",
        )

        # Create system prompt
        system_prompt = _create_kb_system_prompt()

        # Create agent
        agent = create_react_agent(
            model=llm,
            tools=[retriever_tool],
            prompt=system_prompt,
            name="knowledge_base_agent",
        )

        logger.info("Knowledge base agent created successfully")
        return agent

    except Exception as e:
        logger.error(f"Failed to create knowledge base agent: {str(e)}")
        raise KnowledgeBaseAgentError(
            f"Knowledge base agent creation failed: {str(e)}"
        ) from e


def _initialize_embeddings(provider: str = "huggingface") -> Any:
    """
    Initialize the embedding model.

    Args:
        provider: Embedding provider to use

    Returns:
        Configured embedding model

    Raises:
        ValueError: If provider is not supported
    """
    try:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise KnowledgeBaseAgentError(
                    "OPENAI_API_KEY environment variable not set"
                )

            return OpenAIEmbeddings(
                model_name="text-embedding-3-small", openai_api_key=api_key
            )

        elif provider == "huggingface":
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )

        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {str(e)}")
        raise KnowledgeBaseAgentError(
            f"Embedding initialization failed: {str(e)}"
        ) from e


def _initialize_knowledge_base(
    knowledge_base_dir: str = "./knowledge_base",
    embeddings_provider: str = "huggingface",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> FAISS:
    """
    Initialize the knowledge base from document files.

    Args:
        knowledge_base_dir: Directory containing knowledge base files
        embeddings_provider: Provider to use for embeddings
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks

    Returns:
        Configured FAISS vector store

    Raises:
        KnowledgeBaseAgentError: If initialization fails
    """
    try:
        # Resolve knowledge base directory
        kb_path = Path(
            knowledge_base_dir or os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base")
        )

        if not kb_path.exists():
            raise KnowledgeBaseAgentError(
                f"Knowledge base directory not found: {kb_path}"
            )

        logger.info(f"Loading documents from: {kb_path}")

        # Load documents
        loader = DirectoryLoader(
            str(kb_path),
            glob="**/*.md",  # Load all markdown files
            loader_cls=TextLoader,
            show_progress=True,
        )

        documents = loader.load()

        if not documents:
            logger.warning(f"No documents found in {kb_path}")
            # Create empty vector store
            embeddings = _initialize_embeddings(embeddings_provider)
            return FAISS.from_texts(["No documents available"], embeddings)

        logger.info(f"Loaded {len(documents)} documents")

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        splits = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(splits)} chunks")

        # Initialize embeddings
        embeddings = _initialize_embeddings(embeddings_provider)

        # Create vector store
        vectorstore = FAISS.from_documents(splits, embeddings)

        # Save vector store if path is specified
        vector_store_path = os.getenv("VECTOR_STORE_PATH")
        if vector_store_path:
            try:
                vectorstore.save_local(vector_store_path)
                logger.info(f"Vector store saved to: {vector_store_path}")
            except Exception as e:
                logger.warning(f"Failed to save vector store: {str(e)}")

        return vectorstore

    except Exception as e:
        logger.error(f"Failed to initialize knowledge base: {str(e)}")
        raise KnowledgeBaseAgentError(
            f"Knowledge base initialization failed: {str(e)}"
        ) from e


def _create_kb_system_prompt() -> str:
    """
    Create the system prompt for the knowledge base agent.

    Returns:
        Formatted system prompt
    """
    return """You are a knowledgeable customer service agent specializing in answering questions about store policies, products, and general information.

**Your Responsibilities:**
- Answer questions based on information from the knowledge base
- Provide accurate and helpful responses about store policies, shipping, returns, etc.
- Use the knowledge base retrieval tool to find relevant information
- Only provide information that is directly supported by the knowledge base
- If information is not available, clearly state that you don't have that information

**Response Guidelines:**
- Be friendly and professional in your tone
- Provide clear, concise answers
- Include relevant details when available
- If you need to search the knowledge base, do so before responding
- Always cite the source of information when possible

**Topics You Can Handle:**
- Store policies and procedures
- Shipping information and costs
- Return and refund policies
- Product information and availability
- FAQ responses
- General customer service questions

**When You Cannot Answer:**
- If the question requires database access (customer data, orders, etc.)
- If the information is not in the knowledge base
- If the question is outside your scope

Remember: Always prioritize accuracy and customer satisfaction in your responses."""


def validate_knowledge_base_directory(directory: str) -> Dict[str, Any]:
    """
    Validate a knowledge base directory.

    Args:
        directory: Directory path to validate

    Returns:
        Validation results dictionary
    """
    try:
        kb_path = Path(directory)

        if not kb_path.exists():
            return {"valid": False, "error": f"Directory does not exist: {directory}"}

        if not kb_path.is_dir():
            return {"valid": False, "error": f"Path is not a directory: {directory}"}

        # Count markdown files
        md_files = list(kb_path.rglob("*.md"))

        return {
            "valid": True,
            "directory": str(kb_path),
            "markdown_files": len(md_files),
            "total_files": len(list(kb_path.rglob("*"))),
            "files": [str(f) for f in md_files[:10]],  # First 10 files
        }

    except Exception as e:
        logger.error(f"Failed to validate knowledge base directory: {str(e)}")
        return {"valid": False, "error": str(e)}


def get_knowledge_base_stats(agent: Any) -> Dict[str, Any]:
    """
    Get statistics about the knowledge base.

    Args:
        agent: Knowledge base agent instance

    Returns:
        Statistics dictionary
    """
    try:
        # Extract information from the agent's tools
        tools = agent.tools if hasattr(agent, "tools") else []

        stats = {
            "tools_available": len(tools),
            "retriever_tool": False,
            "vector_store_type": "unknown",
        }

        # Check for retriever tool
        for tool in tools:
            if hasattr(tool, "name") and tool.name == "retrieve_knowledge_base":
                stats["retriever_tool"] = True
                break

        return stats

    except Exception as e:
        logger.error(f"Failed to get knowledge base stats: {str(e)}")
        return {"error": str(e)}
