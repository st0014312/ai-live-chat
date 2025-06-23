# this agent is used to answer questions about the knowledge base
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from utils import initialize_llm


def create_knowledge_base_agent(
    knowledge_base_dir: str = "./knowledge_base",
    embeddings_provider: str = "huggingface",
):
    def _initialize_embeddings(provider: str = "huggingface"):
        """Initialize the embedding model."""
        if provider == "openai":
            return OpenAIEmbeddings(model_name="text-embedding-3-small")
        elif provider == "huggingface":
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def _initialize_knowledge_base(
        knowledge_base_dir: str = "./knowledge_base",
        embeddings_provider: str = "huggingface",
    ):
        """Initialize the knowledge base from markdown files."""
        # Load documents
        knowledge_base_dir = knowledge_base_dir or os.getenv("KNOWLEDGE_BASE_DIR")
        loader = DirectoryLoader(
            knowledge_base_dir,
            glob="**/*.md",  # Load all markdown files
            loader_cls=TextLoader,
        )
        documents = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        )
        splits = text_splitter.split_documents(documents)

        embeddings = _initialize_embeddings(embeddings_provider)
        # Create vector store
        vectorstore = FAISS.from_documents(splits, embeddings)

        # Save vector store if path is specified
        if os.getenv("VECTOR_STORE_PATH"):
            vectorstore.save_local(os.getenv("VECTOR_STORE_PATH"))

        return vectorstore

    llm = initialize_llm(model_name="gpt-4o-mini")
    knowledge_base = _initialize_knowledge_base(knowledge_base_dir, embeddings_provider)
    retriever = knowledge_base.as_retriever(
        search_kwargs={
            "k": 3,  # Limit to top 3 most relevant chunks
            "fetch_k": 5,  # Fetch more initially to filter
        }
    )
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_knowledge_base",
        "Search and return relevant information from the knowledge base. Only return information that directly answers the user's question.",
    )
    return create_react_agent(
        model=llm,
        tools=[retriever_tool],
        prompt="you are a knowledge base agent that is responsible for answering questions about the knowledge base",
        name="knowledge_base_agent",
    )
