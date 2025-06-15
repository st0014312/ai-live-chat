import os
from typing import List, Dict, Optional, Literal, TypedDict, Annotated, Any
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import uuid


class AgentState(TypedDict):
    """State for the agent."""

    thread_id: str
    messages: List[Dict[str, Any]]
    next: str
    context: Optional[List[Any]]
    error: Optional[str]


class AIChatAgent:
    def __init__(
        self,
        knowledge_base_dir: Optional[str] = None,
        llm_provider: Literal["openai", "openrouter"] = "openai",
        embedding_provider: Literal["openai", "huggingface"] = "openai",
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        """Initialize the AI chat agent."""
        # Load environment variables
        load_dotenv()

        # Set configuration from environment variables or defaults
        self.knowledge_base_dir = knowledge_base_dir or os.getenv(
            "KNOWLEDGE_BASE_DIR", "knowledge_base"
        )
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.model_name = model_name or os.getenv(
            f"{llm_provider.upper()}_MODEL_NAME",
            "gpt-4o-mini" if llm_provider == "openai" else "openai/gpt-4o-mini",
        )
        self.temperature = temperature or float(os.getenv("TEMPERATURE", "0.7"))

        # Initialize components
        self.embeddings = self._initialize_embeddings(embedding_provider)
        self.knowledge_base = self._initialize_knowledge_base()
        self.llm = self._initialize_llm(
            provider=llm_provider,
            model_name=self.model_name,
            temperature=self.temperature,
        )

        # Initialize memory saver for LangGraph
        self.memory = MemorySaver()

        # Create the agent graph
        self.agent = self._create_agent_graph()

    def _create_agent_graph(self) -> Any:
        """Create the agent graph with all necessary nodes."""

        def retrieve(state: AgentState) -> AgentState:
            """Retrieve relevant documents from knowledge base."""
            try:
                messages = state["messages"]
                last_message = messages[-1]["content"]

                # Get relevant documents
                docs = self.knowledge_base.similarity_search(last_message, k=3)

                # Update state with retrieved documents
                state["context"] = docs
                state["next"] = "generate"
            except Exception as e:
                state["error"] = f"Error retrieving documents: {str(e)}"
                state["next"] = END
            return state

        def generate(state: AgentState) -> AgentState:
            """Generate response using LLM."""
            try:
                messages = state["messages"]
                context = state.get("context", [])

                # Create a more comprehensive prompt
                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            """You are an AI assistant for an online store. Your role is to help customers with their questions about products, shipping, returns, and store information.

Guidelines:
1. Use the provided context from our knowledge base to answer questions accurately
2. If the answer isn't in the context, say you don't have that information
3. Keep responses concise (2-3 sentences maximum)
4. If the question is about previous conversation, use the chat history
5. For product questions, include relevant details like price, features, and availability
6. For shipping/returns questions, be specific about policies and timeframes
7. For store information, provide accurate contact details and hours

Context from knowledge base:
{context}

Previous conversation:
{messages}

Please provide a helpful and accurate response based on the above information.""",
                        ),
                        MessagesPlaceholder(variable_name="messages"),
                    ]
                )

                # Generate response
                response = self.llm.invoke(
                    prompt.format(
                        context="\n".join([doc.page_content for doc in context]),
                        messages=messages,
                    )
                )

                # Update state
                state["messages"].append(
                    {"role": "assistant", "content": response.content}
                )
                state["next"] = END
            except Exception as e:
                state["error"] = f"Error generating response: {str(e)}"
                state["next"] = END
            return state

        # Create the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate", generate)

        # Add edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile(checkpointer=self.memory)

    def chat(self, user_input: str, thread_id: Optional[str] = None) -> Dict:
        """Process user input and return a response."""
        try:
            # Generate or use thread ID
            thread_id = thread_id or str(uuid.uuid4())

            # Create initial state
            state = {
                "thread_id": thread_id,
                "messages": [{"role": "user", "content": user_input}],
                "next": "retrieve",
                "context": None,
                "error": None,
            }

            # Configure the run with thread ID for memory
            config = {"configurable": {"thread_id": thread_id}}

            # Run the agent with memory checkpointing
            result = self.agent.invoke(
                state,
                config=config,
            )

            if result.get("error"):
                return {
                    "response": f"Error: {result['error']}",
                    "context": [],
                    "sources": [],
                    "thread_id": thread_id,
                }

            # Get the response
            response = result["messages"][-1]["content"]

            # Get relevant context
            relevant_docs = self.get_relevant_context(user_input)

            return {
                "response": response,
                "context": relevant_docs,
                "sources": result.get("context", []),
                "thread_id": thread_id,
            }
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "context": [],
                "sources": [],
                "thread_id": thread_id,
            }

    def get_relevant_context(self, query: str, k: int = 3) -> List[Dict]:
        """Get relevant context from the knowledge base."""
        try:
            docs = self.knowledge_base.similarity_search(query, k=k)
            return [
                {"content": doc.page_content, "metadata": doc.metadata} for doc in docs
            ]
        except Exception as e:
            print(f"Error getting relevant context: {str(e)}")
            return []

    def _initialize_embeddings(self, provider: str):
        """Initialize the embedding model."""
        if provider == "openai":
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        elif provider == "huggingface":
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def _initialize_llm(
        self, provider: str, model_name: Optional[str] = None, temperature: float = 0.7
    ):
        """Initialize the LLM based on the provider."""
        if provider == "openai":
            return ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=int(os.getenv("MAX_TOKENS", "2000")),
            )
        elif provider == "openrouter":
            return ChatOpenAI(
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
                model_name=model_name,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _initialize_knowledge_base(self):
        """Initialize the knowledge base from markdown files."""
        # Load documents
        loader = DirectoryLoader(
            self.knowledge_base_dir,
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

        # Create vector store
        vectorstore = FAISS.from_documents(splits, self.embeddings)

        # Save vector store if path is specified
        if os.getenv("VECTOR_STORE_PATH"):
            vectorstore.save_local(os.getenv("VECTOR_STORE_PATH"))

        return vectorstore
