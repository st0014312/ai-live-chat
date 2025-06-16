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
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
import uuid
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import HumanMessage
from langchain.tools.retriever import create_retriever_tool


class AgentState(TypedDict):
    """State for the agent."""

    messages: Annotated[list[AnyMessage], add_messages]
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
        self.retriever_tool = self._initialize_retriever_tool()
        self.llm = self._initialize_llm(
            provider=llm_provider,
            model_name=self.model_name,
            temperature=self.temperature,
        )

        # Initialize memory saver for LangGraph
        self.memory = MemorySaver()

        # Create the agent graph
        self.agent = self._create_agent_graph()

    def _create_agent_graph(self) -> AgentState:
        """Create the agent graph with all necessary nodes."""

        def generate_query_or_response(state: AgentState) -> AgentState:
            """Generate query or response based on the state."""
            try:
                response = self.llm.bind_tools([self.retriever_tool]).invoke(
                    state["messages"]
                )
                state["messages"].append(response)
                return state
            except Exception as e:
                state["error"] = f"Error generating response: {str(e)}"
                return state

        def generate(state: AgentState) -> AgentState:
            """Generate response using LLM."""
            try:
                print("state", state)
                messages = state["messages"]
                # Get the content from the tool message
                tool_message = state["messages"][-1]

                # Check if we have valid content from the tool
                if not tool_message.content:
                    # If no content, generate a response indicating no information found
                    response = self.llm.invoke(
                        "I couldn't find specific information about that in our knowledge base. Could you please rephrase your question or ask about something else?"
                    )
                    state["messages"].append(response)
                    return state

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
4. Maintain awareness of user identity and preferences mentioned in the conversation
5. For product questions, include relevant details like price, features, and availability
6. For shipping/returns questions, be specific about policies and timeframes
7. For store information, provide accurate contact details and hours

Context from knowledge base:
{context}

Previous conversation:
{formatted_messages}""",
                        ),
                    ]
                )

                # Format messages for better context
                formatted_messages = []
                for msg in messages[:-1]:  # Exclude the last message (tool response)
                    if hasattr(msg, "content"):
                        role = "user" if isinstance(msg, HumanMessage) else "assistant"
                        formatted_messages.append(f"{role}: {msg.content}")

                # Generate response
                response = self.llm.invoke(
                    prompt.format(
                        context=tool_message.content,  # Use the tool response content directly
                        formatted_messages="\n".join(formatted_messages),
                    )
                )

                # Update state
                state["messages"].append(response)
            except Exception as e:
                state["error"] = f"Error generating response: {str(e)}"
            return state

        # Create the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("generate_query_or_response", generate_query_or_response)
        workflow.add_node("retrieve", ToolNode([self.retriever_tool]))
        workflow.add_node("generate", generate)

        # Add edges
        workflow.add_edge(START, "generate_query_or_response")
        workflow.add_conditional_edges(
            "generate_query_or_response",
            tools_condition,
            {"tools": "retrieve", END: END},
        )
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
                "messages": [{"role": "user", "content": user_input}],
            }

            # Configure the run with thread ID for memory
            config = {"configurable": {"thread_id": thread_id}}

            # Run the agent with memory checkpointing
            result = self.agent.invoke(
                state,
                config=config,
            )

            if result.get("error"):
                return {"response": f"Error: {result['error']}"}

            # Get the response
            response = result["messages"][-1]

            return {"response": response}
        except Exception as e:
            return {"response": f"Error: {str(e)}"}

    def _initialize_embeddings(self, provider: str):
        """Initialize the embedding model."""
        if provider == "openai":
            return OpenAIEmbeddings(model_name="text-embedding-3-small")
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

    def _initialize_retriever_tool(self):
        """Initialize the retriever."""
        retriever = self.knowledge_base.as_retriever(
            search_kwargs={
                "k": 3,  # Limit to top 3 most relevant chunks
                "fetch_k": 5,  # Fetch more initially to filter
            }
        )
        return create_retriever_tool(
            retriever,
            "retrieve_knowledge_base",
            "Search and return relevant information from the knowledge base. Only return information that directly answers the user's question.",
        )

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
