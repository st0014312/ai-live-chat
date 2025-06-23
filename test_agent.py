from agent import Agent
import argparse
import uuid


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test AI Chat Agent with different providers"
    )
    parser.add_argument(
        "--llm",
        choices=["openai", "openrouter"],
        default="openrouter",
        help="LLM provider to use (default: openai)",
    )
    parser.add_argument(
        "--embeddings",
        choices=["openai", "huggingface"],
        default="huggingface",
        help="Embedding provider to use (default: openai)",
    )
    parser.add_argument("--model", type=str, help="Specific model name to use")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for model responses (default: 0.7)",
    )

    args = parser.parse_args()

    # Initialize the agent with specified providers
    agent = Agent(
        llm_provider=args.llm,
        embedding_provider=args.embeddings,
        model_name=args.model,
        temperature=args.temperature,
    )

    print(f"🤖 AI Chat Agent initialized!")
    print(f"LLM Provider: {args.llm}")
    print(f"Embedding Provider: {args.embeddings}")
    if args.model:
        print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print("\nType 'quit' to exit.")
    print("Ask any question about the store's policies, shipping, or FAQs.")
    thread_id = str(uuid.uuid4())
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        # Check for quit command
        if user_input.lower() == "quit":
            print("Goodbye! 👋")
            break

        try:
            # Get response from agent
            response = agent.chat(user_input, thread_id)
            # print(response["response"])

        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()
