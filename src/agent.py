
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from src.db_tools import build_database_tools


# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"

load_dotenv(ENV_FILE)


def get_required_env(name: str) -> str:
    """Return a required environment variable."""
    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(
            f"{name} is missing or empty in {ENV_FILE}"
        )

    return value


# ============================================================
# 2. BUILD THE MULTI-TOOL AGENT
# ============================================================

def build_agent():
    # CheaperInference configuration
    api_key = get_required_env("API_KEY")

    base_url = os.getenv(
        "BASE_URL",
        "https://api.cheaperinference.com/v1",
    ).strip()

    model_name = os.getenv("MODEL", "glm-5.3").strip()

    # Tavily configuration
    tavily_key = get_required_env("TAVILY_API_KEY")

    # LangChain ChatOpenAI using the compatible API endpoint
    llm = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=0,
        timeout=120,
        max_retries=2,
    )

    # SQLite database tools
    database_tools = build_database_tools(llm)

    # Tavily web search tool
    # TavilySearch reads TAVILY_API_KEY from the environment.
    web_tool = TavilySearch(max_results=5)
    web_tool.name = "WebSearchTool"
    web_tool.description = (
        "Search the internet for general knowledge and current "
        "information, especially Bangladesh government policies, "
        "healthcare, news, and other information not available "
        "in the local databases. Summarize relevant findings and "
        "include source URLs when available."
    )

    all_tools = database_tools + [web_tool]

    # Create the LangChain 1.x agent
    agent = create_agent(
        model=llm,
        tools=all_tools,
        system_prompt=(
            "You are a helpful Bangladesh Multi-Tool AI Assistant.\n\n"

            "TOOL ROUTING RULES:\n"
            "1. For educational institutions and institutional records, "
            "use the institutions database tool.\n"
            "2. For hospital records, locations, and related database "
            "information, use the hospitals database tool.\n"
            "3. For restaurant records and restaurant-related database "
            "questions, use the restaurants database tool.\n"
            "4. For general knowledge and current information, use "
            "WebSearchTool when web research is appropriate.\n"
            "5. If a question requires multiple sources, use the "
            "appropriate tools as needed.\n\n"

            "ANSWERING RULES:\n"
            "- Never invent database records, addresses, phone numbers, "
            "statistics, or search results.\n"
            "- Use database results for information stored in the "
            "local databases.\n"
            "- If a database has no matching record, explain that "
            "no matching record was found.\n"
            "- Clearly distinguish database results from web findings.\n"
            "- Include useful source URLs for web-based answers "
            "when available.\n"
            "- If the available information is insufficient, say so.\n"
            "- Respond clearly and helpfully."
        ),
    )

    print(f"LLM provider: CheaperInference")
    print(f"Model: {model_name}")
    print(f"Database tools loaded: {len(database_tools)}")
    print("Web search tool: Tavily")

    return agent


# ============================================================
# 3. INTERACTIVE CHAT
# ============================================================

def main():
    print("=" * 60)
    print("       BANGLADESH MULTI-TOOL AI AGENT")
    print("=" * 60)
    print("Provider : CheaperInference")
    print("Model    : glm-5.3")
    print("Commands : type 'exit' or 'quit' to stop")

    try:
        agent = build_agent()
    except Exception as exc:
        print(f"\nAgent initialization failed: {exc}")
        return

    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            response = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": question,
                        }
                    ]
                }
            )

            messages = response.get("messages", [])

            # Find the latest assistant response.
            answer = None

            for message in reversed(messages):
                if (
                    getattr(message, "type", "") == "ai"
                    and getattr(message, "content", None)
                ):
                    answer = message.content
                    break

            if answer is not None:
                print(f"\nAgent: {answer}")
            else:
                print("\nAgent: No answer was returned.")

        except Exception as exc:
            print(f"\nRequest failed: {exc}")
            print(
                "Check your API configuration, provider balance, "
                "model access, Tavily key, and database files."
            )


if __name__ == "__main__":
    main()
