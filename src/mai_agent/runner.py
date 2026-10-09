from langchain_core.messages import HumanMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import InMemorySaver

from mai_agent.graph import create_agent_graph


def main() -> None:
    user_id = "local-user"
    graph = create_agent_graph(checkpointer=InMemorySaver())
    langfuse_handler = CallbackHandler()
    langfuse = get_client()

    config = {
        "configurable": {"thread_id": user_id},
        "callbacks": [langfuse_handler],
        "run_name": "mai-cli-chat",
        "metadata": {
            "langfuse_user_id": user_id,
            "langfuse_session_id": user_id,
            "langfuse_tags": ["mai-agent", "cli"],
        },
    }

    print("Mai đã sẵn sàng.")

    try:
        while True:
            text = input("Bạn: ").strip()

            result = graph.invoke(
                {
                    "user_id": user_id,
                    "messages": [HumanMessage(content=text)],
                },
                config=config,
            )

            if reply := result.get("reply", ""):
                print(f"Mai: {reply}\n")
    finally:
        langfuse.flush()
