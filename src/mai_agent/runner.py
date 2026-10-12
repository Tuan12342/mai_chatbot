from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from mai_agent.graph import create_agent_graph
from mai_agent.observability import create_langfuse_handler, get_langfuse_client


def main() -> None:
    user_id = "local-user"
    graph = create_agent_graph(checkpointer=InMemorySaver())
    langfuse_handler = create_langfuse_handler()
    langfuse = get_langfuse_client()

    config = {
        "configurable": {"thread_id": user_id},
        "run_name": "mai-cli-chat",
        "metadata": {
            "langfuse_user_id": user_id,
            "langfuse_session_id": user_id,
            "langfuse_tags": ["mai-agent", "cli"],
        },
    }
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]

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
        if langfuse:
            langfuse.flush()
