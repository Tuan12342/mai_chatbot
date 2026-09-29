from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from mai_agent.graph import create_agent_graph


def main() -> None:
    user_id = "local-user"
    graph = create_agent_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": user_id}}
    print("Mai đã sẵn sàng.")

    while True:
        text = input("Bạn: ").strip()
        if text.lower() in {"exit", "quit", "thoát"}:
            break
        if not text:
            continue

        result = graph.invoke(
            {
                "user_id": user_id,
                "messages": [HumanMessage(content=text)],
            },
            config=config,
        )
        print(f"Mai: {result['reply']}\n")


if __name__ == "__main__":
    main()
