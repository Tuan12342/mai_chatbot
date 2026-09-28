from langchain_core.messages import HumanMessage

from mai_agent.graph import create_agent_graph


def main() -> None:
    graph = create_agent_graph()
    print("Mai đã sẵn sàng.")

    while True:
        text = input("Bạn: ").strip()
        if text.lower() in {"exit", "quit", "thoát"}:
            break
        if not text:
            continue

        result = graph.invoke(
            {
                "user_id": "local-user",
                "messages": [HumanMessage(content=text)],
            }
        )
        print(f"Mai: {result['reply']}\n")


if __name__ == "__main__":
    main()
