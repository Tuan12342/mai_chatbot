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


        result = graph.invoke(
            {
                "user_id": user_id,
                "messages": [HumanMessage(content=text)],
            },
            config=config,
        )
        reply = result.get("reply", "")
        if reply:
            print(f"Mai: {reply}\n")


if __name__ == "__main__":
    main()
