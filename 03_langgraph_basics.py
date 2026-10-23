from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    score: int

def get_score(state):
    return {"score": 75}

def show_score(state):
    print("Score:", state["score"])
    return {}

def decide(state):
    if state["score"] >= 70:
        return "show_score"
    return END

graph = StateGraph(State)
graph.add_node("get_score", get_score)
graph.add_node("show_score", show_score)
graph.add_edge(START, "get_score")
graph.add_conditional_edges("get_score", decide)
graph.add_edge("show_score", END)

app = graph.compile()
app.invoke({"score": 0})