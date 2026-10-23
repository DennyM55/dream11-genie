from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    message: str

def first(state):
    print("First step")
    return state

graph = StateGraph(State)
graph.add_node("first", first)
graph.add_edge(START, "first")
graph.add_edge("first", END)

app = graph.compile()
app.invoke({"message": "Hello"})