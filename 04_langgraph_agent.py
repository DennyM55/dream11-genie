from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

class State(TypedDict):
    question: str
    answer: str

def ai_node(state):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "user",
             "content": state["question"]}
        ]
    )
    return {
        "answer": response.choices[0].message.content
    }

graph = StateGraph(State)
graph.add_node("ai", ai_node)
graph.add_edge(START, "ai")
graph.add_edge("ai", END)

app = graph.compile()

result = app.invoke({
    "question": "What is cricket?",
    "answer": ""
})

print(result["answer"])