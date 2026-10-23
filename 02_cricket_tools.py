players = {
    "MS Dhoni": 100,
    "Virat Kohli": 75,
    "Rohit Sharma": 60
}

def get_player_stats(name):
    return players.get(name)

print(get_player_stats("Virat Kohli"))

tool_schema = {
    "type": "function",
    "function": {
        "name": "get_player_stats",
        "description": "Get a player's sample score",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }
    }
}

def get_pitch_report():
    return "Spin-friendly pitch"

print(get_pitch_report())

pitch_tool_schema = {
    "type": "function",
    "function": {
        "name": "get_pitch_report",
        "description": "Get sample pitch conditions",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

tools = [
    tool_schema,
    pitch_tool_schema
]

print(len(tools))

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "What is the pitch like?"}],
    tools=tools
)

print(response.choices[0].message.tool_calls)

tool_call = response.choices[0].message.tool_calls[0]

if tool_call.function.name == "get_pitch_report":
    result = get_pitch_report()
else:
    raise ValueError("Unsupported tool")

print("Tool result:", result)

final_response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "user", "content": "What is the pitch like?"},
        response.choices[0].message,
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        }
    ]
)

print(final_response.choices[0].message.content)
