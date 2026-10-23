
import json
import os

from dotenv import load_dotenv
from groq import Groq


# 1. Sample cricket data (not real match statistics)
players = {
    "MS Dhoni": 100,
    "Virat Kohli": 75,
    "Rohit Sharma": 60
}


# 2. Actual Python functions
def get_player_stats(name):
    return players.get(name)


def get_pitch_report():
    return "Spin-friendly pitch"


# 3. Describe both functions to the AI
tool_schema = {
    "type": "function",
    "function": {
        "name": "get_player_stats",
        "description": "Get a player's sample score",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Full name of the player"
                }
            },
            "required": ["name"]
        }
    }
}

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


# 4. Connect to Groq
load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

model = "openai/gpt-oss-20b"
question = "What is Virat Kohli's sample score?"


# 5. Ask the AI to choose a tool
messages = [
    {"role": "user", "content": question}
]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools
)

assistant_message = response.choices[0].message


# 6. Execute the requested Python function
if assistant_message.tool_calls:
    messages.append(assistant_message)

    for tool_call in assistant_message.tool_calls:
        tool_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if tool_name == "get_pitch_report":
            result = get_pitch_report()

        elif tool_name == "get_player_stats":
            result = get_player_stats(args["name"])

        else:
            raise ValueError(f"Unsupported tool: {tool_name}")

        print("Tool called:", tool_name)
        print("Tool result:", result)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })

    # 7. Send tool results back to the AI
    final_response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    print("Final AI answer:",
          final_response.choices[0].message.content)

else:
    print("AI answer:", assistant_message.content)
