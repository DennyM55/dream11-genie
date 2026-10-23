
import json
import os

from dotenv import load_dotenv
from groq import Groq


# 1. SAMPLE CRICKET DATA
# These are made-up scores for learning.
players = {
    "MS Dhoni": 100,
    "Virat Kohli": 75,
    "Rohit Sharma": 60
}


# 2. ACTUAL PYTHON FUNCTIONS
def get_player_stats(name):
    return players.get(name)


def get_pitch_report():
    return "Spin-friendly pitch"


# 3. TOOL DESCRIPTIONS FOR THE AI
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_player_stats",
            "description": "Get the sample score of a cricket player.",
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_pitch_report",
            "description": "Get the sample cricket pitch conditions.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


# 4. CONNECT TO GROQ
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing from .env")

client = Groq(api_key=api_key)

model = "openai/gpt-oss-20b"


# 5. USER QUESTION
question = (
    "Check Virat Kohli's sample score and "
    "the pitch conditions. Use both tools."
)

messages = [
    {
        "role": "system",
        "content": (
            "You are a cricket assistant. "
            "Use only the provided sample data. "
            "Do not invent cricket statistics. "
            "Call the appropriate tools before answering."
        )
    },
    {
        "role": "user",
        "content": question
    }
]


# 6. AGENT LOOP
MAX_STEPS = 5

for step in range(MAX_STEPS):

    print(f"\n--- Agent Step {step + 1} ---")

    # Ask AI what to do next
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    assistant_message = response.choices[0].message

    # If no tools requested, the AI is finished
    if not assistant_message.tool_calls:
        print("\nFinal AI Answer:")
        print(assistant_message.content)
        break

    # Save the AI's tool requests
    messages.append(assistant_message)

    # Execute each requested tool
    for tool_call in assistant_message.tool_calls:

        tool_name = tool_call.function.name
        arguments = json.loads(
            tool_call.function.arguments
        )

        print("Tool requested:", tool_name)
        print("Arguments:", arguments)

        # Execute the matching Python function
        if tool_name == "get_player_stats":
            player_name = arguments["name"]
            result = get_player_stats(player_name)

        elif tool_name == "get_pitch_report":
            result = get_pitch_report()

        else:
            result = {
                "error": f"Unknown tool: {tool_name}"
            }

        print("Tool result:", result)

        # Return the result to the AI
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })

else:
    print("\nAgent stopped: maximum steps reached.")
