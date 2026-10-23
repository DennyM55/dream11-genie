
import os

from dotenv import load_dotenv
from groq import Groq


# 1. Load API key from .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing")


# 2. Create Groq client
client = Groq(api_key=api_key)
print("Groq client created")


# 3. Ask the AI model a question
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "What is cricket?"
        }
    ]
)


# 4. Print the AI's answer
answer = response.choices[0].message.content
print("AI Answer:", answer)