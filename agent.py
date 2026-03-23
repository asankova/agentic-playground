from openai import OpenAI
import os

api_key = os.getenv("PENNYWISE_GROQAI_API_KEY")
if not api_key:
    raise ValueError("Set PENNYWISE_GROQAI_API_KEY in your environment before running.")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models. Respond in JSON.",
        }
    ],
    model="openai/gpt-oss-20b"
)

print(response.choices[0].message.content)
