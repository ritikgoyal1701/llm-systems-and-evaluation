from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

@app.post("/generate")
def generate(prompt: str):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.7,
    )

    return {
        "prompt": prompt,
        "output": response.choices[0].message.content
    }