from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

def generate(prompt, stream=True):
    response = client.chat.completions.create(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        stream=stream,
    )

    if stream:
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
        print()
    else:
        print(response.choices[0].message.content)


if __name__ == "__main__":
    generate("Explain distributed systems in simple terms.")