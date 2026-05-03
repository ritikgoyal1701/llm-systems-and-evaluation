# improve/infer.py

import argparse
from collections import Counter
from openai import OpenAI

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

# ---------------------------
# FEW-SHOT EXAMPLES (KEY PART)
# ---------------------------

FEW_SHOTS = """
Example 1:
Context: A man is cooking in the kitchen.
A. He plays football
B. He cuts vegetables
C. He drives a car
D. He sleeps
Answer: B

Example 2:
Context: A person is typing on a laptop.
A. They are coding
B. They are swimming
C. They are running
D. They are dancing
Answer: A
"""

# ---------------------------
# PROMPT BUILDER
# ---------------------------

def build_prompt(context, choices, use_fewshot=True):
    prompt = ""

    if use_fewshot:
        prompt += FEW_SHOTS + "\n\n"

    prompt += "Choose the most logical continuation.\n\n"
    prompt += f"Context:\n{context}\n\n"

    for i, c in enumerate(choices):
        prompt += f"{chr(65+i)}. {c}\n"

    prompt += "\nAnswer (ONLY A/B/C/D):"
    return prompt


# ---------------------------
# NORMALIZATION
# ---------------------------

def normalize(output):
    if not output:
        return "A"

    token = output.strip().split()[0]
    token = token.replace(".", "").strip()

    if token not in ["A", "B", "C", "D"]:
        return "A"

    return token


# ---------------------------
# GENERATION
# ---------------------------

def single_run(prompt, temperature=0):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=temperature,
        top_p=0.9,
    )

    out = resp.choices[0].message.content.strip()
    return normalize(out)


# ---------------------------
# SELF-CONSISTENCY
# ---------------------------

def generate_answer(prompt, k=3, temperature=0.7):
    answers = []

    for _ in range(k):
        ans = single_run(prompt, temperature=temperature)
        answers.append(ans)

    final = Counter(answers).most_common(1)[0][0]
    return final


# ---------------------------
# SAMPLE DATA (SMALL LOCAL TEST)
# ---------------------------

SAMPLE_DATA = [
    {
        "context": "A man is playing a guitar on stage.",
        "choices": [
            "He starts cooking food",
            "He performs music",
            "He drives a car",
            "He goes to sleep"
        ],
        "answer": "B"
    },
    {
        "context": "A woman is reading a book in the library.",
        "choices": [
            "She is swimming",
            "She is studying",
            "She is running",
            "She is dancing"
        ],
        "answer": "B"
    }
]


# ---------------------------
# EVALUATION
# ---------------------------

def run(mode="baseline"):
    correct = 0

    for i, item in enumerate(SAMPLE_DATA):
        prompt = build_prompt(
            item["context"],
            item["choices"],
            use_fewshot=(mode == "improved")
        )

        if mode == "baseline":
            pred = single_run(prompt, temperature=0)
        else:
            pred = generate_answer(prompt, k=3, temperature=0.7)

        print(f"\nSample {i}")
        print("Prediction:", pred)
        print("Ground Truth:", item["answer"])

        if pred == item["answer"]:
            correct += 1

    acc = correct / len(SAMPLE_DATA)
    print(f"\nMode: {mode} | Accuracy: {acc:.2f}")


# ---------------------------
# MAIN
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "improved"], default="baseline")

    args = parser.parse_args()

    run(mode=args.mode)