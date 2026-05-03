from client import generate

prompts = [
    "What is CAP theorem?",
    "Explain load balancing",
    "What is rate limiting?"
]

for p in prompts:
    print(f"\nPrompt: {p}")
    generate(p)