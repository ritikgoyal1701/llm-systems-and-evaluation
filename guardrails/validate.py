import re
from openai import OpenAI

# ---------------------------
# CONFIG
# ---------------------------

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

# Deterministic settings (CRITICAL for Part D)
DETERMINISTIC = dict(
    temperature=0,
    top_p=1,
    max_tokens=10,
)

# ---------------------------
# GENERATION
# ---------------------------

def generate(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        **DETERMINISTIC,
    )
    return response.choices[0].message.content.strip()


# ---------------------------
# NORMALIZATION + VALIDATION
# ---------------------------

MCQ_REGEX = re.compile(r"^[A-D]$")

def normalize_output(output: str) -> str:
    """
    Normalize model output to a single token (A/B/C/D)
    """
    if not output:
        return ""

    token = output.strip().split()[1]  # take first token
    token = token.replace(".", "").strip()
    return token


def validate_mcq(output: str) -> bool:
    return bool(MCQ_REGEX.match(output))


def safe_output(output: str) -> str:
    """
    Ensures output conforms to schema
    """
    norm = normalize_output(output)

    if validate_mcq(norm):
        return norm

    # fallback (simple deterministic fallback)
    return "A"


# ---------------------------
# TEST 1 — DETERMINISM
# ---------------------------

def test_determinism():
    prompt = "Answer with ONLY one letter (A/B/C/D). What is idempotency?"

    outputs = [generate(prompt) for _ in range(3)]

    print("\nDeterminism Test Outputs:")
    for i, o in enumerate(outputs):
        print(f"{i}: {o}")

    all_same = all(o == outputs[0] for o in outputs)
    print("Deterministic:", all_same)

    return all_same


# ---------------------------
# TEST 2 — VALIDATION
# ---------------------------

def test_validation():
    prompt = (
        "Choose the correct option. Respond ONLY with A, B, C, or D.\n"
        "Q: Idempotency means?\n"
        "A) Repeatable without side effects\n"
        "B) Faster execution\n"
        "C) Parallel processing\n"
        "D) Load balancing\n"
        "Answer:"
    )

    raw_output = generate(prompt)

    print("\nValidation Test:")
    print("Raw output:", raw_output)

    normalized = normalize_output(raw_output)
    print("Normalized:", normalized)

    is_valid = validate_mcq(normalized)
    print("Valid:", is_valid)

    safe = safe_output(raw_output)
    print("Final (safe):", safe)

    return is_valid


# ---------------------------
# MAIN
# ---------------------------

if __name__ == "__main__":
    print("=== Guardrails Validation ===")

    det = test_determinism()
    val = test_validation()

    print("\nSummary:")
    print({
        "deterministic": det,
        "validation_passed": val
    })