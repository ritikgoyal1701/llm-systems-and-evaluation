from concurrent.futures import ThreadPoolExecutor
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def single_request(i):
    print(f"Starting request {i}")

    start = time.time()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "What is caching?"}],
            max_tokens=10,
        )

        latency = time.time() - start
        print(f"Completed request {i} in {latency:.2f}s")

        return latency

    except Exception as e:
        print(f"Error in request {i}: {e}")
        return None


def run_test(n=3):  # reduce concurrency for CPU
    print("Starting load test...\n")

    with ThreadPoolExecutor(max_workers=n) as executor:
        results = list(executor.map(single_request, range(n)))

    print("\nSummary:")
    valid = [r for r in results if r]
    if valid:
        print(f"Avg latency: {sum(valid)/len(valid):.2f}s")


if __name__ == "__main__":
    run_test(3)