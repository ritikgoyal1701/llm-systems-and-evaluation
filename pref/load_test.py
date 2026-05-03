# perf/load_test.py

import time
import threading
import csv
from openai import OpenAI

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

results = []
lock = threading.Lock()


def send_request(prompt, req_id):
    start_time = time.time()
    first_token_time = None
    token_count = 0

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        stream=True,
        temperature=0.7,
    )

    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            if first_token_time is None:
                first_token_time = time.time()
            token_count += 1

    end_time = time.time()

    ttft = first_token_time - start_time if first_token_time else None
    latency = end_time - start_time
    tpot = token_count / (end_time - first_token_time) if first_token_time else 0

    with lock:
        results.append({
            "req_id": req_id,
            "ttft": ttft,
            "latency": latency,
            "tpot": tpot,
            "tokens": token_count,
        })


def run_load(concurrency=3, prompt_type="short"):
    threads = []

    if prompt_type == "short":
        prompt = "What is caching?"
    else:
        prompt = "Explain distributed systems with examples in detail."

    for i in range(concurrency):
        t = threading.Thread(target=send_request, args=(prompt, i))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


def save_metrics():
    with open("perf/metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    print("Running load test...")
    run_load(concurrency=3, prompt_type="short")
    save_metrics()
    print("Metrics saved to perf/metrics.csv")