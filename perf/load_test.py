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
    start = time.time()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=0.7,
    )

    end = time.time()

    text = response.choices[0].message.content or ""
    token_count = len(text.split())

    latency = end - start

    # approximate TTFT (no streaming)
    ttft = latency

    tpot = token_count / latency if latency > 0 else 0

    with lock:
        results.append({
            "req_id": req_id,
            "ttft": ttft,
            "latency": latency,
            "tpot": tpot,
            "tokens": token_count
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
    run_load(concurrency=1, prompt_type="short")
    save_metrics()
    print("Metrics saved to perf/metrics.csv")