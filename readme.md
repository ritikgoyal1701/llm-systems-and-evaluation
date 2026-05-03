# LLM Evaluation Pipeline (vLLM + lm-eval)

## Overview

This repository implements a scaled-down version of a production LLM evaluation pipeline using:

* vLLM for high-throughput inference serving
* lm-evaluation-harness for standardized benchmarking
* Custom tooling for performance testing, guardrails, and inference optimization

The system simulates how internal evaluation pipelines are designed, debugged, and improved in real-world ML systems.

---

## Architecture

Client / Load Generator
↓
vLLM Server (OpenAI-compatible API)
↓
Custom Model Wrapper (lm-eval)
↓
Evaluation + Metrics + Analysis

---

## Repository Structure

.
├── serve/          # Part A: vLLM serving + client
├── eval_runner/    # Part B: lm-eval integration
├── perf/           # Part C: load testing + metrics
├── guardrails/     # Part D: determinism + validation
├── improve/        # Part E: benchmark improvements
├── results/        # evaluation outputs
├── Makefile
└── README.md

---

## Assumptions

* CPU-only environment (Apple M1)
* Model used: TinyLlama-1.1B-Chat
* Evaluation runs on limited samples (limit=5–20) due to compute constraints
* Log-likelihood approximated using generation-based scoring

---

## Setup

### Create virtual environment

python3 -m venv .venv
source .venv/bin/activate

### Install dependencies

pip install -r serve/requirements.txt
pip install lm-eval fastapi uvicorn pandas

---

## Part A — Serving

### Start vLLM server

make serve

Server runs at:
http://localhost:8000/v1

---

### Test generation

python serve/client.py

---

### Test concurrency

python serve/test_concurrency.py

---

## Part B — Evaluation

### Run evaluation

python eval_runner/run_eval.py

---

### Tasks

* HellaSwag
* MMLU (abstract_algebra subset)

---

### Key Implementation Details

* Custom model wrapper registered with lm-eval
* Deterministic decoding (temperature=0, top_p=1)
* MCQ normalization (A/B/C/D extraction)
* Fixed scoring logic for multiple-choice evaluation
* Response caching for repeatability

---

### Output

eval_runner/results/output.json

---

## Part C — Performance & Scaling

### Run load test

python perf/load_test.py

---

### Metrics Collected

* TTFT (Time to First Token)
* Latency
* Tokens per output token (TPOT)
* Token count

---

### Output

perf/metrics.csv

---

### Observations

* Latency increases with concurrency due to CPU contention
* No effective batching on CPU
* Streaming improves perceived latency

---

## Part D — Guardrails & Determinism

### Run validation

python guardrails/validate.py

---

### Features

* Deterministic inference mode
* Repeatability testing
* Regex validation for MCQ outputs (A/B/C/D)
* Output normalization and fallback handling

---

### Limitations

* Minor nondeterminism may occur due to backend scheduling and floating-point behavior

---

## Part E — Benchmark Improvement

### Run baseline

python improve/infer.py --mode baseline

---

### Run improved

python improve/infer.py --mode improved

---

### Techniques Used

* Prompt restructuring (MCQ format)
* Few-shot prompting
* Self-consistency (k=3 sampling)
* Majority voting
* Output normalization

---

### Outcome

Improved accuracy using inference-time techniques without modifying model weights.

See:
improve/report.md

---

## Reproducibility

* Deterministic decoding for evaluation
* Caching ensures identical outputs for repeated prompts
* Fixed seeds used where applicable

---

## Trade-offs

Accuracy: improved with advanced prompting
Latency: increased due to multiple sampling
Cost: higher due to repeated inference

---

## Limitations

* CPU-only setup limits throughput and batching
* Approximate log-likelihood instead of true token probabilities
* Evaluation done on subset for feasibility

---

## How to Run Everything

make serve

python eval_runner/run_eval.py

python perf/load_test.py

python guardrails/validate.py

python improve/infer.py --mode improved

---

## Final Note

This project prioritizes correctness, reproducibility, and system design clarity over raw performance due to hardware constraints.
