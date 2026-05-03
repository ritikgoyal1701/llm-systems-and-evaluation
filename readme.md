# LLM Evaluation Pipeline (vLLM + lm-eval)

## Overview

This repository implements a scaled-down version of a production LLM evaluation pipeline using:

- vLLM for inference serving  
- lm-evaluation-harness for benchmarking  
- Custom tooling for performance testing, guardrails, and optimization  

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
├── serve/ # Part A: vLLM serving + client
├── eval_runner/ # Part B: lm-eval integration
├── perf/ # Part C: load testing + metrics
├── guardrails/ # Part D: determinism + validation
├── improve/ # Part E: benchmark improvement
├── results/ # evaluation outputs
├── Makefile
└── README.md


---

## Assumptions

- CPU-only environment (Apple M1)
- Model: TinyLlama-1.1B-Chat
- Evaluation done on small subsets (`limit=5–20`)
- Log-likelihood approximated via generation

---

## Setup

### Create environment

python3 -m venv .venv
source .venv/bin/activate


### Install dependencies

pip install -r serve/requirements.txt
pip install lm-eval fastapi uvicorn pandas


---

## Part A — Serving

Start server:
make serve


Server runs at:
http://localhost:8000/v1


Test:

python serve/client.py
python serve/test_concurrency.py


---

## Part B — Evaluation

Run:
python eval_runner/run_eval.py


Tasks:
- HellaSwag  
- MMLU (abstract_algebra subset)

Output:

eval_runner/results/output.json


---

## Part C — Performance

Run:

python perf/load_test.py


Metrics:
- TTFT
- Latency
- TPOT

Output:
perf/metrics.csv


---

## Part D — Guardrails

Run:

python guardrails/validate.py


Features:
- Deterministic mode
- Output validation (A/B/C/D)
- Normalization

---

## Part E — Improvement

Baseline:

python improve/infer.py --mode baseline


Improved:

python improve/infer.py --mode improved

Techniques:
- Few-shot prompting
- Structured prompts
- Self-consistency (k=3)
- Majority voting

---

## Reproducibility

- Deterministic decoding used in evaluation
- Caching enabled
- Fixed seeds where applicable

---

## Trade-offs

| Aspect   | Baseline | Improved |
|----------|----------|----------|
| Accuracy | Lower    | Higher   |
| Latency  | Low      | Higher   |

---

## Limitations

- CPU-only setup
- Approximate scoring
- Small evaluation subset

---

## Run Everything
 -- make serve
 --python eval_runner/run_eval.py
 --python perf/load_test.py
 --python guardrails/validate.py
 --python improve/infer.py --mode improved


---

## Final Note

Focus is on correctness, reproducibility, and system design clarity over raw performance.