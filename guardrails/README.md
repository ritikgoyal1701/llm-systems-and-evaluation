# Guardrails

Lightweight checks around a local OpenAI-compatible server (vLLM-style) for **deterministic generation** and **output validation**.

## What `validate.py` does

1. **Deterministic mode** — Each completion uses `temperature=0`, `top_p=1`, and a fixed `max_tokens` cap. The script does **not** set a server-side seed; if your stack supports `seed`, add it in `DETERMINISTIC` in `validate.py` for stronger guarantees.

2. **Determinism check** — The same user prompt is sent three times in a row. The test passes only if all three decoded strings are **byte-for-byte identical** (after `.strip()` on the assistant message).

3. **Validation** — For a multiple-choice style prompt, the raw string is normalized (second whitespace-separated token, punctuation stripped) and checked against `^[A-D]$`. `safe_output()` returns the normalized letter if valid, otherwise a fixed fallback (`"A"`).

Run against a live server (defaults in script: `MODEL`, `base_url=http://localhost:8000/v1`):

```bash
python guardrails/validate.py
```

A successful run prints a short summary, for example:

```text
{'deterministic': True, 'validation_passed': True}
```

## What we tested

| Area | Method |
|------|--------|
| Identical prompts → identical outputs | Three repeated generations on one fixed prompt |
| Structured MCQ answer | Regex on normalized single letter A–D |
| Safe surface | Reject or coerce invalid text via `safe_output()` |

## Where nondeterminism can still persist

Even with `temperature=0` and `top_p=1`, you may see **non-identical** runs across processes, restarts, or upgrades because of:

- **No explicit seed** — Many APIs expose `seed`; without it, some backends still vary (e.g. parallel kernels, nondeterministic reductions).
- **Batched / concurrent traffic** — Scheduling and batch composition can affect numeric paths on GPU.
- **Model or server updates** — Any change to weights, kernels, or decoding fixes can change the greedy path.
- **Context length and truncation** — Different effective context or truncation rules change the argmax state.
- **Hardware / driver** — Mixed precision and non-associative floating-point can introduce rare drift unless the stack is fully deterministic end-to-end.

Treat **determinism as an empirical property**: re-run this script after infra changes. If you need hard guarantees, wire in an API `seed` (when supported), pin model and server versions, and document any known nondeterministic features of your inference engine.
