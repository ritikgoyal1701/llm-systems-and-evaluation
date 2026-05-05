"""
vLLM OpenAI-server integration for lm-evaluation-harness.

Multiple-choice tasks (MMLU, HellaSwag, etc.) require log P(continuation | context).
That needs the **completions** API with echo + logprobs — not chat.completions.
See: lm_eval.models.openai_completions.LocalCompletionsAPI
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import handle_stop_sequences
from openai import OpenAI

eval_logger = logging.getLogger(__name__)


def _logprobs_as_dict(logprobs: Any) -> dict[str, Any] | None:
    if logprobs is None:
        return None
    if hasattr(logprobs, "model_dump"):
        return logprobs.model_dump()
    if isinstance(logprobs, dict):
        return logprobs
    return None


def _continuation_score_tokens(
    logprobs: dict[str, Any],
    ctx_token_len: int,
    *,
    drop_last_generated: bool,
) -> tuple[float, bool]:
    """
    Sum log-probs over continuation tokens (same indexing idea as
    ``LocalCompletionsAPI.parse_logprobs``: ``token_logprobs[ctxlen:-1]``).

    ``ctx_token_len`` must count prompt tokens the same way the vLLM server
    tokenizes ``context`` (HuggingFace tokenizer for the same model id).
    """
    tokens = logprobs.get("tokens") or []
    tlp = logprobs.get("token_logprobs") or []
    top_lp = logprobs.get("top_logprobs") or []

    if not tokens or not tlp or ctx_token_len < 0:
        return float("-inf"), False

    idx = min(ctx_token_len, len(tlp))
    end = len(tlp) - 1 if drop_last_generated and len(tlp) > idx + 1 else len(tlp)
    slice_lp = tlp[idx:end]

    total = 0.0
    counted = 0
    for x in slice_lp:
        if x is None:
            continue
        total += float(x)
        counted += 1

    if counted == 0:
        return float("-inf"), False

    is_greedy = True
    for i in range(idx, min(end, len(tokens), len(top_lp))):
        tok = tokens[i]
        top = top_lp[i]
        if not top:
            continue
        best = max(top, key=top.get)
        if best != tok:
            is_greedy = False
            break

    return total, is_greedy


def _context_token_len_hf(context: str, continuation: str, tok: Any) -> int:
    """
    Match lm_eval ``TemplateLM._encode_pair`` (causal): move trailing spaces from
    context onto continuation so BPE boundaries line up with the harness.
    """
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]

    whole: list[int] = tok(context + continuation, add_special_tokens=False)["input_ids"]
    ctx_ids: list[int] = tok(context, add_special_tokens=False)["input_ids"]

    if len(ctx_ids) > len(whole):
        return 0
    if whole[: len(ctx_ids)] != ctx_ids:
        eval_logger.warning(
            "Tokenizer/context mismatch when splitting logprobs "
            "(HF tokenization of context is not a prefix of context+continuation). "
            "Trying longest shared prefix length."
        )
        lo = 0
        for i in range(min(len(ctx_ids), len(whole))):
            if whole[i] != ctx_ids[i]:
                break
            lo = i + 1
        return lo

    return len(ctx_ids)


@register_model("vllm_local")
class VLLMModel(LM):
    def __init__(
        self,
        batch_size: int = 1,
        device: str = "cpu",
        pretrained: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        top_logprobs: int = 5,
        seed: int = 0,
        max_batch_size: int | None = None,
        tokenizer: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _ = max_batch_size
        kwargs.pop("trust_remote_code", None)
        tokenizer = tokenizer or kwargs.pop("tokenizer", None)
        if kwargs:
            eval_logger.warning("Ignoring unused vllm_local kwargs: %s", sorted(kwargs.keys()))

        self._batch_size = int(batch_size)
        self._device = device
        self.model = model or pretrained or os.environ.get(
            "VLLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        self._tokenizer_name = tokenizer or os.environ.get("VLLM_TOKENIZER", self.model)
        self._tok = None  # lazy AutoTokenizer — must align with server's model weights

        base = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "dummy")
        self.client = OpenAI(base_url=base, api_key=key)
        self.top_logprobs = int(top_logprobs)
        self.seed = int(seed)

        self._ll_cache: dict[str, tuple[float, bool]] = {}
        self._gen_cache: dict[str, str] = {}

    @property
    def max_length(self) -> int:
        return 2048

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> str:
        return self._device

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    def _hf_tokenizer(self) -> Any:
        if self._tok is None:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(
                self._tokenizer_name,
                trust_remote_code=True,
            )
        return self._tok

    def _hash_key(self, parts: list[Any]) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _completion_loglikelihood(self, context: str, continuation: str) -> tuple[float, bool]:
        prompt = context + continuation
        # Bump version when scoring logic changes so old in-process caches are not reused.
        cache_key = self._hash_key(["loglikelihood", 2, self.model, self._tokenizer_name, prompt])
        if cache_key in self._ll_cache:
            return self._ll_cache[cache_key]

        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            echo=True,
            max_tokens=1,
            temperature=0,
            logprobs=self.top_logprobs,
            seed=self.seed,
        )
        choice = response.choices[0]
        lp = _logprobs_as_dict(choice.logprobs)
        if not lp:
            out = (float("-inf"), False)
            self._ll_cache[cache_key] = out
            return out

        ctx_toks = _context_token_len_hf(context, continuation, self._hf_tokenizer())
        out = _continuation_score_tokens(lp, ctx_toks, drop_last_generated=True)
        self._ll_cache[cache_key] = out
        return out

    def loglikelihood(self, requests):
        return [self._completion_loglikelihood(*req.args) for req in requests]

    def loglikelihood_rolling(self, requests):
        """
        Best-effort string log-prob: one echo pass over the full prompt (not full
        rolling-window PPL). Sufficient only for short texts / smoke tests.
        """
        results: list[float] = []
        for req in requests:
            text = req.args[0]
            key = self._hash_key(["loglikelihood_rolling", self.model, text])
            if key in self._ll_cache:
                results.append(self._ll_cache[key][0])
                continue

            response = self.client.completions.create(
                model=self.model,
                prompt=text,
                echo=True,
                max_tokens=1,
                temperature=0,
                logprobs=self.top_logprobs,
                seed=self.seed,
            )
            choice = response.choices[0]
            lp = _logprobs_as_dict(choice.logprobs)
            if not lp or not lp.get("token_logprobs"):
                results.append(float("-inf"))
                continue
            tlp = lp["token_logprobs"]
            # Skip leading null (first prompt token) and trailing sampled token.
            s = 0.0
            for x in tlp[1:-1]:
                if x is not None:
                    s += float(x)
            self._ll_cache[key] = (s, True)
            results.append(s)
        return results

    def generate_until(self, requests):
        out: list[str] = []
        for req in requests:
            context, gen_kwargs = req.args
            g = dict(gen_kwargs)
            g.pop("do_sample", None)
            max_tokens = g.pop("max_tokens", g.pop("max_gen_toks", 256))
            temperature = float(g.pop("temperature", 0.0))
            until = g.pop("until", None)
            stop = handle_stop_sequences(until, eos=None) or None

            cache_key = self._hash_key(
                ["generate_until", self.model, context, max_tokens, temperature, stop]
            )
            if cache_key in self._gen_cache:
                out.append(self._gen_cache[cache_key])
                continue

            response = self.client.completions.create(
                model=self.model,
                prompt=context,
                max_tokens=int(max_tokens),
                temperature=temperature,
                stop=stop,
                seed=self.seed,
            )
            text = response.choices[0].text
            self._gen_cache[cache_key] = text
            out.append(text)
        return out
