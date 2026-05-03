from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from openai import OpenAI
import hashlib

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DETERMINISTIC = dict(
    temperature=0,
    top_p=1,
    max_tokens=2,
)


@register_model("vllm_local")
class VLLMModel(LM):
    def __init__(self, batch_size=1, device="cpu", **kwargs):
        super().__init__()

        self._batch_size = batch_size
        self._device = device

        self.client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="dummy"
        )

        self.cache = {}

    # ---------------------------
    # REQUIRED PROPERTIES
    # ---------------------------
    @property
    def max_length(self):
        return 1024

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    @property
    def tokenizer_name(self):
        return "dummy"

    # ---------------------------
    # CACHING
    # ---------------------------
    def _cache_key(self, prompt):
        return hashlib.md5(prompt.encode()).hexdigest()

    def _cached_generate(self, prompt):
        prompt = prompt + "\nAnswer (ONLY A/B/C/D):"

        key = self._cache_key(prompt)
        if key in self.cache:
            return self.cache[key]

        print(f"[MODEL CALL] {prompt[:60]}")

        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            **DETERMINISTIC,
        )

        output = response.choices[0].message.content.strip()
        self.cache[key] = output
        return output

    # ---------------------------
    # GENERATION
    # ---------------------------
    def generate_until(self, requests):
        results = []

        for req in requests:
            prompt = req.args[0]
            output = self._cached_generate(prompt)
            results.append(output)

        return results

    # ---------------------------
    # LOG LIKELIHOOD (MCQ FIXED)
    # ---------------------------
    def loglikelihood(self, requests):
        results = []

        for req in requests:
            context, continuation = req.args

            # Generate prediction (A/B/C/D)
            output = self._cached_generate(context)

            def extract(text):
                if not text:
                    return ""
                token = text.strip().split()[0]
                return token.replace(".", "").strip()

            pred = extract(output)
            gold = extract(continuation)

            score = float(pred == gold)

            results.append((score, True))

        return results

    # ---------------------------
    # REQUIRED (dummy impl)
    # ---------------------------
    def loglikelihood_rolling(self, requests):
        results = []

        for req in requests:
            text = req.args[0]
            score = -len(text)
            results.append(score)

        return results