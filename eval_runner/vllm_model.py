from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from openai import OpenAI
import hashlib

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DETERMINISTIC = dict(
    temperature=0,
    top_p=1,
    max_tokens=2,  # keep small for CPU
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
    def _cache_key(self, prompt, stop):
        return hashlib.md5((prompt + str(stop)).encode()).hexdigest()

    def _cached_generate(self, prompt, stop):
        key = self._cache_key(prompt, stop)

        if key in self.cache:
            return self.cache[key]

        print(f"[MODEL CALL] Prompt snippet: {prompt[:50]}")

        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stop=stop,
            **DETERMINISTIC,
        )

        output = response.choices[0].message.content
        self.cache[key] = output

        return output

    # ---------------------------
    # GENERATION
    # ---------------------------
    def generate_until(self, requests):
        results = []

        for req in requests:
            prompt = req.args[0]
            stop = req.args[1]

            output = self._cached_generate(prompt, stop)
            results.append(output)

        return results

    # ---------------------------
    # LOG LIKELIHOOD (approx)
    # ---------------------------
    def loglikelihood(self, requests):
        results = []

        for req in requests:
            context, continuation = req.args

            prompt = (context + continuation)[:150]
            output = self._cached_generate(prompt, stop=None)

            def extract_choice(text):
                if not text:
                    return ""
                token = text.strip().split()[0]
                return token.replace(".", "").strip()


            pred = extract_choice(output)
            gold = extract_choice(continuation)

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
            score = -len(text)  # dummy
            results.append(score)

        return results