import json

# IMPORTANT: ensures model gets registered
import vllm_model

from lm_eval import evaluator


def main():
    results = evaluator.simple_evaluate(
        model="vllm_local",
        tasks=["hellaswag", "mmlu_abstract_algebra"],
        num_fewshot=0,
        batch_size=1,
        limit=5,
    )

    print("\n===== RESULTS =====")
    print(json.dumps(results["results"], indent=2))

    # Safe JSON serialization
    def safe(obj):
        try:
            json.dumps(obj)
            return obj
        except:
            return str(obj)

    safe_results = json.loads(json.dumps(results, default=safe))

    with open("eval_runner/results/output.json", "w") as f:
        json.dump(safe_results, f, indent=2)

    print("\nSaved to eval_runner/results/output.json")


if __name__ == "__main__":
    main()