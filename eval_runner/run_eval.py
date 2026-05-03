import vllm_model
from lm_eval import evaluator

def main():
    results = evaluator.simple_evaluate(
        model="vllm_local",
        model_args="",
        tasks=[
            "hellaswag",
            "mmlu_abstract_algebra"
        ],
        num_fewshot=0,
        batch_size=1,
        limit=5,
    )

    print("\n=== RESULTS ===")
    print(results)

    import json, os
    os.makedirs("eval_runner/results", exist_ok=True)

    with open("eval_runner/results/output.json", "w") as f:
        def safe_serialize(obj):
            try:
                json.dumps(obj)
                return obj
            except:
                return str(obj)

        safe_results = json.loads(
            json.dumps(results, default=safe_serialize)
        )

        with open("eval_runner/results/output.json", "w") as f:
            json.dump(safe_results, f, indent=2)


if __name__ == "__main__":
    main()