# scripts/run_eval.py

import argparse
from loguru import logger
from src.data.dataset_loader import DatasetLoader
from src.models.openai_client import OpenAIClient
from src.models.groq_client import GroqClient
from src.evaluator import Evaluator
from src.metrics.factual_accuracy import FactualAccuracyMetric

_accuracy_metric = FactualAccuracyMetric()


def main():
    parser = argparse.ArgumentParser(description="Run LLM evaluation")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--run-name", type=str, default="manual_run")
    parser.add_argument("--dataset", type=str, default="mmlu")
    parser.add_argument("--use-judge", action="store_true", default=False)
    args = parser.parse_args()

    clients = [
        OpenAIClient("gpt-4o-mini"),
        GroqClient("llama-3.1-8b-instant"),
        GroqClient("llama-3.3-70b-versatile"),
        GroqClient("openai/gpt-oss-120b"),
    ]

    loader = DatasetLoader()
    samples = loader.load(args.dataset, sample_size=args.sample_size)

    evaluator = Evaluator(
        clients=clients,
        run_name=args.run_name,
        use_llm_judge=args.use_judge,
    )
    results = evaluator.run(samples)

    print("\n" + "="*60)
    print(f"EVAL COMPLETE — {args.run_name}")
    print("="*60)

    for model_name, eval_results in results.items():
        print(f"\n{model_name}:")
        for r in eval_results:
            if r.error:
                print(f"  [ERROR] {r.sample.id} | {r.error}")
                continue

            if r.sample.dataset == "mmlu":
                extracted = _accuracy_metric._extract_letter(r.response.content)
                answer_display = extracted or f"FAIL({r.response.content[:20]!r})"
                correct = "✓" if extracted == r.sample.expected_answer else "✗"
            else:
                answer_display = r.response.content[:40]
                correct = ""

            print(
                f"  [{correct}] {r.sample.id} | "
                f"expected: {r.sample.expected_answer} | "
                f"got: {answer_display} | "
                f"{r.response.latency_ms}ms"
            )


if __name__ == "__main__":
    main()