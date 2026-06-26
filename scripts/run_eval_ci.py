# scripts/run_eval_ci.py
"""
Lightweight eval runner for CI.
- Uses 20 samples (not 50+)
- No LLM judge (saves cost)
- Writes results to results/ci_run.json
- No PostgreSQL dependency (CI doesn't have a DB)
"""

import json
import asyncio
from pathlib import Path
from loguru import logger

from src.data.dataset_loader import DatasetLoader
from src.models.openai_client import OpenAIClient
from src.models.groq_client import GroqClient
from src.metrics.factual_accuracy import FactualAccuracyMetric
from src.metrics.latency_cost import LatencyMetric, CostEfficiencyMetric
from src.metrics.instruction_following import InstructionFollowingMetric
from src.models.base_client import LLMClient
from src.data.dataset_loader import EvalSample


async def evaluate_sample(client: LLMClient, sample: EvalSample) -> dict:
    """Run one sample against one model, return metric scores."""
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None, lambda: client.generate(sample.question)
        )
    except Exception as e:
        logger.error(f"API call failed: {client.model_name} / {sample.id}: {e}")
        return None

    metrics = [
        FactualAccuracyMetric(),
        LatencyMetric(target_ms=1000),
        CostEfficiencyMetric(target_cost_usd=0.001),
        InstructionFollowingMetric(),
    ]

    scores = {}
    for metric in metrics:
        result = metric.compute(sample, response)
        scores[result.metric_name] = result.score

    return scores


async def main():
    clients = [
        OpenAIClient("gpt-4o-mini"),
        GroqClient("llama-3.1-8b-instant"),
        GroqClient("llama-3.3-70b-versatile"),
        GroqClient("openai/gpt-oss-120b"),
    ]

    loader = DatasetLoader()
    samples = loader.load("mmlu", sample_size=50, seed=42)
    logger.info(f"CI eval: {len(samples)} samples × {len(clients)} models")

    # results[model_name][metric_name] = list of scores
    raw: dict[str, dict[str, list[float]]] = {
        c.model_name: {} for c in clients
    }

    for i, sample in enumerate(samples):
        logger.info(f"Sample {i+1}/{len(samples)}")
        tasks = [evaluate_sample(client, sample) for client in clients]
        results = await asyncio.gather(*tasks)

        for client, scores in zip(clients, results):
            if scores is None:
                continue
            for metric_name, score in scores.items():
                if metric_name not in raw[client.model_name]:
                    raw[client.model_name][metric_name] = []
                raw[client.model_name][metric_name].append(score)

    # average scores per model per metric
    averaged = {}
    for model_name, metrics in raw.items():
        averaged[model_name] = {
            metric: round(sum(scores) / len(scores), 4)
            for metric, scores in metrics.items()
            if scores
        }

    output_path = Path("results/ci_run.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(averaged, indent=2))

    total_cost = sum(
        score * 0.001
        for model_metrics in averaged.values()
        for score in model_metrics.values()
    )
    logger.info(f"CI eval complete. Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())