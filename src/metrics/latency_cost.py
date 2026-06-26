# src/metrics/latency_cost.py

from src.metrics.base_metric import BaseMetric, MetricResult
from src.data.dataset_loader import EvalSample
from src.models.base_client import LLMResponse


class LatencyMetric(BaseMetric):
    """
    Scores response latency. Lower is better.
    
    We normalize against a target latency so the score
    is always 0.0-1.0 regardless of actual milliseconds.
    
    Score of 1.0 = at or below target
    Score of 0.0 = 3x slower than target (or worse)
    """

    name = "latency"
    threshold = 0.5

    def __init__(self, target_ms: int = 1000):
        """
        target_ms: the latency we consider "good" (score=1.0)
        Anything faster scores 1.0, anything slower scores lower.
        """
        self.target_ms = target_ms

    def compute(self, sample: EvalSample, response: LLMResponse) -> MetricResult:
        latency = response.latency_ms

        # linear scale: target=1.0, 3×target=0.0
        score = max(0.0, 1.0 - (latency - self.target_ms) / (2 * self.target_ms))
        score = min(1.0, score)   # cap at 1.0 for faster-than-target

        return MetricResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=score >= self.threshold,
            reasoning=f"{latency}ms vs target {self.target_ms}ms → score {score:.3f}",
            raw_value={
                "latency_ms": latency,
                "target_ms": self.target_ms,
            }
        )


class CostEfficiencyMetric(BaseMetric):
    """
    Scores cost per response. Lower cost = higher score.
    Normalized against a target cost threshold.
    """

    name = "cost_efficiency"
    threshold = 0.5

    def __init__(self, target_cost_usd: float = 0.001):
        self.target_cost_usd = target_cost_usd

    def compute(self, sample: EvalSample, response: LLMResponse) -> MetricResult:
        cost = response.cost_usd

        if cost == 0:
            score = 1.0
        else:
            score = max(0.0, 1.0 - (cost / self.target_cost_usd))
            score = min(1.0, score)

        return MetricResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=score >= self.threshold,
            reasoning=f"${cost:.6f} vs target ${self.target_cost_usd:.6f}",
            raw_value={
                "cost_usd": cost,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            }
        )