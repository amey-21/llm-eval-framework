# src/metrics/base_metric.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from src.data.dataset_loader import EvalSample
from src.models.base_client import LLMResponse


@dataclass
class MetricResult:
    """
    Standard output for every metric.
    
    score is always 0.0 to 1.0:
        1.0 = perfect
        0.0 = complete failure
    
    This lets the dashboard and leaderboard treat all
    metrics uniformly without special-casing any of them.
    """
    metric_name: str
    score: float
    passed: bool
    reasoning: str
    raw_value: dict = field(default_factory=dict)

    def __post_init__(self):
        # enforce the 0.0-1.0 contract at construction time
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"Score must be 0.0-1.0, got {self.score} "
                f"for metric {self.metric_name}"
            )


class BaseMetric(ABC):
    """
    Abstract base class for all metrics.
    Same pattern as LLMClient — enforces a contract.
    
    Every metric must implement compute().
    The MetricsEngine only ever calls metric.compute().
    """

    # subclasses set this — used for logging and DB storage
    name: str = "base_metric"

    # score below this = failed
    threshold: float = 0.5

    @abstractmethod
    def compute(
        self,
        sample: EvalSample,
        response: LLMResponse,
    ) -> MetricResult:
        pass