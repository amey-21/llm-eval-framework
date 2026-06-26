# src/api/schemas.py

from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class RunSummary(BaseModel):
    """One row in the list of eval runs."""
    id: int
    run_name: str
    created_at: datetime
    total_cost: Optional[float]
    num_models: int
    num_samples: int


class LeaderboardEntry(BaseModel):
    """One model's score on one metric."""
    model_name: str
    metric_name: str
    avg_score: float
    sample_count: int


class ModelResult(BaseModel):
    """One raw model response with its metric scores."""
    id: int
    model_name: str
    dataset: str
    prompt: str
    response: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    metric_scores: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    database: str
    total_runs: int