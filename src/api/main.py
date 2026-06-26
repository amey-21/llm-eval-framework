from fastapi import FastAPI, HTTPException
from loguru import logger

from src.api.schemas import (
    RunSummary,
    LeaderboardEntry,
    ModelResult,
    HealthResponse,
)
from src.db.repository import EvalRepository

app = FastAPI(
    title="LLM Eval Framework API",
    description="Query evaluation results across models and metrics",
    version="0.1.0",
)

repo = EvalRepository()


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Is the server alive and connected to the database?
    
    Why does a health endpoint matter?
    CI pipelines, deployment scripts, and load balancers
    all need a way to verify the service is up before
    routing traffic to it. A /health endpoint is standard
    in every production service.
    """
    try:
        runs = repo.get_all_runs()
        return HealthResponse(
            status="ok",
            database="connected",
            total_runs=len(runs),
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.get("/runs", response_model=list[RunSummary])
def list_runs():
    """List all evaluation runs, newest first."""
    try:
        return repo.get_all_runs()
    except Exception as e:
        logger.error(f"Failed to list runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/runs/latest", response_model=dict)
def get_latest_run():
    """
    Convenience endpoint — returns the most recent run_id.
    The dashboard calls this on startup to know which run to display.
    """
    run_id = repo.get_latest_run_id()
    if run_id is None:
        raise HTTPException(status_code=404, detail="No runs found")
    return {"run_id": run_id}


@app.get("/runs/{run_id}/leaderboard", response_model=list[LeaderboardEntry])
def get_leaderboard(run_id: int):
    """
    Returns average score per model per metric for a run.
    This is the primary data source for the dashboard.
    """
    leaderboard = repo.get_leaderboard(run_id)
    if not leaderboard:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for run_id={run_id}"
        )
    return leaderboard


@app.get("/runs/{run_id}/results", response_model=list[ModelResult])
def get_results(run_id: int, model_name: str | None = None):
    """
    Returns raw model responses with metric scores attached.
    Optional ?model_name= filter to see one model's results.
    """
    results = repo.get_run_results(run_id)
    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for run_id={run_id}"
        )
    if model_name:
        results = [r for r in results if r["model_name"] == model_name]
    return results