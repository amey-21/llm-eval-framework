import json
import psycopg2
from contextlib import contextmanager
from loguru import logger
from dotenv import load_dotenv
import os

load_dotenv()


@contextmanager
def get_connection():
    """
    Context manager for database connections.
    
    Why a context manager?
    Database connections are expensive resources. If your code
    crashes between opening and closing a connection, it leaks.
    A context manager guarantees cleanup even if an exception
    occurs — the 'finally' block always runs.
    
    Usage:
        with get_connection() as conn:
            cursor = conn.cursor()
            ...
        # connection automatically closed here, even if error occurred
    """
    conn = psycopg2.connect(os.getenv("POSTGRES_URL"))
    try:
        yield conn
        conn.commit()      # save changes if everything worked
    except Exception:
        conn.rollback()    # undo changes if anything failed
        raise
    finally:
        conn.close()       # always close, no matter what


class EvalRepository:
    """
    All database operations for the eval framework.
    
    Why a repository class?
    It centralizes all SQL in one place. If you switch from
    PostgreSQL to SQLite tomorrow, you change this file only.
    Nothing else in the system knows SQL exists.
    
    This is the same decoupling principle as model clients.
    """

    def create_run(self, run_name: str, config: dict) -> int:
        """
        Create a new eval run record.
        Returns the run_id — everything else references this.
        """
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO eval_runs (run_name, config)
                VALUES (%s, %s)
                RETURNING id
                """,
                (run_name, json.dumps(config))
            )
            run_id = cursor.fetchone()[0]
            logger.info(f"Created eval run: id={run_id} name={run_name}")
            return run_id

    def save_model_result(
        self,
        run_id: int,
        model_name: str,
        dataset: str,
        prompt: str,
        response: str,
        latency_ms: int,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> int:
        """
        Save one model response to the database.
        Returns model_result_id for attaching metric scores.
        """
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO model_results
                    (run_id, model_name, dataset, prompt, response,
                     latency_ms, input_tokens, output_tokens, cost_usd)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    run_id, model_name, dataset, prompt, response,
                    latency_ms, input_tokens, output_tokens, cost_usd
                )
            )
            return cursor.fetchone()[0]

    def save_metric_score(
        self,
        model_result_id: int,
        metric_name: str,
        score: float,
        raw_value: dict,
    ) -> None:
        """Save one metric score for one model result."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO metric_scores
                    (model_result_id, metric_name, score, raw_value)
                VALUES (%s, %s, %s, %s)
                """,
                (model_result_id, metric_name, score, json.dumps(raw_value))
            )

    def get_leaderboard(self, run_id: int) -> list[dict]:
        """
        Returns average scores per model per metric for a run.
        This is the query that powers the dashboard.
        """
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    mr.model_name,
                    ms.metric_name,
                    ROUND(AVG(ms.score)::numeric, 4) as avg_score,
                    COUNT(*) as sample_count
                FROM model_results mr
                JOIN metric_scores ms ON ms.model_result_id = mr.id
                WHERE mr.run_id = %s
                GROUP BY mr.model_name, ms.metric_name
                ORDER BY mr.model_name, ms.metric_name
                """,
                (run_id,)
            )
            rows = cursor.fetchall()
            return [
                {
                    "model_name": row[0],
                    "metric_name": row[1],
                    "avg_score": float(row[2]),
                    "sample_count": row[3],
                }
                for row in rows
            ]

    def update_run_cost(self, run_id: int, total_cost: float) -> None:
        """Update the total cost after a run completes."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE eval_runs SET total_cost = %s WHERE id = %s",
                (total_cost, run_id)
            )

    def get_all_runs(self) -> list[dict]:
        """List all eval runs with summary stats."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    r.id,
                    r.run_name,
                    r.created_at,
                    r.total_cost,
                    COUNT(DISTINCT mr.model_name) as num_models,
                    COUNT(DISTINCT mr.id) / 
                        NULLIF(COUNT(DISTINCT mr.model_name), 0) as num_samples
                FROM eval_runs r
                LEFT JOIN model_results mr ON mr.run_id = r.id
                GROUP BY r.id, r.run_name, r.created_at, r.total_cost
                ORDER BY r.created_at DESC
                """
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "run_name": row[1],
                    "created_at": row[2],
                    "total_cost": float(row[3]) if row[3] else None,
                    "num_models": row[4],
                    "num_samples": row[5] or 0,
                }
                for row in rows
            ]

    def get_run_results(self, run_id: int) -> list[dict]:
        """
        Get all model results for a run, with their metric scores
        collapsed into a single dict per result.
        
        Why collapse metrics into a dict?
        The DB stores one row per metric per result (normalized).
        The API consumer wants one object per result with all metrics
        attached. We do that transformation here, not in the API layer.
        """
        with get_connection() as conn:
            cursor = conn.cursor()

            # get all results with their metric scores
            cursor.execute(
                """
                SELECT
                    mr.id,
                    mr.model_name,
                    mr.dataset,
                    mr.prompt,
                    mr.response,
                    mr.latency_ms,
                    mr.input_tokens,
                    mr.output_tokens,
                    mr.cost_usd,
                    ms.metric_name,
                    ms.score
                FROM model_results mr
                LEFT JOIN metric_scores ms ON ms.model_result_id = mr.id
                WHERE mr.run_id = %s
                ORDER BY mr.id, ms.metric_name
                """,
                (run_id,)
            )
            rows = cursor.fetchall()

            # collapse multiple metric rows into one result dict
            results_map: dict[int, dict] = {}
            for row in rows:
                result_id = row[0]
                if result_id not in results_map:
                    results_map[result_id] = {
                        "id": row[0],
                        "model_name": row[1],
                        "dataset": row[2],
                        "prompt": row[3],
                        "response": row[4],
                        "latency_ms": row[5] or 0,
                        "input_tokens": row[6] or 0,
                        "output_tokens": row[7] or 0,
                        "cost_usd": float(row[8]) if row[8] else 0.0,
                        "metric_scores": {},
                    }
                if row[9]:  # metric_name
                    results_map[result_id]["metric_scores"][row[9]] = float(row[10])

            return list(results_map.values())

    def get_latest_run_id(self) -> int | None:
        """Returns the most recent run id."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM eval_runs ORDER BY created_at DESC LIMIT 1"
            )
            row = cursor.fetchone()
            return row[0] if row else None