# src/evaluator.py

import asyncio
from dataclasses import dataclass
from typing import Optional
from loguru import logger

from src.models.base_client import LLMClient, LLMResponse
from src.data.dataset_loader import EvalSample
from src.db.repository import EvalRepository

from src.metrics.factual_accuracy import FactualAccuracyMetric
from src.metrics.latency_cost import LatencyMetric, CostEfficiencyMetric
from src.metrics.instruction_following import InstructionFollowingMetric

@dataclass
class EvalResult:
    """
    The output of evaluating one sample against one model.
    Pairs the original sample with the model's response.
    The metrics engine will receive this and compute scores.
    """
    sample: EvalSample
    response: LLMResponse
    model_result_id: int          # DB id for attaching metric scores later
    error: Optional[str] = None   # if the API call failed


class Evaluator:
    """
    Orchestrates evaluation runs.
    
    Responsibilities:
    1. Create a run record in the DB
    2. For each sample, call all models concurrently
    3. Save every response to the DB immediately
    4. Return structured EvalResult objects for the metrics engine
    
    What it does NOT do:
    - Load datasets (DatasetLoader does that)
    - Compute metrics (MetricsEngine will do that)
    - Display results (Dashboard does that)
    
    Each class has one job.
    """

    def __init__(
        self,
        clients: list[LLMClient],
        run_name: str,
        use_llm_judge: bool = False,   # off by default
    ):
        self.clients = clients
        self.run_name = run_name
        self.repo = EvalRepository()

        self.metrics = [
            FactualAccuracyMetric(),
            LatencyMetric(target_ms=1000),
            CostEfficiencyMetric(target_cost_usd=0.001),
            InstructionFollowingMetric(),
        ]

        if use_llm_judge:
            from src.metrics.llm_judge import LLMJudgeMetric
            self.metrics.append(LLMJudgeMetric())
            logger.info("LLM judge enabled — additional API costs apply")

    def run(self, samples: list[EvalSample]) -> dict[str, list[EvalResult]]:
        """
        Main entry point. Synchronous wrapper around async logic.
        
        Returns a dict mapping model_name → list of EvalResult.
        
        Why wrap async in sync?
        Most calling code (scripts, tests) is synchronous.
        asyncio.run() lets us use async internally without
        forcing async all the way up the call stack.
        """
        return asyncio.run(self._run_async(samples))

    async def _run_async(
        self, samples: list[EvalSample]
    ) -> dict[str, list[EvalResult]]:
        """
        Core async evaluation logic.
        
        For each sample, we fire one API call per model simultaneously.
        asyncio.gather() waits for ALL calls to complete before moving
        to the next sample.
        """
        # create the run record — everything references this id
        config = {
            "models": [c.model_name for c in self.clients],
            "num_samples": len(samples),
        }
        run_id = self.repo.create_run(self.run_name, config)
        logger.info(
            f"Starting eval run '{self.run_name}' | "
            f"{len(samples)} samples × {len(self.clients)} models = "
            f"{len(samples) * len(self.clients)} total API calls"
        )

        # results[model_name] = list of EvalResult
        results: dict[str, list[EvalResult]] = {
            c.model_name: [] for c in self.clients
        }

        for i, sample in enumerate(samples):
            logger.info(f"Sample {i+1}/{len(samples)}: {sample.id}")

            # fire all model calls for this sample concurrently
            tasks = [
                self._evaluate_single(client, sample, run_id)
                for client in self.clients
            ]
            sample_results = await asyncio.gather(*tasks)

            # collect results per model
            for result in sample_results:
                results[result.response.model_name].append(result)

        # update total cost on the run record
        total_cost = sum(
            result.response.cost_usd
            for model_results in results.values()
            for result in model_results
            if result.error is None
        )
        self.repo.update_run_cost(run_id, total_cost)

        logger.info(
            f"Run complete | "
            f"Total cost: ${total_cost:.4f} | "
            f"run_id: {run_id}"
        )

        return results

    async def _evaluate_single(
        self,
        client: LLMClient,
        sample: EvalSample,
        run_id: int,
    ) -> EvalResult:
        """
        Call one model on one sample. Save result to DB.
        
        Why catch exceptions here instead of letting them propagate?
        If Groq's API is down, we don't want to lose all OpenAI results.
        We record the error and continue. Partial results are better
        than no results.
        """
        try:
            # run the blocking API call in a thread pool
            # so it doesn't block the event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,                           # use default thread pool
                lambda: client.generate(sample.question)
            )

            # persist to DB immediately — don't wait until the end
            model_result_id = self.repo.save_model_result(
                run_id=run_id,
                model_name=response.model_name,
                dataset=sample.dataset,
                prompt=sample.question,
                response=response.content,
                latency_ms=response.latency_ms,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=response.cost_usd,
            )

        except Exception as e:
            logger.error(
                f"Failed: {client.model_name} on {sample.id}: {e}"
            )
            # return an error result so the run continues
            return EvalResult(
                sample=sample,
                response=LLMResponse(
                    content="",
                    model_name=client.model_name,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=0,
                    cost_usd=0.0,
                ),
                model_result_id=-1,
                error=str(e),
            )
        
        
        # compute and store metrics
        for metric in self.metrics:
            try:
                result = metric.compute(sample, response)
                self.repo.save_metric_score(
                    model_result_id=model_result_id,
                    metric_name=result.metric_name,
                    score=result.score,
                    raw_value=result.raw_value,
                )
            except Exception as e:
                logger.warning(f"Metric {metric.name} failed: {e}")

        return EvalResult(
            sample=sample,
            response=response,
            model_result_id=model_result_id,
        )