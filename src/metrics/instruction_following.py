# src/metrics/instruction_following.py

import re
from src.metrics.base_metric import BaseMetric, MetricResult
from src.data.dataset_loader import EvalSample
from src.models.base_client import LLMResponse


class InstructionFollowingMetric(BaseMetric):
    """
    Measures whether the model followed the format instructions.
    
    For MMLU: did it answer with just a letter (as instructed)?
    For HumanEval: did it return actual code?
    For TruthfulQA: did it give a substantive answer?
    
    This is a proxy for instruction following — a model that
    writes an essay when asked for one letter is less useful
    in production, even if technically correct.
    """

    name = "instruction_following"
    threshold = 0.5

    def compute(self, sample: EvalSample, response: LLMResponse) -> MetricResult:
        if sample.dataset == "mmlu":
            return self._check_mmlu_format(response)
        elif sample.dataset == "humaneval":
            return self._check_code_format(response)
        else:
            return self._check_substantive_answer(response)

    def _check_mmlu_format(self, response: LLMResponse) -> MetricResult:
        """
        Did the model answer concisely?
        A single letter = perfect score.
        A letter with brief explanation = partial credit.
        A long essay = low score.
        """
        content = response.content.strip()
        word_count = len(content.split())

        if content.upper() in ("A", "B", "C", "D"):
            score, reasoning = 1.0, "Perfect: single letter answer"
        elif word_count <= 5:
            score, reasoning = 0.8, f"Good: brief answer ({word_count} words)"
        elif word_count <= 20:
            score, reasoning = 0.6, f"Acceptable: moderate length ({word_count} words)"
        elif word_count <= 50:
            score, reasoning = 0.3, f"Poor: verbose answer ({word_count} words)"
        else:
            score, reasoning = 0.1, f"Very poor: essay-length answer ({word_count} words)"

        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=score >= self.threshold,
            reasoning=reasoning,
            raw_value={"word_count": word_count, "response_preview": content[:100]}
        )

    def _check_code_format(self, response: LLMResponse) -> MetricResult:
        """Did the model return actual Python code?"""
        content = response.content
        has_def = "def " in content
        has_return = "return " in content
        has_code_block = "```" in content

        score = sum([has_def, has_return]) * 0.4 + (0.2 if has_code_block else 0)
        score = min(1.0, score)

        return MetricResult(
            metric_name=self.name,
            score=round(score, 4),
            passed=score >= self.threshold,
            reasoning=f"has_def={has_def} has_return={has_return}",
            raw_value={"has_def": has_def, "has_return": has_return}
        )

    def _check_substantive_answer(self, response: LLMResponse) -> MetricResult:
        """Did the model give a real answer (not refuse or deflect)?"""
        content = response.content.lower()
        word_count = len(content.split())

        refusal_phrases = [
            "i cannot", "i can't", "i don't know",
            "i'm not sure", "as an ai", "i apologize"
        ]
        refused = any(phrase in content for phrase in refusal_phrases)

        if refused:
            score, reasoning = 0.0, "Model refused to answer"
        elif word_count < 5:
            score, reasoning = 0.3, f"Answer too short ({word_count} words)"
        else:
            score, reasoning = 1.0, f"Substantive answer ({word_count} words)"

        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=score >= self.threshold,
            reasoning=reasoning,
            raw_value={"word_count": word_count, "refused": refused}
        )