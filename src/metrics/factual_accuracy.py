# src/metrics/factual_accuracy.py

import re
from src.metrics.base_metric import BaseMetric, MetricResult
from src.data.dataset_loader import EvalSample
from src.models.base_client import LLMResponse


class FactualAccuracyMetric(BaseMetric):
    """
    Measures whether the model got the right answer.

    For MMLU (multiple choice): extracts A/B/C/D and compares.
    For TruthfulQA (open-ended): checks if expected answer
    appears in the response (token overlap).

    Why two strategies?
    Multiple choice has a single correct letter — exact matching works.
    Open-ended answers can be phrased many ways — we need fuzzy matching.
    """

    name = "factual_accuracy"
    threshold = 0.5

    def compute(self, sample: EvalSample, response: LLMResponse) -> MetricResult:
        if sample.dataset == "mmlu":
            return self._score_multiple_choice(sample, response)
        else:
            return self._score_open_ended(sample, response)

    def _score_multiple_choice(
        self, sample: EvalSample, response: LLMResponse
    ) -> MetricResult:
        """Extract letter answer and compare to expected."""
        extracted = self._extract_letter(response.content)
        expected = sample.expected_answer.upper().strip()

        if extracted is None:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                reasoning=f"Could not extract A/B/C/D from response",
                raw_value={
                    "expected": expected,
                    "extracted": None,
                    "raw_response": response.content[:200],
                }
            )

        correct = extracted == expected
        return MetricResult(
            metric_name=self.name,
            score=1.0 if correct else 0.0,
            passed=correct,
            reasoning=(
                f"Expected {expected}, extracted {extracted} → "
                f"{'CORRECT' if correct else 'WRONG'}"
            ),
            raw_value={
                "expected": expected,
                "extracted": extracted,
                "raw_response": response.content[:200],
            }
        )

    def _score_open_ended(
        self, sample: EvalSample, response: LLMResponse
    ) -> MetricResult:
        """
        Scores open-ended answers using token overlap.
        
        Two strategies based on expected answer length:
        
        SHORT expected answer (≤3 tokens, e.g. "Paris", "12", "Au"):
            Use RECALL only — did the response contain the answer?
            Precision doesn't matter — the model should explain itself.
        
        LONG expected answer (>3 tokens):
            Use F1 — balance between containing the answer and
            not hallucinating irrelevant content.
        """
        # normalize both strings
        def normalize(text: str) -> set[str]:
            stopwords = {"the", "a", "an", "is", "it", "of",
                        "and", "or", "to", "for", "in", "on"}
            tokens = set(text.lower().split())
            # strip punctuation from each token
            import string
            tokens = {t.strip(string.punctuation) for t in tokens}
            tokens -= stopwords
            tokens.discard("")    # remove empty strings after stripping
            return tokens

        response_tokens = normalize(response.content)
        expected_tokens = normalize(sample.expected_answer)

        if not expected_tokens:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                reasoning="Expected answer is empty after normalization",
                raw_value={}
            )

        overlap = response_tokens & expected_tokens

        # SHORT answer strategy: recall only
        # "did the model say the right thing?" not "was everything it said right?"
        if len(expected_tokens) <= 3:
            recall = len(overlap) / len(expected_tokens)
            score = recall
            reasoning = (
                f"Short answer — recall only: {recall:.3f} | "
                f"expected={expected_tokens} | "
                f"overlap={overlap}"
            )
            return MetricResult(
                metric_name=self.name,
                score=round(score, 4),
                passed=score >= self.threshold,
                reasoning=reasoning,
                raw_value={
                    "strategy": "recall_only",
                    "recall": recall,
                    "expected_tokens": list(expected_tokens),
                    "overlap_tokens": list(overlap),
                }
            )

        # LONG answer strategy: F1
        precision = len(overlap) / len(response_tokens) if response_tokens else 0
        recall = len(overlap) / len(expected_tokens)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return MetricResult(
            metric_name=self.name,
            score=round(f1, 4),
            passed=f1 >= self.threshold,
            reasoning=(
                f"Long answer — F1={f1:.3f} | "
                f"precision={precision:.3f} recall={recall:.3f} | "
                f"overlap={len(overlap)} tokens"
            ),
            raw_value={
                "strategy": "f1",
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "overlap_tokens": list(overlap),
            }
        )

    def _extract_letter(self, response: str) -> str | None:
        """
        Extract the answer letter from a model response.
        Tries multiple patterns in priority order.
        
        Priority order matters — we go from most specific
        to least specific to avoid false matches.
        """
        response = response.strip()

        # 1. entire response is just one letter
        if response.upper() in ("A", "B", "C", "D"):
            return response.upper()

        # 2. starts with letter + punctuation: "B." or "B)" or "B:"
        match = re.match(r'^([ABCD])[.):\s]', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # 3. explicit answer pattern: "answer is B" or "answer: B"
        match = re.search(
            r'(?:answer|correct)[^\w]*(?:is|:)[^\w]*([ABCD])',
            response, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()

        # 4. "therefore B" or "thus B" — conclusion language
        match = re.search(
            r'(?:therefore|thus|so)[,\s]+([ABCD])\b',
            response, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()

        # 5. last resort: find last standalone A/B/C/D in the response
        # "last" because models often conclude with the answer
        matches = re.findall(r'\b([ABCD])\b', response, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

        return None