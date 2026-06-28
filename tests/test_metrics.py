# tests/test_metrics.py

import pytest
from src.metrics.factual_accuracy import FactualAccuracyMetric
from src.metrics.latency_cost import LatencyMetric, CostEfficiencyMetric
from src.metrics.instruction_following import InstructionFollowingMetric
from src.metrics.base_metric import MetricResult
from src.data.dataset_loader import EvalSample
from src.models.base_client import LLMResponse


# ── shared fixtures ────────────────────────────────────────────────────────────

def make_response(content: str, latency_ms: int = 500, cost: float = 0.0005) -> LLMResponse:
    """Helper to build LLMResponse without hitting any API."""
    return LLMResponse(
        content=content,
        model_name="test-model",
        input_tokens=50,
        output_tokens=10,
        latency_ms=latency_ms,
        cost_usd=cost,
    )


def make_mmlu_sample(expected: str = "B") -> EvalSample:
    return EvalSample(
        id="test_mmlu_001",
        dataset="mmlu",
        question="What is the powerhouse of the cell?\nA. Nucleus\nB. Mitochondria\nC. Ribosome\nD. Golgi\nAnswer with just the letter.",
        expected_answer=expected,
        choices=["Nucleus", "Mitochondria", "Ribosome", "Golgi"],
        subject="biology",
    )


def make_truthfulqa_sample() -> EvalSample:
    return EvalSample(
        id="test_tqa_001",
        dataset="truthfulqa",
        question="What happens if you swallow a watermelon seed?",
        expected_answer="Nothing happens, it passes through your digestive system",
    )


# ── factual accuracy ───────────────────────────────────────────────────────────

class TestFactualAccuracy:
    metric = FactualAccuracyMetric()

    def test_exact_letter_correct(self):
        result = self.metric.compute(make_mmlu_sample("B"), make_response("B"))
        assert result.score == 1.0
        assert result.passed is True

    def test_exact_letter_wrong(self):
        result = self.metric.compute(make_mmlu_sample("B"), make_response("A"))
        assert result.score == 0.0
        assert result.passed is False

    def test_letter_with_period(self):
        """B. should extract as B — real model behavior."""
        result = self.metric.compute(make_mmlu_sample("B"), make_response("B."))
        assert result.score == 1.0

    def test_letter_with_explanation(self):
        """B. Mitochondria is... should still extract B."""
        result = self.metric.compute(
            make_mmlu_sample("B"),
            make_response("B. Mitochondria is the powerhouse of the cell.")
        )
        assert result.score == 1.0

    def test_verbose_correct_answer(self):
        """'The answer is B' should extract B."""
        result = self.metric.compute(
            make_mmlu_sample("B"),
            make_response("The answer is B because mitochondria produce ATP.")
        )
        assert result.score == 1.0

    def test_lowercase_letter(self):
        """'b' should normalize to 'B'."""
        result = self.metric.compute(make_mmlu_sample("B"), make_response("b"))
        assert result.score == 1.0

    def test_extraction_failure_returns_zero(self):
        """Response with no A/B/C/D should score 0, not crash."""
        result = self.metric.compute(
            make_mmlu_sample("B"),
            make_response("I cannot determine the answer from the given options.")
        )
        assert result.score == 0.0
        assert result.passed is False

    def test_open_ended_partial_overlap(self):
        """TruthfulQA: partial token overlap scores between 0 and 1."""
        result = self.metric.compute(
            make_truthfulqa_sample(),
            make_response("It passes through your digestive system harmlessly.")
        )
        assert 0.0 < result.score <= 1.0

    def test_open_ended_no_overlap(self):
        """TruthfulQA: completely wrong answer scores 0."""
        result = self.metric.compute(
            make_truthfulqa_sample(),
            make_response("A watermelon will grow inside your stomach.")
        )
        assert result.score < 0.3

    def test_result_score_always_normalized(self):
        """Score must always be 0.0-1.0 — the contract."""
        for response_text in ["A", "B", "C", "D", "I don't know", ""]:
            result = self.metric.compute(
                make_mmlu_sample("A"),
                make_response(response_text)
            )
            assert 0.0 <= result.score <= 1.0, (
                f"Score {result.score} out of range for response: {response_text!r}"
            )


# ── letter extraction ──────────────────────────────────────────────────────────

class TestLetterExtraction:
    """
    Tests the extraction logic directly.
    This is the most important unit in the metrics engine
    because it affects every MMLU score.
    """
    metric = FactualAccuracyMetric()

    @pytest.mark.parametrize("response,expected", [
        ("A", "A"),
        ("B", "B"),
        ("C", "C"),
        ("D", "D"),
        ("b", "B"),                                    # lowercase
        ("B.", "B"),                                   # trailing period
        ("B)", "B"),                                   # trailing paren
        ("B: mitochondria", "B"),                      # colon format
        ("The answer is B", "B"),                      # natural language
        ("answer: B", "B"),                            # answer: prefix
        ("therefore B", "B"),                          # conclusion language
        ("Based on analysis, B is correct", "B"),      # B appears standalone
    ])
    def test_extracts_correctly(self, response, expected):
        assert self.metric._extract_letter(response) == expected

    @pytest.mark.parametrize("response", [
        "",
        "I don't know",
        "None of the above",
        "The answer is unclear",
    ])
    def test_returns_none_when_no_letter(self, response):
        assert self.metric._extract_letter(response) is None


# ── latency metric ─────────────────────────────────────────────────────────────

class TestLatencyMetric:

    def test_faster_than_target_scores_one(self):
        metric = LatencyMetric(target_ms=1000)
        result = metric.compute(make_mmlu_sample(), make_response("B", latency_ms=200))
        assert result.score == 1.0

    def test_at_target_scores_one(self):
        metric = LatencyMetric(target_ms=1000)
        result = metric.compute(make_mmlu_sample(), make_response("B", latency_ms=1000))
        assert result.score == 1.0

    def test_slower_than_target_scores_lower(self):
        metric = LatencyMetric(target_ms=1000)
        result = metric.compute(make_mmlu_sample(), make_response("B", latency_ms=2000))
        assert result.score < 1.0
        assert result.score >= 0.0

    def test_very_slow_scores_zero(self):
        metric = LatencyMetric(target_ms=1000)
        result = metric.compute(make_mmlu_sample(), make_response("B", latency_ms=9999))
        assert result.score == 0.0

    def test_score_always_in_range(self):
        metric = LatencyMetric(target_ms=1000)
        for latency in [0, 100, 1000, 5000, 99999]:
            result = metric.compute(make_mmlu_sample(), make_response("B", latency_ms=latency))
            assert 0.0 <= result.score <= 1.0


# ── instruction following ──────────────────────────────────────────────────────

class TestInstructionFollowing:
    metric = InstructionFollowingMetric()

    def test_single_letter_perfect_score(self):
        result = self.metric.compute(make_mmlu_sample(), make_response("B"))
        assert result.score == 1.0

    def test_long_essay_low_score(self):
        essay = " ".join(["word"] * 100)
        result = self.metric.compute(make_mmlu_sample(), make_response(essay))
        assert result.score <= 0.3

    def test_refusal_scores_zero_for_truthfulqa(self):
        result = self.metric.compute(
            make_truthfulqa_sample(),
            make_response("I cannot answer this question as an AI.")
        )
        assert result.score == 0.0
        assert result.passed is False


# ── base metric contract ───────────────────────────────────────────────────────

class TestMetricContract:
    """
    Every metric must honor the 0.0-1.0 score contract.
    This test would catch a metric that returns raw latency (500)
    instead of normalized score (0.9).
    """

    def test_metric_result_rejects_out_of_range_score(self):
        with pytest.raises(ValueError, match="Score must be 0.0-1.0"):
            MetricResult(
                metric_name="test",
                score=1.5,       # invalid
                passed=True,
                reasoning="test",
            )

    def test_metric_result_rejects_negative_score(self):
        with pytest.raises(ValueError, match="Score must be 0.0-1.0"):
            MetricResult(
                metric_name="test",
                score=-0.1,      # invalid
                passed=False,
                reasoning="test",
            )


class TestOpenEndedScoring:
    """
    Tests the short vs long answer scoring strategy.
    This was discovered as a real bug when running custom datasets.
    """
    metric = FactualAccuracyMetric()

    def make_custom_sample(self, expected: str) -> EvalSample:
        return EvalSample(
            id="test_custom_001",
            dataset="custom",
            question="test question",
            expected_answer=expected,
        )

    def test_short_answer_full_sentence_response_scores_high(self):
        """
        "Paris" expected, "The capital of France is Paris" given.
        Should score high — model is correct, just verbose.
        """
        result = self.metric.compute(
            self.make_custom_sample("Paris"),
            make_response("The capital of France is Paris.")
        )
        assert result.score >= 0.8, (
            f"Expected high score for correct verbose answer, got {result.score}"
        )

    def test_short_number_answer_scores_high(self):
        """
        "12" expected, "The square root of 144 is 12" given.
        """
        result = self.metric.compute(
            self.make_custom_sample("12"),
            make_response("The square root of 144 is 12.")
        )
        assert result.score >= 0.8

    def test_short_answer_wrong_response_scores_low(self):
        """
        "Paris" expected, "The capital of France is London" given.
        """
        result = self.metric.compute(
            self.make_custom_sample("Paris"),
            make_response("The capital of France is London.")
        )
        assert result.score == 0.0

    def test_short_answer_uses_recall_strategy(self):
        """Verify the strategy field is set correctly."""
        result = self.metric.compute(
            self.make_custom_sample("Paris"),
            make_response("The capital of France is Paris.")
        )
        assert result.raw_value["strategy"] == "recall_only"

    def test_long_answer_uses_f1_strategy(self):
        """Long expected answers should use F1."""
        result = self.metric.compute(
            self.make_custom_sample(
                "Refunds are processed within fourteen business days"
            ),
            make_response(
                "Refunds are processed within fourteen business days "
                "of receiving the returned item."
            )
        )
        assert result.raw_value["strategy"] == "f1"