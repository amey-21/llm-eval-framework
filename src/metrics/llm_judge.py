# src/metrics/llm_judge.py

import json
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from src.metrics.base_metric import BaseMetric, MetricResult
from src.data.dataset_loader import EvalSample
from src.models.base_client import LLMResponse

load_dotenv()


class JudgeOutput(BaseModel):
    """
    Structured output from the LLM judge.
    
    Why Pydantic here?
    LLMs can return malformed JSON. Pydantic validates the
    structure and gives clear errors if fields are missing.
    This makes the judge reliable enough to run in CI.
    """
    score: int              # 1-5
    correct: bool
    confidence: str         # "high", "medium", "low"
    failure_mode: str       # "correct", "factual_error", "misread_question",
                            # "refused", "extraction_failed", "off_topic"
    reasoning: str          # one sentence explanation


JUDGE_SYSTEM_PROMPT = """You are an expert AI evaluation judge.
Your job is to assess whether a model's response correctly answers a question.

You must respond with ONLY valid JSON matching this exact structure:
{
  "score": <integer 1-5>,
  "correct": <boolean>,
  "confidence": <"high"|"medium"|"low">,
  "failure_mode": <"correct"|"factual_error"|"misread_question"|"refused"|"extraction_failed"|"off_topic">,
  "reasoning": <one sentence string>
}

Scoring rubric:
5 = Correct and well-reasoned
4 = Correct but incomplete or verbose  
3 = Partially correct
2 = Incorrect but shows some understanding
1 = Completely wrong or refused to answer

failure_mode options:
- correct: answer is right
- factual_error: model stated wrong facts confidently
- misread_question: model answered a different question
- refused: model declined to answer
- extraction_failed: answer is ambiguous or unextractable
- off_topic: model went off on a tangent

IMPORTANT: Your 'reasoning' field must be plain English only.
Do not copy the question text. Do not use LaTeX, math notation,
or special characters. One sentence maximum."""


class LLMJudgeMetric(BaseMetric):
    """
    Uses GPT-4o to evaluate response quality.
    
    Why GPT-4o as the judge and not Groq?
    The judge needs to be the most capable, reliable model
    we have access to. Its job is to evaluate others —
    a weak judge produces unreliable evaluations.
    This is why companies use frontier models as judges
    even when evaluating cheaper models.
    
    Cost note: this doubles your API costs since every
    response gets evaluated by GPT-4o. Use sparingly
    during development — run on 10 samples, not 500.
    """

    name = "llm_judge"
    threshold = 0.6

    def __init__(self):
        self.client = OpenAI()

    def compute(self, sample: EvalSample, response: LLMResponse) -> MetricResult:
        if not response.content:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                reasoning="Empty response — nothing to judge",
                raw_value={"failure_mode": "empty_response"}
            )

        judge_output = self._call_judge(sample, response)

        if judge_output is None:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                reasoning="Judge call failed",
                raw_value={"failure_mode": "judge_error"}
            )

        # normalize 1-5 scale to 0.0-1.0
        normalized_score = (judge_output.score - 1) / 4

        return MetricResult(
            metric_name=self.name,
            score=round(normalized_score, 4),
            passed=judge_output.correct,
            reasoning=judge_output.reasoning,
            raw_value={
                "raw_score": judge_output.score,
                "correct": judge_output.correct,
                "confidence": judge_output.confidence,
                "failure_mode": judge_output.failure_mode,
            }
        )

    def _call_judge(
        self, sample: EvalSample, response: LLMResponse
    ) -> JudgeOutput | None:
        """
        Call GPT-4o with the evaluation prompt.
        
        Robust JSON parsing handles two failure modes:
        1. Model wraps JSON in markdown fences — strip them
        2. Question text contains LaTeX backslashes — sanitize
        before parsing
        """

        # sanitize the question before embedding it in the prompt
        # LaTeX backslashes (\frac, \alpha) break JSON parsing
        # when the judge includes them in its reasoning field
        safe_question = sample.question.replace("\\", "/")
        safe_expected = sample.expected_answer.replace("\\", "/")
        safe_response = response.content.replace("\\", "/")

        user_prompt = f"""Evaluate this model response:

    QUESTION:
    {safe_question}

    EXPECTED ANSWER:
    {safe_expected}

    MODEL RESPONSE:
    {safe_response}

    MODEL BEING EVALUATED: {response.model_name}

    Respond with JSON only. Do not include any LaTeX or special characters in your reasoning field."""

        try:
            api_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=300,
                temperature=0,
            )

            raw = api_response.choices[0].message.content.strip()

            # strip markdown fences if present
            if "```" in raw:
                # extract content between fences
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            
            # strip any leading/trailing whitespace again after fence removal
            raw = raw.strip()

            # sanitize remaining backslashes that aren't valid JSON escapes
            # valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
            # everything else needs to be escaped
            import re
            raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)

            parsed = json.loads(raw)
            return JudgeOutput(**parsed)

        except json.JSONDecodeError as e:
            logger.error(f"Judge JSON parse failed: {e}")
            logger.debug(f"Raw judge output was: {raw[:300]}")
            return None
        except Exception as e:
            logger.error(f"Judge call failed: {e}")
            return None