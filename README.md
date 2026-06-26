# LLM Evaluation Framework

A production-grade benchmarking system that evaluates multiple LLMs 
across quality, speed, and cost dimensions — with automated regression 
detection on every commit.

[![Unit Tests](https://github.com/YOUR_USERNAME/llm-eval-framework/actions/workflows/tests.yaml/badge.svg)](https://github.com/YOUR_USERNAME/llm-eval-framework/actions)
[![Regression Check](https://github.com/YOUR_USERNAME/llm-eval-framework/actions/workflows/eval_ci.yaml/badge.svg)](https://github.com/YOUR_USERNAME/llm-eval-framework/actions)

## Key Findings

Evaluated 4 models across 50 MMLU questions (5 metrics each):

| Model | Factual Accuracy | Latency | Cost/query |
|---|---|---|---|
| openai/gpt-oss-120b | 100% | ~800ms | $0.00003 |
| llama-3.3-70b (Groq) | 90% | ~400ms | $0.00008 |
| gpt-4o-mini (OpenAI) | 80% | ~1100ms | $0.00002 |
| llama-3.1-8b (Groq) | 40% | ~250ms | $0.000005 |

**Key insight:** Groq's LPU hardware makes the 70B model faster than 
OpenAI's smaller model. Size doesn't determine speed when the 
hardware differs.

## Architecture
## What Makes This Production-Grade

- **Automated regression detection** — CI fails if factual accuracy 
  drops >10% vs committed baseline
- **LLM-as-judge** — GPT-4o evaluates subjective quality dimensions
  that rule-based metrics can't capture
- **Three-tier architecture** — dashboard never touches the database
  directly; all data access through FastAPI
- **Strategy Pattern** — adding a new model provider = one new file,
  zero changes to evaluation logic
- **Fixed-seed sampling** — CI always evaluates identical questions,
  making score changes meaningful not noisy

## Metrics

| Metric | Method | Notes |
|---|---|---|
| Factual Accuracy | Letter extraction + F1 overlap | MMLU: exact match, TruthfulQA: token F1 |
| Instruction Following | Rule-based format checks | Penalizes verbose answers on MCQ |
| Latency | Wall-clock measurement | Observed in dashboard, not CI-gated |
| Cost Efficiency | Token × price from YAML config | Single source of truth for pricing |
| LLM Judge | GPT-4o with structured rubric | Diagnoses failure modes, not just pass/fail |

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/llm-eval-framework
cd llm-eval-framework
python -m venv .venv && source .venv/bin/activate
pip install -e .

# add API keys
cp .env.example .env

# start postgres
docker run -d --name llm-eval-postgres \
  -e POSTGRES_DB=llm_eval \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 postgres:16

# run an evaluation
python scripts/run_eval.py

# start the dashboard
uvicorn src.api.main:app --port 8000 &
streamlit run dashboard/app.py
```

## CI

Every PR triggers:
1. 36 unit tests (~3 seconds, no API calls)
2. 20-sample eval across all models
3. Regression check against committed baseline
4. Build fails if factual accuracy or instruction following 
   drops more than 10%

## Adding a New Model

1. Create `src/models/your_client.py` inheriting from `LLMClient`
2. Add the model to `configs/models.yaml`
3. Add it to the client list in `scripts/run_eval.py`

Nothing else changes. That's the Strategy Pattern.

## Stack

Python · FastAPI · Streamlit · PostgreSQL · Docker · 
GitHub Actions · OpenAI API · Groq API · HuggingFace Datasets