# Benchmark Results — MMLU Evaluation

**Date:** June 2026  
**Dataset:** MMLU (50 questions, seed=42, diverse subjects)  
**Models evaluated:** 4  
**Metrics:** Factual Accuracy, Instruction Following, LLM Judge, 
             Latency, Cost Efficiency

## Results Table

| Model | Accuracy | Instruction | LLM Judge | Latency Score | Cost Score |
|---|---|---|---|---|---|
| openai/gpt-oss-120b | 1.000 | 1.000 | 1.000 | 0.933 | 0.975 |
| llama-3.3-70b-versatile | 0.900 | 0.960 | 0.900 | 0.985 | 0.910 |
| gpt-4o-mini | 0.800 | 1.000 | 0.800 | 0.894 | 0.981 |
| llama-3.1-8b-instant | 0.400 | 0.960 | 0.400 | 0.989 | 0.992 |

## Key Findings

**1. Groq's hardware changes the speed equation**  
llama-3.3-70b runs on Groq LPUs at ~400ms — faster than gpt-4o-mini 
at ~1100ms, despite being a larger model. Hardware matters as much 
as model size for latency.

**2. The 8B accuracy cliff**  
llama-3.1-8b scores 40% on MMLU — half the 70B model's accuracy. 
The LLM judge confirms these are genuine factual errors, not 
formatting issues. For accuracy-critical tasks, 8B is not 
a viable substitute for 70B.

**3. Cost vs accuracy tradeoff**  
llama-3.1-8b costs $0.000005/query vs $0.00008 for llama-3.3-70b 
— 16x cheaper, but with 2.25x worse accuracy. The right choice 
depends entirely on task requirements.

## Methodology

- Fixed random seed (42) ensures reproducibility
- LLM-as-judge uses GPT-4o-mini with temperature=0 
  for deterministic scoring
- Latency measured as wall-clock time including network
- Cost calculated from token counts × published API pricing