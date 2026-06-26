import time
import yaml
from pathlib import Path
from groq import Groq
from loguru import logger
from dotenv import load_dotenv

from src.models.base_client import LLMClient, LLMResponse

load_dotenv()


def _load_pricing() -> dict:
    """
    Read pricing from configs/models.yaml instead of
    hardcoding it here. Single source of truth.
    """
    config_path = Path(__file__).parent.parent.parent / "configs" / "models.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    pricing = {}
    for model in config["models"]:
        if model["provider"] == "groq":
            pricing[model["name"]] = {
                "input": model["pricing"]["input_per_1k"],
                "output": model["pricing"]["output_per_1k"],
            }
    return pricing


class GroqClient(LLMClient):
    """
    Concrete LLM client for Groq-hosted open source models.
    
    Groq's SDK is OpenAI-compatible, so this looks very similar
    to OpenAIClient. The difference is:
    - Different API key (GROQ_API_KEY)
    - Different model names (llama, mixtral, gemma)
    - Different pricing
    - Groq is significantly faster (LPU hardware)
    
    The evaluator doesn't know or care about any of this.
    It just calls client.generate(prompt).
    """

    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        super().__init__(model_name)
        # Groq() automatically reads GROQ_API_KEY from environment
        self.client = Groq()
        self.pricing = _load_pricing()

    def generate(self, prompt: str, max_tokens: int = 1024) -> LLMResponse:
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"Groq API call failed for {self.model_name}: {e}")
            raise

        latency_ms = int((time.time() - start_time) * 1000)

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = self._calculate_cost(input_tokens, output_tokens)

        logger.info(
            f"Groq [{self.model_name}] | "
            f"{latency_ms}ms | "
            f"{input_tokens}in/{output_tokens}out tokens | "
            f"${cost:.6f}"
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model_name=self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            raw_response=response.model_dump(),
        )

    def count_tokens(self, text: str) -> int:
        # Llama models use a similar tokenizer to GPT
        # approximation: 1 token ≈ 4 characters
        return len(text) // 4

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        # Fall back to 8b pricing if model not found in config
        pricing = self.pricing.get(
            self.model_name,
            {"input": 0.00005, "output": 0.00008}
        )
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost

