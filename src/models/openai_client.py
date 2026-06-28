import time
import yaml
from pathlib import Path
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv

from src.models.base_client import LLMClient, LLMResponse

load_dotenv()

def _load_pricing() -> dict:
        config_path = Path(__file__).parent.parent.parent / "configs" / "models.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        pricing = {}
        for model in config["models"]:
            if model["provider"] == "openai":
                pricing[model["name"]] = {
                    "input": model["pricing"]["input_per_1k"],
                    "output": model["pricing"]["output_per_1k"],
                }
        return pricing

class OpenAIClient(LLMClient):
    """
    Concrete implementation for OpenAI models.
    
    Notice: this class knows everything about OpenAI's API.
    The rest of the system knows nothing about it.
    That's the point.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        super().__init__(model_name)
        # OpenAI() automatically reads OPENAI_API_KEY from environment
        self.client = OpenAI()
        self.pricing = _load_pricing()

    def generate(self, prompt: str, max_tokens: int = 1024) -> LLMResponse:
        """
        Call OpenAI API and return a standardized LLMResponse.
        
        Notice we measure latency by wrapping the API call with time.time().
        We calculate cost immediately so it's never lost.
        """
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature = 0,
            )
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

        latency_ms = int((time.time() - start_time) * 1000)

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = self._calculate_cost(input_tokens, output_tokens)

        logger.info(
            f"OpenAI [{self.model_name}] | "
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
        """
        OpenAI uses tiktoken for tokenization.
        We approximate here — for production you'd use the tiktoken library.
        """
        # rough approximation: 1 token ≈ 4 characters for English
        return len(text) // 4

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Private method — only this class needs to know OpenAI's pricing.
        Uses _ prefix by convention to signal "internal use only".
        """
        pricing = self.pricing.get(self.model_name, self.pricing["gpt-4o-mini"])
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost