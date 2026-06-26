from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """
    A standard envelope that every model client must return.
    
    Why a dataclass? Because we need structured data, not a raw string.
    Every part of the system that receives a response knows exactly
    what fields to expect — regardless of which model produced it.
    """
    content: str            # the actual text response
    model_name: str         # which model produced this
    input_tokens: int       # tokens in the prompt
    output_tokens: int      # tokens in the response
    latency_ms: int         # how long the call took
    cost_usd: float         # calculated cost in dollars
    raw_response: Optional[dict] = None  # full API response for debugging


class LLMClient(ABC):
    """
    Abstract base class for all LLM clients.
    
    Why ABC (Abstract Base Class)?
    - It enforces a contract: any class inheriting from LLMClient
      MUST implement generate() and count_tokens()
    - If you forget to implement one, Python raises an error at
      import time, not at runtime when it's too late
    - The evaluator only ever types: client: LLMClient
      It never knows or cares which concrete client it's using
    
    This is the Strategy Pattern.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1024) -> LLMResponse:
        """
        Send a prompt, get a structured response back.
        Every subclass must implement this.
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count how many tokens a string uses for this model.
        Different models use different tokenizers, so each
        client implements this differently.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"