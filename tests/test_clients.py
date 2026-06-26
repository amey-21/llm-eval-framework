# tests/test_clients.py

import pytest
from src.models.base_client import LLMResponse
from src.models.openai_client import OpenAIClient
from src.models.groq_client import GroqClient


def assert_valid_response(response: LLMResponse, expected_model: str):
    """Helper that checks every field of LLMResponse is sane."""
    assert isinstance(response, LLMResponse)
    assert isinstance(response.content, str)
    assert len(response.content) > 0,      "response should not be empty"
    assert response.model_name == expected_model
    assert response.input_tokens > 0,      "must have counted input tokens"
    assert response.output_tokens > 0,     "must have counted output tokens"
    assert response.latency_ms > 0,        "latency must be positive"
    assert response.cost_usd >= 0,         "cost must be non-negative"


def test_openai_client_returns_valid_response():
    client = OpenAIClient(model_name="gpt-4o-mini")
    response = client.generate("Say hello in exactly 3 words.")
    assert_valid_response(response, "gpt-4o-mini")
    print(f"\nOpenAI response: {response.content}")
    print(f"Latency: {response.latency_ms}ms | Cost: ${response.cost_usd:.6f}")


def test_groq_llama_8b_returns_valid_response():
    client = GroqClient(model_name="llama-3.1-8b-instant")
    response = client.generate("Say hello in exactly 3 words.")
    assert_valid_response(response, "llama-3.1-8b-instant")
    print(f"\nGroq 8B response: {response.content}")
    print(f"Latency: {response.latency_ms}ms | Cost: ${response.cost_usd:.6f}")


def test_groq_llama_70b_returns_valid_response():
    client = GroqClient(model_name="llama-3.3-70b-versatile")
    response = client.generate("Say hello in exactly 3 words.")
    assert_valid_response(response, "llama-3.3-70b-versatile")
    print(f"\nGroq 70B response: {response.content}")
    print(f"Latency: {response.latency_ms}ms | Cost: ${response.cost_usd:.6f}")

def test_groq_gpt_oss_120b_returns_valid_response():
    client = GroqClient(model_name="openai/gpt-oss-120b")
    response = client.generate("Say hello in exactly 3 words.")
    assert_valid_response(response, "openai/gpt-oss-120b")
    print(f"\nGroq GPT-OSS 120B response: {response.content}")
    print(f"Latency: {response.latency_ms}ms | Cost: ${response.cost_usd:.6f}")


def test_all_clients_return_same_interface():
    """
    This test captures the entire point of the abstract base class.
    Every client returns the same LLMResponse shape — the caller
    never needs to know which model it's talking to.
    """
    clients = [
        OpenAIClient("gpt-4o-mini"),
        GroqClient("llama-3.1-8b-instant"),
        GroqClient("llama-3.3-70b-versatile"),
        GroqClient("openai/gpt-oss-120b"),
    ]
    prompt = "What is 2 + 2? Answer with just the number."

    for client in clients:
        response = client.generate(prompt)
        assert_valid_response(response, client.model_name)
        # Every single client returns "4" — different models,
        # same interface, predictable behavior
        assert "4" in response.content