import time
import random
from typing import Dict, Any, Optional
from .base_model import BaseModelClient, LLMResponse, TokenUsage

class LocalModelClient(BaseModelClient):
    """
    A local mock model client for testing, debugging, and development 
    without incurring API costs or requiring internet access.
    """

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        start_time = time.perf_counter()
        
        # Simulate processing time based on config or random
        latency_sim = self.config.get("latency_ms", 50) / 1000.0
        time.sleep(latency_sim)
        
        # Deterministic dummy response logic
        # We'll just echo parts of the prompt to simulate "relevance"
        response_text = (
            f"[LOCAL_MODEL_DEBUG] Received prompt length: {len(prompt)}. "
            f"Simulated response to: '{prompt[:50]}...'. "
            "This is a safe, locally generated response for testing purposes."
        )

        end_time = time.perf_counter()
        
        # Mock usage
        usage = TokenUsage(
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(response_text) // 4,
            total_tokens=(len(prompt) + len(response_text)) // 4
        )
        
        return LLMResponse(
            content=response_text,
            raw_response={"mock_id": "local-123"},
            token_usage=usage,
            latency_ms=(end_time - start_time) * 1000,
            model_name=self.model_name
        )
