from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import time

class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class LLMResponse(BaseModel):
    content: str
    raw_response: Dict[str, Any] = Field(default_factory=dict)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    model_name: str
    error: Optional[str] = None

class BaseModelClient(ABC):
    """
    Abstract base class for all LLM clients.
    Ensures a consistent interface for the prompt execution engine.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model_name", "unknown-model")

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        """
        Generates a response from the LLM.
        
        Args:
           prompt: The user query or template-rendered prompt.
           system_prompt: Optional system instruction.
           **kwargs: Additional model-specific parameters (temp, top_p, etc.)
           
        Returns:
            LLMResponse object containing text, usage metrics, and latency.
        """
        pass
