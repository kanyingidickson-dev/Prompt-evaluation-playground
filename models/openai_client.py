import os
import time
from typing import Dict, Any, Optional
from .base_model import BaseModelClient, LLMResponse, TokenUsage

try:
    from openai import OpenAI, OpenAIError
except ImportError:
    OpenAI = None
    OpenAIError = Exception

class OpenAIClient(BaseModelClient):
    """
    Client for OpenAI's Chat Completions API.
    Requires 'openai' package and an API Key.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        self.api_key = os.getenv(api_key_env)
        self.client = None
        
        if self.api_key and OpenAI:
            self.client = OpenAI(api_key=self.api_key)
        
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        if not self.client:
           return LLMResponse(
               content="",
               model_name=self.model_name,
               error="OpenAI client not initialized. Check OPENAI_API_KEY and 'openai' package."
           )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        model_params = {
            "model": self.config.get("model_name", "gpt-3.5-turbo"),
            "temperature": self.config.get("temperature", 0.7),
            "max_tokens": self.config.get("max_tokens", 1024),
            **kwargs 
        }

        start_time = time.perf_counter()
        
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                **model_params
            )
            
            end_time = time.perf_counter()
            content = response.choices[0].message.content
            
            # Extract usage
            u = response.usage
            usage = TokenUsage(
                prompt_tokens=u.prompt_tokens,
                completion_tokens=u.completion_tokens,
                total_tokens=u.total_tokens
            )
            
            return LLMResponse(
                content=content,
                raw_response=response.model_dump(),
                token_usage=usage,
                latency_ms=(end_time - start_time) * 1000,
                model_name=self.model_name
            )

        except Exception as e:
            return LLMResponse(
                content="",
                model_name=self.model_name,
                error=str(e),
                latency_ms=(time.perf_counter() - start_time) * 1000
            )
