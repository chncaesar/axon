import httpx
from __future__ import annotations

from typing import Any, Final, cast, Literal
from typing_extensions import Self
from pydantic import BaseModel

from axon.agents.base_agent import BaseLLM
from axon.llms.base_interceptor import BaseInterceptor

SUPPORTED_NATIVE_PROVIDERS: Final[list[str]] = [
    "openai",    
]

class LLM(BaseLLM):
    def __new__(cls, model: str, **kwargs: Any) -> LLM | None:
        """Factory method that routes to native SDK or falls back to LiteLLM.

        Routing priority:
            1. If 'provider' kwarg is present, use that provider with constants
            2. If only 'model' kwarg, use constants to infer provider
            3. If "/" in model name:
               - Check if prefix is a native provider (openai/anthropic/azure/bedrock/gemini)
               - If yes, validate model against constants
               - If valid, route to native SDK; otherwise route to LiteLLM
        """

        if not model or not isinstance(model, str):
            raise ValueError("model must be a non-empty string")
        
        explicit_provider = kwargs.get("provider")

        if explicit_provider:
            provider = explicit_provider
            model_str = model
        elif "/" in model:
            prefix, _, model_part = model.partition("/")
            provider_mapping = {
                "openai": "openai"                
            }
            _provider = provider_mapping.get(prefix.lower())

            if _provider:
                provider = _provider
            else:
                provider = prefix
            model_str = model_part

        native_class = cls._get_provider_class(provider)  
        if provider in SUPPORTED_NATIVE_PROVIDERS and native_class:
            try:
                kwargs_copy = {k: v for k , v in kwargs.items() if k != "provider" }
                return cast(Self, native_class(model=model_str, provider=provider, **kwargs_copy))
            except NotImplementedError:
                raise
            except Exception as e:
                raise ImportError(f"Error importing provider {provider}'s class {e}") from e
        
        # return None, if provider is not supported
        return None        

    

    @classmethod
    def _get_provider_class( provider: str) -> type | None:
        """Get native provider class if available."""
        if provider == "openai":
            from axon.llms.providers.openai.openai_completion import OpenAICompletion
            return OpenAICompletion
        
        return None

    def get_context_window_size(self) -> int:
        return self.context_window_size



    
