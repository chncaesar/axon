import os
import logging
from typing import Any
from pydantic import BaseModel
from collections.abc import Iterator
from axon.llms.base_llm import BaseLLM
from axon.types import LLMMessage
from axon.tasks.task import Task
from axon.agents.agent import Agent
from axon.utilities.agent_utilities import is_context_length_exceeded
from axon.utilities.exceptions.context_window_exceeding_exception import LLMContextLengthExceededError

from openai import APIConnectionError, NotFoundError, OpenAI

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import ChoiceDelta


class OpenAICompletion(BaseLLM):
    """OpenAI completion implementation.
    This class provides direct integration with the OpenAI Python SDK,
    offering native structured outputs, streaming support.
    """
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
        default_query: dict[str, Any] | None = None,
        client_params: dict[str, Any] | None = None,
        temperature: float | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        reasoning_effort: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI completion client.
        """
        if provider is None:
            provider = kwargs.get("provider", "openai")

        self.model = model
        self.base_url = base_url
        self.organization = organization
        self.project = project
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_headers = default_headers
        self.default_query = default_query
        self.response_format = response_format
        self.stream = stream

        super.__init__(
            model = model,
            temperature = temperature,
            api_key = self.api_key,
            base_url = base_url,
            provider = provider,
            **kwargs
        )

        self.client = OpenAI(self._get_client_params())

    def _get_client_params(self) -> dict[str, Any]:
        """Get OpenAI client parameters."""

        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if self.api_key is None:
                raise ValueError("OPENAI_API_KEY is required")

        base_params = {
            "api_key": self.api_key,
            "organization": self.organization,
            "project": self.project,
            "base_url": self.base_url            
            or os.getenv("OPENAI_BASE_URL")
            or None,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }

        client_params = {k: v for k, v in base_params.items() if v is not None}

        if self.client_params:
            client_params.update(self.client_params)

        return client_params   

    def _format_messages(self, messages: str | list[LLMMessage]) -> list[LLMMessage]:        
        return super()._format_messages(messages)
        
    def _prepare_completion_params(
        self, messages: list[LLMMessage]
    ) -> dict[str, Any]:    
        """Prepare parameters for OpenAI chat completion."""
        params.update(self.additional_params)

        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.stream:
            params["stream"] = self.stream

        if self.temperature is not None:
            params["temperature"] = self.temperature
        

        # Filter out CrewAI-specific parameters that shouldn't go to the API
        crewai_specific_params = {
            "callbacks",
            "available_functions",
            "from_task",
            "from_agent",
            "provider",
            "api_key",
            "base_url",
            "api_base",
            "timeout",
        }

        return {k: v for k, v in params.items() if k not in crewai_specific_params}

    def _handle_streaming_completion(
        self,
        params: dict[str, Any],
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        """Handle streaming chat completion."""
        full_response = ""
                
        completion_stream: Iterator[ChatCompletionChunk] = self.client.chat.completions.create(
            **params
        )

        accumulated_content = ""
        for chunk in completion_stream:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta: ChoiceDelta = choice.delta

            if delta.content:
                accumulated_content += delta.content

        if response_model:                
            try:
                parsed_object = response_model.model_validate_json(accumulated_content)
                structured_json = parsed_object.model_dump_json()          
                return structured_json
            except Exception as e:
                logging.error(f"Failed to parse structured output from stream: {e}")                
                return accumulated_content

        accumulated_content = self._apply_stop_words(accumulated_content)        
        return accumulated_content

    def _handle_completion(
        self,
        params: dict[str, Any],
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle non-streaming chat completion."""
        try:
            if response_model:
                parsed_response = self.client.beta.chat.completions.parse(
                    **params,
                    response_format=response_model,
                )
                choice = parsed_response.choices[0].message

                if choice.refusal:
                    logging.warning(f"Model refusal: {choice.refusal}")
                    

                parsed_object = parsed_response.choices[0].message.parsed
                if parsed_object:
                    structured_json = parsed_object.model_dump_json()
                    return structured_json

            response: ChatCompletion = self.client.chat.completions.create(**params)
            choice: Choice = response.choices[0]
            message = choice.message

            content = message.content or ""
            content = self._apply_stop_words(content)

            if self.response_format and isinstance(self.response_format, type):
                try:
                    structured_result = self._validate_structured_output(
                        content, self.response_format
                    )
                    return structured_result
                except ValueError as e:
                    logging.warning(f"Structured output validation failed: {e}")

        except NotFoundError as e:
            error_msg = f"Model {self.model} not found: {e}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise ValueError(error_msg) from e
        except APIConnectionError as e:
            error_msg = f"Failed to connect to OpenAI API: {e}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise ConnectionError(error_msg) from e
        except Exception as e:
            # Handle context length exceeded and other errors
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e

            error_msg = f"OpenAI API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise e from e

        return content

    def call(
        self,
        messages: str | list[LLMMessage],
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Call OpenAI chat completion api.
        Args:
            messages: Input message of the api call.
            from_task: Task that initiated the call.
            from_agent: Agent that initiated the call.
            response_model: Response model of the structured output.
        
        Returns:
            chat completion response.            
        """
        try:
            formatted_messages = self._format_messages(messages)

            completion_params = self._prepare_completion_params(
                messages=formatted_messages
            )

            if self.stream:
                return self._handle_streaming_completion(
                    params=completion_params,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_model=response_model,
                )

            return self._handle_completion(
                params=completion_params,
                from_task=from_task,
                from_agent=from_agent,
                response_model=response_model,
            )

        except Exception as e:
            error_msg = f"OpenAI API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise
    
    def get_context_window_size(self) -> int:
        """Get the context window size for the model."""
        from axon.llms.llm import CONTEXT_WINDOW_USAGE_RATIO, LLM_CONTEXT_WINDOW_SIZES

        min_context = 1024
        max_context = 2097152

        for key, value in LLM_CONTEXT_WINDOW_SIZES.items():
            if value < min_context or value > max_context:
                raise ValueError(
                    f"Context window for {key} must be between {min_context} and {max_context}"
                )

        # Context window sizes for OpenAI models
        context_windows = {
            "gpt-4": 8192,
            "gpt-4o": 128000,
            "gpt-4o-mini": 200000,
            "gpt-4-turbo": 128000,
            "gpt-4.1": 1047576,
            "gpt-4.1-mini-2025-04-14": 1047576,
            "gpt-4.1-nano-2025-04-14": 1047576,
            "o1-preview": 128000,
            "o1-mini": 128000,
            "o3-mini": 200000,
            "o4-mini": 200000,
        }

        # Find the best match for the model name
        for model_prefix, size in context_windows.items():
            if self.model.startswith(model_prefix):
                return int(size * CONTEXT_WINDOW_USAGE_RATIO)

        # Default context window size
        return int(8192 * CONTEXT_WINDOW_USAGE_RATIO)