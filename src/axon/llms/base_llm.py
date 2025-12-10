import logging
import re
import json
from abc import ABC, abstractmethod
from typing import Any, Final
from pydantic import BaseModel

from axon.types import LLMMessage
from axon.tasks.task import Task
from axon.agents.agent import Agent

DEFAULT_CONTEXT_WINDOW_SIZE: Final[int] = 8192
_JSON_EXTRACTION_PATTERN: Final[re.Pattern[str]] = re.compile(r"\{.*}", re.DOTALL)


class BaseLLM(ABC):
    """Abstract base class for LLM implementations.
    """

    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
        **kwargs : Any
    ) -> None:
        """ Creates an BaseLLM instances
        Args:
            model: The model name
            temperature: Optional model's temperature
            api_key: Model's api key
            base_url: Model base url
            provider: Model provider, for instance anthropic
            kwargs: Additional provider specific parameters            
        """
        if not model:
            raise ValueError("Model name is required")

        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.base_url = base_url
        self.provider = provider
        self.additional_parameters = kwargs

    def _format_messages(self, messages: str | list[LLMMessage]) -> list[LLMMessage]:
        """Convert messages to standard format.

        Args:
            messages: Input messages (string or list of message dicts)

        Returns:
            List of message dictionaries with 'role' and 'content' keys

        Raises:
            ValueError: If message format is invalid
        """
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        # Validate message format
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"Message at index {i} must be a dictionary")
            if "role" not in msg or "content" not in msg:
                raise ValueError(
                    f"Message at index {i} must have 'role' and 'content' keys"
                )

        return messages

    def _apply_stop_words(self, content: str) -> str:
        """Apply stop words to truncate response content.

        This method provides consistent stop word behavior across all native SDK providers.
        Native providers should call this method to post-process their responses.

        Args:
            content: The raw response content from the LLM

        Returns:
            Content truncated at the first occurrence of any stop word

        Example:
            >>> llm = MyNativeLLM(stop=["Observation:", "Final Answer:"])
            >>> response = (
            ...     "I need to search.\\n\\nAction: search\\nObservation: Found results"
            ... )
            >>> llm._apply_stop_words(response)
            "I need to search.\\n\\nAction: search"
        """
        if not self.stop or not content:
            return content

        # Find the earliest occurrence of any stop word
        earliest_stop_pos = len(content)
        found_stop_word = None

        for stop_word in self.stop:
            stop_pos = content.find(stop_word)
            if stop_pos != -1 and stop_pos < earliest_stop_pos:
                earliest_stop_pos = stop_pos
                found_stop_word = stop_word

        # Truncate at the stop word if found
        if found_stop_word is not None:
            truncated = content[:earliest_stop_pos].strip()
            logging.debug(
                f"Applied stop word '{found_stop_word}' at position {earliest_stop_pos}"
            )
            return truncated

        return content    
    
    @staticmethod
    def _validate_structured_output(
        response: str,
        response_format: type[BaseModel] | None,
    ) -> str | BaseModel:
        """Validate and parse structured output.

        Args:
            response: Raw response string
            response_format: Optional Pydantic model for structured output

        Returns:
            Parsed response (BaseModel instance if response_format provided, otherwise string)

        Raises:
            ValueError: If structured output validation fails
        """
        if response_format is None:
            return response

        try:
            # Try to parse as JSON first
            if response.strip().startswith("{") or response.strip().startswith("["):
                data = json.loads(response)
                return response_format.model_validate(data)

            json_match = _JSON_EXTRACTION_PATTERN.search(response)
            if json_match:
                data = json.loads(json_match.group())
                return response_format.model_validate(data)

            raise ValueError("No JSON found in response")

        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"Failed to parse structured output: {e}")
            raise ValueError(
                f"Failed to parse response into {response_format.__name__}: {e}"
            ) from e

    @abstractmethod
    def call(
        self,
        message: str | list[LLMMessage],
        from_task: Task | None = None,
        from_agent: Agent | None = None,
    ) -> str:
        """Call the LLM with the given messages.

        Args:
            message: Input messages for the LLM.
                        Can be a string or list of message dictionaries.
                        If string, it will be converted to a single user message.
                        If list, each dict must have 'role' and 'content' keys.
            from_task: Optional task context
            from_agent: Optional agent context
        Returns:
            A text response from the LLM (str)                         

        Raises:
            ValueError: If the messages format is invalid.
            TimeoutError: If the LLM request times out.
            RuntimeError: If the LLM request fails for other reasons.
        """        
