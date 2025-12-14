from typing import Literal
from axon.utilities.exceptions.context_window_exceeding_exception import LLMContextLengthExceededError
from axon.types import LLMMessage
from axon.utilities.printer import Printer

def is_context_length_exceeded(exception: Exception) -> bool:
    """Check if the exception is due to context length exceeding.

    Args:
        exception: The exception to check

    Returns:
        bool: True if the exception is due to context length exceeding
    """
    return LLMContextLengthExceededError(str(exception))._is_context_limit_error(
        str(exception)
    )

def format_message_for_llm(
    prompt: str, role: Literal["user", "assistant", "system"] = "user"
) -> LLMMessage:
    """Format a message for the LLM.

    Args:
        prompt:  The message content.
        role:  The role of the message sender, either 'user' or 'assistant'.

    Returns:
        A dictionary with 'role' and 'content' keys.

    """
    prompt = prompt.rstrip()
    return {"role": role, "content": prompt}

def handle_unknown_error(printer: Printer, exception: Exception) -> None:
    """Handle unknown errors by informing the user.

    Args:
        printer: Printer instance for output
        exception: The exception that occurred
    """
    error_message = str(exception)

    if "litellm" in error_message:
        return

    printer.print(
        content="An unknown error occurred. Please check the details below.",
        color="red",
    )
    printer.print(
        content=f"Error details: {error_message}",
        color="red",
    )