from typing import Literal
from pydantic import BaseModel
from axon.utilities.exceptions.context_window_exceeding_exception import LLMContextLengthExceededError
from axon.types import LLMMessage
from axon.utilities.printer import Printer
from axon.agents.parser import (
    AgentAction,
    AgentFinish,
    OutputParserError,
    parse,
)
from axon.prompt.prompt import Prompt
from axon.llms.llm import LLM
from axon.llms.base_llm import BaseLLM
from axon.tasks.task import Task
from axon.agents.agent import Agent
from axon.agents.agent_executor import AgentExecutor
from axon.agents.constants import FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE

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

def has_reached_max_iterations(iterations: int, max_iterations: int) -> bool:
    return iterations >= max_iterations    

def format_answer(answer: str) -> AgentAction | AgentFinish:
    """Format a response from the LLM into an AgentAction or AgentFinish.

    Args:
        answer: The raw response from the LLM

    Returns:
        Either an AgentAction or AgentFinish
    """
    try:
        return parse(answer)
    except Exception:
        return AgentFinish(
            thought="Failed to parse LLM response",
            output=answer,
            text=answer,
        )


def handle_max_iterations_exceeded(
    formatted_answer : AgentAction | AgentFinish | None,
    printer: Printer,
    prompt: Prompt,
    messages: list[LLMMessage],
    llm: LLM | BaseLLM,
) -> AgentFinish:
    printer.print(
        content="Maximum iterations reached. Requesting final answer.",
        color="yellow",
    )
    if formatted_answer and hasattr(formatted_answer, "text"):
        assistant_message = (
            formatted_answer.text + f"\n{prompt.errors('force_final_answer')}"
        )
    else:
        assistant_message = prompt.errors("force_final_answer")

    messages.append(format_message_for_llm(assistant_message, role="assistant"))    
    # Perform one more LLM call to get the final answer
    answer = llm.call(messages)
    if answer is None or answer == "":
        printer.print(
            content="Received None or empty response from LLM call.",
            color="red",
        )
        raise ValueError("Invalid response from LLM call - None or empty.")

    formatted = format_answer(answer=answer)

    if isinstance(formatted, AgentFinish):
        return formatted
    return AgentFinish(
        thought=formatted.thought,
        output=formatted.text,
        text=formatted.text,
    )
   
def get_llm_response(
    llm: LLM | BaseLLM,
    messages: list[LLMMessage],
    printer: Printer,
    from_task: Task | None = None,
    from_agent: Agent | None = None,
    response_model: type[BaseModel] | None = None,
    executor_context: AgentExecutor | None = None,
) -> str:
    """Call the LLM and return the response, handling any invalid responses.

    Args:
        llm: The LLM instance to call
        messages: The messages to send to the LLM
        printer: Printer instance for output
        from_task: Optional task context for the LLM call
        from_agent: Optional agent context for the LLM call
        response_model: Optional Pydantic model for structured outputs
        executor_context: Optional executor context for hook invocation

    Returns:
        The response from the LLM as a string

    Raises:
        Exception: If an error occurs.
        ValueError: If the response is None or empty.
    """

    if executor_context is not None:        
        messages = executor_context.messages

    try:
        answer = llm.call(
            messages,            
            from_task=from_task,
            from_agent=from_agent,  # type: ignore[arg-type]
            response_model=response_model,
        )
    except Exception as e:
        raise e
    if not answer:
        printer.print(
            content="Received None or empty response from LLM call.",
            color="red",
        )
        raise ValueError("Invalid response from LLM call - None or empty.")

    return answer

def process_llm_response(
    answer: str,
    use_stop_words: bool
) -> AgentAction | AgentFinish:
    """Process the LLM response and format it into an AgentAction or AgentFinish.

    Args:
        answer: The raw response from the LLM
        use_stop_words: Whether to use stop words in the LLM call

    Returns:
        Either an AgentAction or AgentFinish
    """
    if not use_stop_words:
        try:
            format_answer(answer)
        except OutputParserError as e:
            if FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE in e.error:
                answer = answer.split("Observation:")[0].strip()

    return format_answer(answer)

def handle_output_parser_exception(
    e: OutputParserError,
    messages: list[LLMMessage],
    iterations: int,
    log_error_after: int = 3,
    printer: Printer | None = None,
) -> AgentAction:
    """Handle OutputParserError by updating messages and formatted_answer.

    Args:
        e: The OutputParserError that occurred
        messages: List of messages to append to
        iterations: Current iteration count
        log_error_after: Number of iterations after which to log errors
        printer: Optional printer instance for logging

    Returns:
        AgentAction: A formatted answer with the error
    """
    messages.append({"role":"user", "content": e.error})
    formatted_answer = AgentAction(
        text = e.error,
        tool = "",
        tool_input = "",
        thought = ""
    )

    if printer and  iterations > log_error_after:
        printer.print(
            content = f"Error parsing LLM output, agent will retry , {e.error}",
            color = "red",
        )
    
    return formatted_answer

def is_context_length_exceeded(exception: Exception) -> bool:
    """Check if the exception is due to context length exceeding.

    Args:
        exception: The exception to check

    Returns:
        bool: True if the exception is due to context length exceeding
    """
    return LLMContextLengthExceededError(str(exception))._is_context_limit_error(str(exception))

def summarize_messages(
    messages: list[LLMMessage],
    llm: LLM | BaseLLM,
    prompt: Prompt
) -> list[LLMMessage]:
    """Summarize messages to fit within context window.

    Args:
        messages: List of messages to summarize
        llm: LLM instance for summarization
        prompt: prompt instance for messages
    """
    messages_str = " ".join([message["content"] for message in messages])
    cut_size = llm.get_context_window_size()
    message_group = [{"content": messages_str[i: i + cut_size]} for i in range(0, len(messages_str), cut_size)]

    summarized_messages = []
    group_size = len(message_group)

    for idx, group in enumerate(message_group):
        Printer.print(f"Summarizing group {idx + 1} of {group_size}", color="yellow")
        messages = [
            format_message_for_llm(prompt.slice("summarizer_system_message"), role="system"),
            format_message_for_llm(prompt.slice("summarize_instruction").format(group=group), role="user"),
        ]

        summary = llm.call(messages)
        summarized_messages.append({"content": str(summary)})
   
    merged_summary = " ".join([message["content"] for message in summarized_messages])
    
    result = [
        format_message_for_llm(prompt.slice("summary").format(summary=merged_summary)),
    ]
    return result

def handle_context_length(
    messages: list[LLMMessage],
    llm: LLM | BaseLLM,
    prompt: Prompt,
    printer: Printer
) -> list[LLMMessage]:
    """Handle context length by summarizing messages if needed."""
    if llm.get_context_window_size() < len(messages):
        printer.print(
            content="Context length exceeded. Summarizing content to fit the model context window. Might take a while...",
            color="yellow",
        )
        return summarize_messages(messages, llm, prompt)
    return messages