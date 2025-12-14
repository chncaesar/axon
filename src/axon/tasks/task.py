import json
from typing import Any
from pydantic import BaseModel,Field,PrivateAttr
from axon.utilities.string_utils import interpolate_only
from axon.prompt.prompt import Prompt, get_prompt
from axon.utilities.printer import Printer


_printer = Printer()


class Task(BaseModel):
    """"""
    prompt: Prompt = Field(default_factory=get_prompt)
    description: str = Field(description="Description of the actual task.")
    expected_output: str = Field(
        description="Clear definition of expected output for the task."
    )
    _original_description: str | None = PrivateAttr(default=None)
    _original_expected_output: str | None = PrivateAttr(default=None)
    _original_output_file: str | None = PrivateAttr(default=None)
    output_file: str | None = Field(
        description="A file path to be used to create a file output.",
        default=None,
    )
    callback: Any | None = Field(
        description="Callback to be executed after the task is completed.", default=None
    )
    allow_trigger_context: bool | None = Field(
        default=None,
        description="Whether this task should append 'Trigger Payload: {crewai_trigger_payload}' to the task description when crewai_trigger_payload exists in crew inputs.",
    )

    def interpolate_inputs_and_add_conversation_history(
        self, inputs: dict[str, str | int | float | dict[str, Any] | list[Any]]
    ) -> None:
        """Interpolate inputs into the task description, expected output, and output file path.
           Add conversation history if present.

        Args:
            inputs: Dictionary mapping template variables to their values.
                   Supported value types are strings, integers, and floats.

        Raises:
            ValueError: If a required template variable is missing from inputs.
        """
        if self._original_description is None:
            self._original_description = self.description
        if self._original_expected_output is None:
            self._original_expected_output = self.expected_output
        if self.output_file is not None and self._original_output_file is None:
            self._original_output_file = self.output_file

        if not inputs:
            return

        try:
            self.description = interpolate_only(
                input_string=self._original_description, inputs=inputs
            )
        except KeyError as e:
            raise ValueError(
                f"Missing required template variable '{e.args[0]}' in description"
            ) from e
        except ValueError as e:
            raise ValueError(f"Error interpolating description: {e!s}") from e

        try:
            self.expected_output = interpolate_only(
                input_string=self._original_expected_output, inputs=inputs
            )
        except (KeyError, ValueError) as e:
            raise ValueError(f"Error interpolating expected_output: {e!s}") from e

        if self.output_file is not None:
            try:
                self.output_file = interpolate_only(
                    input_string=self._original_output_file, inputs=inputs
                )
            except (KeyError, ValueError) as e:
                raise ValueError(f"Error interpolating output_file path: {e!s}") from e

        if inputs.get("chat_messages"):
            conversation_instruction = self.prompt.slice(
                "conversation_history_instruction"
            )

            chat_messages_json = str(inputs["chat_messages"])

            try:
                chat_messages = json.loads(chat_messages_json)
            except json.JSONDecodeError as e:
                _printer.print(
                    f"An error occurred while parsing crew chat messages: {e}",
                    color="red",
                )
                raise

            conversation_history = "\n".join(
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in chat_messages
                if isinstance(msg, dict) and "role" in msg and "content" in msg
            )

            self.description += (
                f"\n\n{conversation_instruction}\n\n{conversation_history}"
            )