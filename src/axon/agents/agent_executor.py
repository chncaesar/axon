from typing import Any, cast
from pydantic import BaseModel

from axon.agents.base_agent import BaseLLM
from axon.tasks.task import Task
from axon.orchestrator import Orchestrator
from axon.agents.agent import Agent
from axon.utilities.prompts import StandardPromptResult, SystemPromptResult
from axon.utilities.agent_utilities import format_message_for_llm, handle_unknown_error
from axon.prompt.prompt import Prompt, get_prompt
from axon.types import LLMMessage
from axon.agents.parser import AgentFinish

class AgentExecutor():
    """Executor for agents.

    Manages the execution lifecycle of an agent including prompt formatting,
    LLM interactions, and feedback handling.
    """

    def __init__(
        self,
        llm: BaseLLM,
        task: Task,
        orchestrator: Orchestrator,
        agent: Agent,
        prompt: SystemPromptResult | StandardPromptResult,
        max_iter: int,
        stop_words: list[str],
        response_model: type[BaseModel] | None = None
    ) -> None:
        """Initialize executor.

            Args:
                llm: Language model instance.
                task: Task to execute.
                orchestrator: Orchestrator instance.
                agent: Agent to execute.
                prompt: Prompt templates.
                max_iter: Maximum iterations.
                stop_words: Stop word list.
                response_model: Optional Pydantic model for structured outputs.
        """
        self._prompt: Prompt = get_prompt()
        self.llm = llm
        self.task = task
        self.orchestrator = orchestrator
        self.prompt = prompt
        self.max_iter = max_iter
        self.stop_words = stop_words
        self.response_model = response_model
        self.messages: list[LLMMessage] = []
        self.iterations = 0
        self.log_error_after = 3

        if self.llm:
            existing_stop = getattr(self.llm, "stop", [])
            self.llm.stop = list[str](   
                set[str](
                    existing_stop + self.stop
                    if isinstance(existing_stop, list)
                    else self.stop
                )
            )

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent with given inputs.
        Args:
            inputs: Input dictionary containing prompt variables.

        Returns:
            Dictionary with agent output.
        """
        if "system" in self.prompt:
            system_prompt = self._format_promt(
                cast(str, self.prompt.get("system","")), inputs
            )
            user_prompt = self._format_prompt(
                cast(str, self.prompt("user","")), inputs
            )
            self.messages.append(format_message_for_llm(system_prompt,role="system"))
            self.messages.append(format_message_for_llm(user_prompt))
        else:
            user_prompt = self._format_prompt(
                self.prompt.get("prompt",""), inputs
            )
            self.messages.append(format_message_for_llm(user_prompt))

        try:
            formatted_answer = self._invoke_loop()
        except AssertionError:
            self._printer.print(
                content="Agent failed to reach a final answer. This is likely a bug - please report it.",
                color="red",
            )
            raise
        except Exception as e:
            handle_unknown_error(self._printer, e)
            raise    
        
        
        formatted_answer = self._handle_human_feedback(formatted_answer)

        return {"output": formatted_answer.output}

    def _invoke_loop(self) -> AgentFinish:
        """Execute agent loop until completion.

        Returns:
            Final answer from the agent.
        """    

    @staticmethod
    def _format_promt(prompt: str, inputs: dict[str, Any]) -> str:
        """Format prompt with input values.

        Args:
            prompt: prompt template string.
            inputs: Values to substitute.
        Returns:
            Formatted prompt
        """
        prompt = prompt.replace("{input}", inputs["inputs"])
        prompt = prompt.replace("{tool_names}", inputs["tool_names"])
        return prompt.replace("{tool}", inputs["tool"])

    
    def _handle_human_feedback(self, formatted_answer: AgentFinish) -> AgentFinish:
        """Process human feedback.

        Args:
            formatted_answer: Initial agent result.

        Returns:
            Final answer after feedback.
        """
        human_feedback = self._ask_human_input(formatted_answer.output)

        if self._is_training_mode():
            return self._handle_training_feedback(formatted_answer, human_feedback)

        return self._handle_regular_feedback(formatted_answer, human_feedback)

    def _ask_human_input(self, final_answer: str) -> str:
        """Prompt human input with mode-appropriate messaging."""
        
        self._printer.print(
            content=f"\033[1m\033[95m ## Final Result:\033[00m \033[92m{final_answer}\033[00m"
        )

        # Training mode prompt (single iteration)
        if self.orchestrator and getattr(self.orchestrator, "_train", False):
            prompt = (
                "\n\n=====\n"
                "## TRAINING MODE: Provide feedback to improve the agent's performance.\n"
                "This will be used to train better versions of the agent.\n"
                "Please provide detailed feedback about the result quality and reasoning process.\n"
                "=====\n"
            )
        # Regular human-in-the-loop prompt (multiple iterations)
        else:
            prompt = (
                "\n\n=====\n"
                "## HUMAN FEEDBACK: Provide feedback on the Final Result and Agent's actions.\n"
                "Please follow these guidelines:\n"
                " - If you are happy with the result, simply hit Enter without typing anything.\n"
                " - Otherwise, provide specific improvement requests.\n"
                " - You can provide multiple rounds of feedback until satisfied.\n"
                "=====\n"
            )

        self._printer.print(content=prompt, color="bold_yellow")
        response = input()
        if response.strip() != "":
            self._printer.print(
                content="\nProcessing your feedback...", color="cyan"
            )
        return response
    