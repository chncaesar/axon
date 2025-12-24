import json
from typing import Any, TYPE_CHECKING
from pydantic import Field, model_validator, PrivateAttr
from axon.agents.base_agent import BaseAgent
from axon.prompt.prompt import Prompt
from axon.tasks.task import Task
from axon.utilities.llm_utilities import create_llm
from axon.utilities.prompts import Prompts
from axon.utilities.converter import generate_model_description
from axon.agents.agent_executor import AgentExecutor
from axon.utilities.training_handler import TrainingHandler
from axon.utilities.constants import TRAINING_DATA_FILE


if TYPE_CHECKING:
    from axon.types import LLMMessage


class Agent(BaseAgent):
    orchestrator: Any = Field(default=None, description="The orchestrator to which the agent belongs.")

    reasoning: bool = Field(
        default=False,
        description="Whether the agent should reflect and create a plan before executing a task.",
    )
    max_reasoning_attempts: int | None = Field(
        default=3,
        description="Maximum number of reasoning attempts before executing the task. If None, will try until ready.",
    )
    date_format: str = Field(
        default="%Y-%m-%d",
        description="Format string for date when inject_date is enabled.",
    )
    respect_context_window: bool = Field(
        default=True,
        description="Keep messages under the context window size by summarizing content.",
    )
    use_system_prompt: bool | None = Field(
        default=True,
        description="Use system prompt for the agent.",
    )
    system_template: str | None = Field(
        default=None, description="System format for the agent."
    )
    prompt_template: str | None = Field(
        default=None, description="Prompt format for the agent."
    )
    response_template: str | None = Field(
        default=None, description="Response format for the agent."
    )
    _last_messages: list[LLMMessage] = PrivateAttr(default_factory=list)


    @model_validator(mode="after")
    def init(self):
        """ Initializes the Agent.        
        """
        self.llm = create_llm(self.llm)
        return self


    def execute_task(
        self,
        task: Task,
        context: str | None = None
    ) -> Any:
        """Execute a task with the agent.

        Args:
            task: Task to execute.
            context: Context to execute the task in.

        Returns:
            Output of the agent

        Raises:
            TimeoutError: If execution exceeds the maximum execution time.
            ValueError: If the max execution time is not a positive integer.
            RuntimeError: If the agent execution fails for other reasons.
        """
        if self.reasoning:
            try:
                from axon.utilities.reasoning_handler import (
                    AgentReasoning,
                    AgentReasoningOutput,
                )

                reasoning_handler = AgentReasoning(task=task, agent=self)
                reasoning_output: AgentReasoningOutput = (
                    reasoning_handler.handle_agent_reasoning()
                )

                # Add the reasoning plan to the task description
                task.description += f"\n\nReasoning Plan:\n{reasoning_output.plan.plan}"
            except Exception as e:
                self._logger.log("error", f"Error during reasoning process: {e!s}")
        self._inject_date_to_task(task)

        task_prompt = task.prompt()

        # If the task requires output in JSON or Pydantic format,
        # append specific instructions to the task prompt to ensure
        # that the final answer does not include any code block markers
        # Skip this if task.response_model is set, as native structured outputs handle schema automatically
        if (task.output_json or task.output_pydantic) and not task.response_model:
            # Generate the schema based on the output format
            if task.output_json:
                schema_dict = generate_model_description(task.output_json)
                schema = json.dumps(schema_dict["json_schema"]["schema"], indent=2)
                task_prompt += "\n" + self.prompt.slice(
                    "formatted_task_instructions"
                ).format(output_format=schema)

            elif task.output_pydantic:
                schema_dict = generate_model_description(task.output_pydantic)
                schema = json.dumps(schema_dict["json_schema"]["schema"], indent=2)
                task_prompt += "\n" + self.prompt.slice(
                    "formatted_task_instructions"
                ).format(output_format=schema)

        if context:
            task_prompt = self.prompt.slice("task_with_context").format(
                task=task_prompt, context=context
            )


        self.create_agent_executor(task=task)

        if self.orchestrator and self.orchestrator._train:
            task_prompt = self._training_handler(task_prompt=task_prompt)
        else:
            task_prompt = self._use_trained_data(task_prompt=task_prompt)

        result = self._execute_without_timeout(task_prompt, task)   
        

        self._last_messages = (
            self.agent_executor.messages.copy()
            if self.agent_executor and hasattr(self.agent_executor, "messages")
            else []
        )


        return result

    def create_agent_executor(
        self, 
        task: Task | None = None
    ) -> None:
        """Create an agent executor for the agent.

        Returns:
            An instance of the AgentExecutor class.
        """        

        prompt = Prompts(
            agent=self,
            prompt=self.prompt,
            use_system_prompt=self.use_system_prompt,
            system_template=self.system_template,
            prompt_template=self.prompt_template,
            response_template=self.response_template,
        ).task_execution()

        stop_words = [self.prompt.slice("observation")]

        if self.response_template:
            stop_words.append(
                self.response_template.split("{{ .Response }}")[1].strip()
            )

        self.agent_executor = AgentExecutor(
            llm=self.llm,
            task=task,  # type: ignore[arg-type]
            agent=self,
            orchestrator=self.orchestrator,            
            prompt=prompt,            
            stop_words=stop_words,
            max_iter=self.max_iter,
            respect_context_window=self.respect_context_window,
            response_model=task.response_model if task else None,
        )    

    def _training_handler(
        self,
        task_prompt: str
    ) -> str:
        """Handle training data for the agent task prompt to improve output on Training."""
        if data := TrainingHandler(TRAINING_DATA_FILE).load():
            agent_id = str(self.id)

            if data.get(agent_id): 
                human_feedbacks = [
                    i["human_feedback"] for i in data.get(agent_id).values()
                ]
            
                task_prompt += (
                    "\n\nYou MUST follow these instructions: \n "
                    + "\n - ".join(human_feedbacks)
                )
        
        return task_prompt


    def _use_trained_data(self, task_prompt: str) -> str:
        """Use trained data for the agent task prompt to improve output."""
        if data := TrainingHandler(TRAINING_DATA_FILE).load():
            if trained_data_output := data.get(self.role):
                task_prompt += (
                    "\n\nYou MUST follow these instructions: \n - "
                    + "\n - ".join(trained_data_output["suggestions"])
                )
        
        return task_prompt

    def _execute_without_timeout(self, task_prompt: str, task: Task) -> Any:
        """Execute a task without a timeout.

        Args:
            task_prompt: The prompt to send to the agent.
            task: The task being executed.

        Returns:
            The output of the agent.
        """
        if not self.agent_executor:
            raise RuntimeError("Agent executor is not initialized")

        return self.agent_executor.invoke(
            {
                "input": task_prompt,
                "ask_for_human_input": task.human_input,
            }
        )["output"]


    def _inject_date_to_task(self, task: Task) -> None:
        """Inject the current date into the task description."""
        from datetime import datetime

        try:
            valid_format_codes = [
                "%Y",
                "%m",
                "%d",
                "%H",
                "%M",
                "%S",
                "%B",
                "%b",
                "%A",
                "%a",
            ]
            is_valid = any(code in self.date_format for code in valid_format_codes)

            if not is_valid:
                raise ValueError(f"Invalid date format: {self.date_format}")

            current_date = datetime.now().strftime(self.date_format)
            task.description += f"\n\nCurrent Date: {current_date}"
        except Exception as e:
            self._logger.log("warning", f"Failed to inject date: {e!s}")