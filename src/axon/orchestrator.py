from pydantic import (
    UUID4,
    BaseModel,
    Field,
    InstanceOf,
    Json,
    PrivateAttr,
    field_validator,
    model_validator,
)
from typing import (
    Any,
    cast,
)
import uuid

from axon.utilities.logger import Logger

from axon.agents.base_agent import BaseAgent
from axon.tasks.task import Task
from axon.llms.base_llm import BaseLLM
from axon.utilities.file_handler import FileHandler
from axon.utilities.task_output_storage_handler import TaskOutputStorageHandler
from axon.utilities.printer import PrinterColor
from axon.axon_output import AxonOutput
from axon.process import Process

class Orchestrator(BaseModel):
    __hash__ = object.__hash__
    _logger: Logger = PrivateAttr()
    _file_handler: FileHandler = PrivateAttr()
    _inputs: dict[str, Any] | None = PrivateAttr(default=None)
    _logging_color: PrinterColor = PrivateAttr(
        default="bold_purple",
    )
    output_log_file: bool | str | None = Field(
        default=None,
        description="Path to the log file to be saved",
    )
    name: str | None = Field(default="orchestrator")
    agents: list[BaseAgent] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)
    cache: bool = Field(default=True)
    memory: bool = Field(
        default=False,
        description="If axon should use memory to store memories of it's execution",
    )
    config: Json[dict[str, Any]] | dict[str, Any] | None = Field(default=None)
    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    prompt_file: str | None = Field(
        default=None,
        description="Path to the prompt json file to be used for the crew.",
    )
    planning: bool | None = Field(
        default=True,
        description="Plan the execution and add the plan.",
    )
    planning_llm: str | InstanceOf[BaseLLM] | Any | None = Field(
        default=None,
        description=(
            "Language model that will run the AgentPlanner if planning is True."
        ),
    )
    _task_output_handler: TaskOutputStorageHandler = PrivateAttr(
        default_factory=TaskOutputStorageHandler
    )
    _train: bool | None = PrivateAttr(default=False)
    process: Process = Field(default=Process.sequential)

    def kickoff(
        self,
        inputs: dict[str, Any] | None = None,
    ) -> AxonOutput:        
        try:            
            self._task_output_handler.reset()
            self._logging_color = "bold_purple"

            if inputs is not None:
                self._inputs = inputs
                self._interpolate_inputs(inputs)
            self._set_tasks_callbacks()
            self._set_allow_trigger_context_for_first_task()

            for agent in self.agents:
                agent.orchestrator = self
                agent.create_agent_executor()

            if self.planning:
                self._handle_crew_planning()

            if self.process == Process.sequential:
                result = self._run_sequential_process()
            elif self.process == Process.hierarchical:
                result = self._run_hierarchical_process()
            else:
                raise NotImplementedError(
                    f"The process '{self.process}' is not implemented yet."
                )

            for after_callback in self.after_kickoff_callbacks:
                result = after_callback(result)

            self.usage_metrics = self.calculate_usage_metrics()

            return result
        except Exception as e:
            raise
        finally:
            detach(token)

    def _interpolate_inputs(self, inputs: dict[str, Any] ) -> None:
        """Interpolates the inputs in the tasks and agents"""
        for agent in self.agents:
            agent.interpolate_inputs(inputs)
        for task in self.tasks:
            task.interpolate_inputs_and_add_conversation_history(inputs)

    def _set_tasks_callbacks(self) -> None:
        """Sets callback for every task suing task_callback"""
        for task in self.tasks:
            if not task.callback:
                task.callback = self.task_callback

    def _set_allow_trigger_context_for_first_task(self) -> None:
        trigger_payload = self._inputs and self._inputs.get(
            "trigger_payload"
        )
        able_to_inject = (
            self.tasks and self.tasks[0].allow_trigger_context is None
        )

        if (
            self.process == Process.sequential
            and trigger_payload
            and able_to_inject
        ):
            self.tasks[0].allow_trigger_context = True