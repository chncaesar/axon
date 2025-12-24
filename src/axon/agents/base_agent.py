
from pydantic import (
    BaseModel,
    Field,
    UUID4,
    PrivateAttr
)
import uuid
from typing import Any
from abc import ABC, abstractmethod
from axon.tools.base_tool import BaseTool
from axon.llms.llm import LLM
from axon.tasks.task import Task
from axon.prompt.prompt import Prompt, get_prompt
from axon.utilities.string_utils import interpolate_only
from axon.utilities.string_utils import interpolate_only
from axon.utilities.logger import Logger


class BaseAgent(BaseModel, ABC):
    uuid: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    name: str = Field()
    llm: LLM = Field()
    task: Task = Field()
    role: str = Field(description="Role of the agent")
    goal: str = Field(description="Objective of the agent")
    backstory: str = Field(description="Backstory of the agent")
    config: dict[str, Any] | None = Field(
        description="Configuration for the agent", default=None, exclude=True
    )
    prompt: Prompt = Field(
        default_factory=get_prompt, description="prompt settings."
    )
    _original_role: str | None = PrivateAttr(default=None)
    _original_goal: str | None = PrivateAttr(default=None)
    _original_backstory: str | None = PrivateAttr(default=None)
    _logger: Logger = PrivateAttr(default_factory=lambda: Logger(verbose=False))
    max_iter: int = Field(
        default=25, description="Maximum iterations for an agent to execute a task"
    )


    
    def interpolate_inputs(self, inputs: dict[str, Any]) -> None:
        """Interpolate inputs into the agent description and backstory."""
        if self._original_role is None:
            self._original_role = self.role
        if self._original_goal is None:
            self._original_goal = self.goal
        if self._original_backstory is None:
            self._original_backstory = self.backstory

        if inputs:
            self.role = interpolate_only(
                input_string=self._original_role, inputs=inputs
            )
            self.goal = interpolate_only(
                input_string=self._original_goal, inputs=inputs
            )
            self.backstory = interpolate_only(
                input_string=self._original_backstory, inputs=inputs
            )


    @abstractmethod
    def execute_task(
        self,
        task: Any,
        context: str | None = None,
    ) -> Any:
        pass

    @abstractmethod
    def create_agent_executor(self, task: Task | None = None ) -> None:
        pass

    def interpolate_inputs(self, inputs: dict[str, Any]) -> None:
        """Interpolate inputs into the agent description and backstory."""
        if self._original_role is None:
            self._original_role = self.role
        if self._original_goal is None:
            self._original_goal = self.goal
        if self._original_backstory is None:
            self._original_backstory = self.backstory

        if inputs:
            self.role = interpolate_only(
                input_string=self._original_role, inputs=inputs
            )
            self.goal = interpolate_only(
                input_string=self._original_goal, inputs=inputs
            )
            self.backstory = interpolate_only(
                input_string=self._original_backstory, inputs=inputs
            )
