
from pydantic import (
    BaseModel,
    Field,
    UUID4,
)
import uuid
from typing import Any
from abc import ABC, abstractmethod
from axon.tools.base_tool import BaseTool
from axon.llms.llm import LLM
from axon.tasks.task import Task
from axon.prompt.prompt import Prompt, get_prompt


class BaseAgent(BaseModel, ABC):
    uuid: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
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

    

    @abstractmethod
    def execute_task(
        self,
        task: Any,
        context: str | None = None,
        tools: list[BaseTool] | None = None,
    ) -> str:
        pass