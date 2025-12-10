import logging
from typing import Any, Final, Literal, cast

from pydantic import BaseModel, Field

from axon.agents.agent import Agent
from axon.llms import LLM
from axon.tasks.task import Task


class ReasoningPlan(BaseModel):
    """Representing a reasoning plan for a task."""

    plan: str = Field(description="The detailed reasoning plan for the task.")
    ready: bool = Field(description="Whether the agent is ready to execute the task.")


class AgentReasoningOutput(BaseModel):
    """Representing the output of the agent reasoning process."""

    plan: ReasoningPlan = Field(description="The reasoning plan for the task.")

class AgentReasoning:
    """
    Handles the agent reasoning process, enabling an agent to reflect and create a plan
    before executing a task.

    Attributes:
        task: The task for which the agent is reasoning.
        agent: The agent performing the reasoning.
        llm: The language model used for reasoning.
        logger: Logger for logging events and errors.
    """

    def __init__(self, task: Task, agent: Agent) -> None:
        """Initialize the AgentReasoning with a task and an agent.

        Args:
            task: The task for which the agent is reasoning.
            agent: The agent performing the reasoning.
        """
        self.task = task
        self.agent = agent
        self.llm = cast(LLM, agent.llm)
        self.logger = logging.getLogger(__name__)

    def handle_agent_reasoning(self) -> AgentReasoningOutput:
        """Public method for the reasoning process that creates
        and refines a plan for the task until the agent is ready 
        to execute it.

        Returns:
            AgentReasoningOutput: The output of the agent reasoning process.
        """
        try:
            plan, ready = self.__create_initial_plan()

            plan, ready = self.__refine_plan_if_needed(plan, ready)
            reasoning_plan = ReasoningPlan(plan=plan, ready=ready)
            return AgentReasoningOutput(plan=reasoning_plan)
        except Exception as e:                   
            logging.error(f"Error creating reasoning plan: {e}")
            raise

    def __create_initial_plan(self) -> tuple[str, bool]:
        """Creates the initial reasoning plan for the task.

        Returns:
            The initial plan and whether the agent is ready to execute the task.
        """
        reasoning_prompt = self.__create_reasoning_prompt()

        response = _call_llm_with_reasoning_prompt(
            llm=self.llm,
            prompt=reasoning_prompt,
            task=self.task,
            reasoning_agent=self.agent,
            backstory=self.__get_agent_backstory(),
            plan_type="initial_plan",
        )

        return self.__parse_reasoning_response(str(response))

    def __refine_plan_if_needed(self, plan: str, ready: bool) -> tuple[str, bool]:
        """Refines the reasoning plan if the agent is not ready to execute the task.

        Args:
            plan: The current reasoning plan.
            ready: Whether the agent is ready to execute the task.

        Returns:
            The refined plan and whether the agent is ready to execute the task.
        """
        attempt = 1
        max_attempts = getattr(self.agent,"max_reasoning_attempts",3)

        while not ready or attempt < max_attempts:
            refine_prompt = self.__create_refine_prompt(plan)

            response = _call_llm_with_reasoning_prompt(
                llm=self.llm,
                prompt=refine_prompt,
                task=self.task,
                reasoning_agent=self.agent,
                backstory=self.__get_agent_backstory(),
                plan_type="refine_plan",
            )
            plan, ready = self.__parse_reasoning_response(str(response))

            attempt += 1

            if max_attempts is not None and attempt >= max_attempts:
                self.logger.warning(
                    f"Agent reasoning reached maximum attempts ({max_attempts}) without being ready. Proceeding with current plan."
                )
                break

        return plan, ready
    

    def __get_agent_backstory(self) -> str:
        """
        Safely gets the agent's backstory, providing a default if not available.

        Returns:
            str: The agent's backstory or a default value.
        """
        return getattr(self.agent, "backstory", "No backstory provided")

    def __create_reasoning_prompt(self) -> str:
        """
        Creates a prompt for the agent to reason about the task.

        Returns:
            str: The reasoning prompt.
        """
        return self.agent.prompt.retrieve("reasoning", "create_plan_prompt").format(
            role=self.agent.role,
            goal=self.agent.goal,
            backstory=self.__get_agent_backstory(),
            description=self.task.description,
            expected_output=self.task.expected_output            
        )
    

    def __create_refine_prompt(self, current_plan: str) -> str:
        """
        Creates a prompt for the agent to refine its reasoning plan.

        Args:
            current_plan: The current reasoning plan.

        Returns:
            str: The refine prompt.
        """
        return self.agent.prompt.retrieve("reasoning", "refine_plan_prompt").format(
            role=self.agent.role,
            goal=self.agent.goal,
            backstory=self.__get_agent_backstory(),
            current_plan=current_plan,
        )

    @staticmethod
    def __parse_reasoning_response(response: str) -> tuple[str, bool]:
        """
        Parses the reasoning response to extract the plan and whether
        the agent is ready to execute the task.

        Args:
            response: The LLM response.

        Returns:
            The plan and whether the agent is ready to execute the task.
        """
        if not response:
            return "No plan was generated.", False

        plan = response
        ready = False

        if "READY: I am ready to execute the task." in response:
            ready = True

        return plan, ready


def _call_llm_with_reasoning_prompt(
    llm: LLM,
    prompt: str,
    task: Task,
    reasoning_agent: Agent,
    backstory: str,
    plan_type: Literal["initial_plan", "refine_plan"],
) -> str:
    """Calls the LLM with the reasoning prompt.

    Args:
        llm: The language model to use.
        prompt: The prompt to send to the LLM.
        task: The task for which the agent is reasoning.
        reasoning_agent: The agent performing the reasoning.
        backstory: The agent's backstory.
        plan_type: The type of plan being created ("initial_plan" or "refine_plan").

    Returns:
        The LLM response.
    """
    system_prompt = reasoning_agent.prompt.retrieve("reasoning", plan_type).format(
        role=reasoning_agent.role,
        goal=reasoning_agent.goal,
        backstory=backstory,
    )

    response = llm.call(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        from_task=task,
        from_agent=reasoning_agent,
    )
    return str(response)
