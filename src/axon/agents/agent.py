import json
from typing import Any
from pydantic import Field, model_validator
from axon.agents.base_agent import BaseAgent
from axon.prompt.prompt import Prompt
from axon.tools.base_tool import BaseTool
from axon.tasks.task import Task
from axon.utilities.llm_utilities import create_llm
from axon.utilities.prompts import Prompts
from axon.utilities.converter import generate_model_description
from axon.agents.agent_executor import AgentExecutor

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

        if self.tools_handler:
            self.tools_handler.last_used_tool = None

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
                task_prompt += "\n" + self.i18n.slice(
                    "formatted_task_instructions"
                ).format(output_format=schema)

            elif task.output_pydantic:
                schema_dict = generate_model_description(task.output_pydantic)
                schema = json.dumps(schema_dict["json_schema"]["schema"], indent=2)
                task_prompt += "\n" + self.i18n.slice(
                    "formatted_task_instructions"
                ).format(output_format=schema)

        if context:
            task_prompt = self.i18n.slice("task_with_context").format(
                task=task_prompt, context=context
            )

        if self._is_any_available_memory():
            

            start_time = time.time()

            contextual_memory = ContextualMemory(
                self.crew._short_term_memory,
                self.crew._long_term_memory,
                self.crew._entity_memory,
                self.crew._external_memory,
                agent=self,
                task=task,
            )
            memory = contextual_memory.build_context_for_task(task, context or "")
            if memory.strip() != "":
                task_prompt += self.i18n.slice("memory").format(memory=memory)

            
        knowledge_config = (
            self.knowledge_config.model_dump() if self.knowledge_config else {}
        )

        if self.knowledge or (self.crew and self.crew.knowledge):
            
            try:
                self.knowledge_search_query = self._get_knowledge_search_query(
                    task_prompt, task
                )
                if self.knowledge_search_query:
                    # Quering agent specific knowledge
                    if self.knowledge:
                        agent_knowledge_snippets = self.knowledge.query(
                            [self.knowledge_search_query], **knowledge_config
                        )
                        if agent_knowledge_snippets:
                            self.agent_knowledge_context = extract_knowledge_context(
                                agent_knowledge_snippets
                            )
                            if self.agent_knowledge_context:
                                task_prompt += self.agent_knowledge_context

                    # Quering crew specific knowledge
                    knowledge_snippets = self.crew.query_knowledge(
                        [self.knowledge_search_query], **knowledge_config
                    )
                    if knowledge_snippets:
                        self.crew_knowledge_context = extract_knowledge_context(
                            knowledge_snippets
                        )
                        if self.crew_knowledge_context:
                            task_prompt += self.crew_knowledge_context

                    
            except Exception as e:
                pass

        tools = tools or self.tools or []
        self.create_agent_executor(tools=tools, task=task)

        if self.crew and self.crew._train:
            task_prompt = self._training_handler(task_prompt=task_prompt)
        else:
            task_prompt = self._use_trained_data(task_prompt=task_prompt)

        # Import agent events locally to avoid circular imports
        

        try:
            

            # Determine execution method based on timeout setting
            if self.max_execution_time is not None:
                if (
                    not isinstance(self.max_execution_time, int)
                    or self.max_execution_time <= 0
                ):
                    raise ValueError(
                        "Max Execution time must be a positive integer greater than zero"
                    )
                result = self._execute_with_timeout(
                    task_prompt, task, self.max_execution_time
                )
            else:
                result = self._execute_without_timeout(task_prompt, task)

        except TimeoutError as e:
            # Propagate TimeoutError without retry
            
            raise e
        except Exception as e:
            if e.__class__.__module__.startswith("litellm"):
                # Do not retry on litellm errors
                
                raise e
            self._times_executed += 1
            if self._times_executed > self.max_retry_limit:
                
                raise e
            result = self.execute_task(task, context, tools)

        if self.max_rpm and self._rpm_controller:
            self._rpm_controller.stop_rpm_counter()

        # If there was any tool in self.tools_results that had result_as_answer
        # set to True, return the results of the last tool that had
        # result_as_answer set to True
        for tool_result in self.tools_results:
            if tool_result.get("result_as_answer", False):
                result = tool_result["result"]
       

        self._last_messages = (
            self.agent_executor.messages.copy()
            if self.agent_executor and hasattr(self.agent_executor, "messages")
            else []
        )

        self._cleanup_mcp_clients()

        return result

    def create_agent_executor(
        self, tools: list[BaseTool] | None = None, task: Task | None = None
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