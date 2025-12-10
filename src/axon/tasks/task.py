from pydantic import BaseModel,Field

class Task(BaseModel):
    """"""
    description: str = Field(description="Description of the actual task.")
    expected_output: str = Field(
        description="Clear definition of expected output for the task."
    )