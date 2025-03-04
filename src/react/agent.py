from enum import Enum, auto
from typing import Union, Callable
from pydantic import BaseModel, Field


class Name(Enum):
    WIKIPEDIA = auto()
    GOOGLE = auto()
    NONE = auto()

    def __str__(self) -> str:
        return self.name.lower()


Observation = Union[str, Exception]


class Message(BaseModel):
    role: str = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message")


class Choice(BaseModel):
    name: str = Field(..., description="The name of the tool chosen")
    reason: str = Field(..., description="The reason for choosing this tool")


class Tool:
    def __init__(self, name: Name, func: Callable[[str], str]):
        self.name = Name
        self.func = func

    def use(self, query: str) -> Observation:
        try:
            return self.func(query)
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {e}")
            return str(e)