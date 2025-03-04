from enum import Enum, auto
from typing import Union, Callable, List, Dict
from pydantic import BaseModel, Field
from openai import OpenAI

Observation = Union[str, Exception]

PROMPT_TEMPLATE_PATH = "./data/input/react.txt"
OUTPUT_TRACE_PATH = "./data/output/trace.txt"


class Name(Enum):
    WIKIPEDIA = auto()
    GOOGLE = auto()
    NONE = auto()

    def __str__(self) -> str:
        return self.name.lower()


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


class Agent:
    def __init__(self, model: OpenAI):
        self.model = model
        self.tools: Dict[Name, Tool] = {}
        self.messages: List[Message] = []
        self.query = ""
        self.max_iteration = 5
        self.current_iteration = 0
        self.template = self.load_template()

    def load_template(self) -> str:
        return read_path(PROMPT_TEMPLATE_PATH)

    def register(self, name: Name, func: Callable[[str], str]):
        self.tools[name] = Tool(name, func)

    def trace(self, role: str, content: str):
        """
        Logs the message with the specified role and content and writes to file.

        Args:
            role (str): The role of the message sender.
            content (str): The content of the message.
        """
        if role != "system":
            self.messages.append(Message(role=role, content=content))
        write_to_file(path=OUTPUT_TRACE_PATH,content=f"{role}:{content}\n")

    def get_history(self)->str:
        return "\n".join([f"{message.role}: {message.content}" for message in self.messages])

    def think(self) -> None: ...

    def decide(self, response: str) -> None: ...

    def act(self, tool_name: Name, query: str) -> None: ...

    def execute(self, query: str) -> str: ...

    def ask_model(self, prompt: str) -> str: ...
