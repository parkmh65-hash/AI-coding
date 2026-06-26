from typing import TypedDict, List
from langchain_core.messages import AnyMessage


class State(TypedDict):
    messages: List[AnyMessage]
