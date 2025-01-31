from pydantic import BaseModel
from typing import List, Literal, Optional

class Message(BaseModel):
    text: str
    isUser: bool

class Selection(BaseModel):
    type: Optional[Literal['location', 'character']] = None
    value: str = ''

class ChatRequest(BaseModel):
    messages: List[Message]
    lastSelection: Selection

class ChatResponse(BaseModel):
    response: str 