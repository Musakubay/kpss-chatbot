from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field

class ChatContentText(BaseModel):
    type: str = "text"
    text: str

class ChatContentImage(BaseModel):
    type: str = "image_url"
    image_url: Dict[str, str]  # {"url": "https://..."}

ChatContent = Union[ChatContentText, ChatContentImage]

class ChatMessage(BaseModel):
    role: str  # 'system' | 'user' | 'assistant'
    content: List[ChatContent]

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    sessionId: Optional[str] = Field(default=None)
    meta: Optional[Dict[str, Any]] = Field(default=None)
    temperature: Optional[float] = None
    stream: Optional[bool] = None