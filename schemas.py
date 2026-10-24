from typing import Literal, Optional
from pydantic import BaseModel, Field

Action = Literal["answer", "ask_to_repeat", "defer", "stay_silent"]
Lang = Literal["ar", "en", "mixed"]


class TwinReply(BaseModel):
    addressed_to_me: bool
    action: Action
    language: Lang
    reply_text: str
    confidence: float = Field(ge=0, le=1)
    tools_used: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class Branch(BaseModel):
    """One candidate action in the decision tree."""
    action: Action
    kind: Literal["check", "llm", "check+llm"]
    threshold: float
    score: Optional[float] = None
    status: Literal["pending", "kept", "pruned", "selected"] = "pending"
    reason: str = ""
