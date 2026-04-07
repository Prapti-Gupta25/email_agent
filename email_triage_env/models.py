from pydantic import BaseModel
from typing import List, Literal, Optional

# Email Object
class Email(BaseModel):
    id: int
    subject: str
    body: str
    sender: str

# Observation (State)
class EmailTriageObservation(BaseModel):
    emails: List[Email]
    step_count: int = 0
    reward: float = 0.0
    done: bool = False
    info: Optional[dict] = None

# Action
class EmailTriageAction(BaseModel):
    type: Literal["classify", "prioritize", "triage"]

    email_id: Optional[int] = None
    label: Optional[Literal["spam","important","normal"]] = None

    order: Optional[List[int]] = None

    action: Optional[Literal["reply", "archive", "escalate"]] = None