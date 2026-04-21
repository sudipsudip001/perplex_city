from pydantic import BaseModel


class UserResponse(BaseModel):
    contexts: str
