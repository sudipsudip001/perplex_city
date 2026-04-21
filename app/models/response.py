from pydantic import BaseModel


class UserResponse(BaseModel):
    answer: str
