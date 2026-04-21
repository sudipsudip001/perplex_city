from pydantic import BaseModel


class Context(BaseModel):
    title: str
    url: str
    text: str


class UserResponse(BaseModel):
    contexts: list[Context]
