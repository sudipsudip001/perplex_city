from pydantic import BaseModel


class Context(BaseModel):
    title: str
    url: str
    text: str


class Detail(BaseModel):
    title: str
    url: str


class GeneratedResponse(BaseModel):
    answer: str
    citations: dict[int, Detail]
    retrieved_contexts: list[str]
