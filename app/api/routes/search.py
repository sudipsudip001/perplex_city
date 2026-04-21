from fastapi import APIRouter
from models.request import UserRequest
from models.response import UserResponse

router = APIRouter()


@router.post("/question")  # type: ignore[misc]
def answer_question(request: UserRequest) -> UserResponse:
    print("yes the answer was obtained")
    print(f"request: {request}")
    return UserResponse(answer="Your logic here")
