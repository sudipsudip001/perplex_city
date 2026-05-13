from fastapi import FastAPI
from dotenv import load_dotenv
from app.api.routes import search

app: FastAPI = FastAPI()

app.include_router(search.router)

load_dotenv()

@app.get("/")  # type: ignore[misc]
def get_health() -> dict[str, str]:
    return {"status": "working"}
