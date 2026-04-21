from api.routes import search
from fastapi import FastAPI

app: FastAPI = FastAPI()

app.include_router(search.router)


@app.get("/")  # type: ignore[misc]
def get_health() -> dict[str, str]:
    return {"status": "working"}
