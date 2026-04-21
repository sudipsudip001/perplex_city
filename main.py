from fastapi import FastAPI

app: FastAPI = FastAPI()


@app.get("/")  # type: ignore[misc]
def get_health() -> dict[str, str]:
    return {"status": "working"}
