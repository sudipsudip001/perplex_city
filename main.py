from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def get_health():
    return {"status": "working"}

