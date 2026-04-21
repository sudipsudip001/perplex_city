import os

import httpx
import trafilatura
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from models.request import UserRequest
from models.response import UserResponse

load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_URL = "https://google.serper.dev/search"

router = APIRouter()


@router.post("/question")  # type: ignore[misc]
async def answer_question(query: UserRequest) -> UserResponse:
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"q": query.question}
    print("The API key is:", SERPER_API_KEY)

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(SERPER_URL, json=payload, headers=headers)
            response.raise_for_status()
            search_data = response.json()

            urls = [result["link"] for result in search_data.get("organic", [])[:5]]
            print("Extracted URLs from the site you mentioned.")

            contexts = []
            for url in urls:
                page_res = await client.get(url)
                if page_res.status_code == 200:
                    downloaded = page_res.text
                    content = trafilatura.extract(downloaded)
                    if content:
                        contexts.append(content)
            print("Extracted contents from .")
            return {"contexts": contexts}
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code, detail="Serper API error"
            ) from exc
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
