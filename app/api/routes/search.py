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

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(SERPER_URL, json=payload, headers=headers)
            response.raise_for_status()
            search_data = response.json()

            work_data = search_data.get("organic", [])[:5]
            context_data = []
            for data in work_data:
                title = data.get("title", [])
                url_link = data.get("link", [])

                page_res = await client.get(url_link)
                if page_res.status_code == 200:
                    clean_text = trafilatura.extract(page_res.text)

                    if clean_text:
                        context_data.append(
                            {"title": title, "url": url_link, "text": clean_text}
                        )
            return {"contexts": context_data}
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code, detail="Serper API error"
            ) from exc
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
