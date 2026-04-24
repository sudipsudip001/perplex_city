# import os
# import httpx
# from dotenv import load_dotenv

from typing import Any

from ddgs import DDGS
from fastapi import HTTPException

# import asyncio

# load_dotenv()

# SERPER_API_KEY = os.getenv("SERPER_API_KEY")
# SERPER_URL = "https://google.serper.dev/search"


class WebSearch:
    def __init__(self) -> None:
        pass

    async def search_urls(
        self,
        queries_string: list[str],
    ) -> list[dict[str, Any]]:
        # headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

        # try:
        #     async with httpx.AsyncClient() as client:
        #         search_tasks = [
        #             client.post(SERPER_URL, json={"q": q}, headers=headers)
        #             for q in queries_string
        #         ]
        #         search_responses = await asyncio.gather(*search_tasks)

        #         # collecting data for serper API
        #         all_work_data = []
        #         for _, resp in enumerate(search_responses):
        #             resp.raise_for_status()
        #             results = resp.json().get("organic", [])[:1]
        #             all_work_data.extend(results)

        # except httpx.HTTPStatusError as exc:
        #     raise HTTPException(
        #         status_code=exc.response.status_code, detail="Serper API error"
        #     ) from exc
        # except Exception as e:
        #     raise HTTPException(status_code=500, detail=str(e)) from e

        try:
            ducky_tool = DDGS()
            all_work_data = []
            for _, query in enumerate(queries_string):
                work_data = ducky_tool.text(query, max_results=1)
                all_work_data.extend(work_data)
            mapping = {"href": "link"}
            renamed_work_data = [
                {mapping.get(k, k): v for k, v in d.items()} for d in all_work_data
            ]
            return renamed_work_data
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
