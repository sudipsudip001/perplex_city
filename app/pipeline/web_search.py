# from asyncddgs import aDDGS
# import random
import asyncio
import logging
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()

load_dotenv()
logger = logging.getLogger(__name__)

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_URL = "https://google.serper.dev/search"


class WebSearch:
    def __init__(self) -> None:
        pass

    async def search_urls(
        self,
        queries_string: list[str],
    ) -> list[dict[str, Any]]:
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

        try:
            async with httpx.AsyncClient() as client:
                search_tasks = [
                    client.post(SERPER_URL, json={"q": q}, headers=headers)
                    for q in queries_string
                ]
                search_responses = await asyncio.gather(*search_tasks)

                all_work_data = []
                for resp in search_responses:
                    resp.raise_for_status()
                    results = resp.json().get("organic", [])[:1]
                    for r in results:
                        all_work_data.append(
                            {
                                "title": r.get("title", ""),
                                "link": r.get("link", ""),
                                "body": r.get("snippet", ""),
                            }
                        )
                    logger.debug("Serper returned %d results", len(results))

            return all_work_data

        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code, detail="Serper API error"
            ) from exc
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

        # try:
        #     all_work_data = []

        #     async with aDDGS() as ducky_tool:
        #         for query in queries_string:
        #             work_data = await ducky_tool.text(query, max_results=1)
        #             all_work_data.extend(work_data)
        #             await asyncio.sleep(random.uniform(1.5, 3.0))

        #     mapping = {"href": "link"}
        #     renamed_work_data = [
        #         {mapping.get(k, k): v for k, v in d.items()} for d in all_work_data
        #     ]
        #     return renamed_work_data
        # except Exception as e:
        #     raise HTTPException(status_code=500, detail=str(e)) from e
