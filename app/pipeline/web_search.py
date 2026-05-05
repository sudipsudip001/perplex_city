import asyncio
import os
from typing import Any, cast
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()

EMAIL = os.getenv("EMAIL")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_URL = "https://google.serper.dev/search"


class WebSearch:
    def __init__(self) -> None:
        pass

    async def fetch_url(self, url: str) -> str:
        parsed = urlparse(url)
        if "wikipedia.org" in parsed.netloc:
            title = parsed.path.replace("/wiki/", "")
            return await self.fetch_wikipedia_api(title)
        else:
            return await self.fetch_generic(url)

    async def fetch_wikipedia_api(self, title: str) -> str:
        api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        headers = {"User-Agent": f"MyApp/1.0 ({EMAIL})"}
        async with httpx.AsyncClient() as client:
            resp = await client.get(api_url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return cast(str, data.get("extract", ""))

    async def fetch_generic(self, url: str) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return cast(str, resp.text)

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

                # collecting data from serper API
                all_work_data = []
                for _, resp in enumerate(search_responses):
                    resp.raise_for_status()
                    results = resp.json().get("organic", [])[:1]
                    all_work_data.extend(results)

                # fetch the actual page content for each result
                fetch_tasks = [
                    self.fetch_url(result["link"])
                    for result in all_work_data
                    if "link" in result
                ]
                fetched_contents = await asyncio.gather(
                    *fetch_tasks, return_exceptions=True
                )

                # attach the fetched content to each result
                for result, content in zip(
                    all_work_data, fetched_contents, strict=False
                ):
                    if isinstance(content, Exception):
                        result["content"] = None
                    else:
                        result["content"] = content

                return all_work_data

        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code, detail="Serper API error"
            ) from exc
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

        # try:
        #     ducky_tool = DDGS()
        #     all_work_data = []
        #     for _, query in enumerate(queries_string):
        #         work_data = ducky_tool.text(query, max_results=1)
        #         all_work_data.extend(work_data)
        #     mapping = {"href": "link"}
        #     renamed_work_data = [
        #         {mapping.get(k, k): v for k, v in d.items()} for d in all_work_data
        #     ]
        #     return renamed_work_data
        # except Exception as e:
        #     raise HTTPException(status_code=500, detail=str(e)) from e
