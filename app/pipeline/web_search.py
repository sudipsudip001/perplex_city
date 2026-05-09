import asyncio
import os
from typing import Any, cast
from urllib.parse import urlparse

import httpx
from ddgs import DDGS
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()

EMAIL = os.getenv("EMAIL")


class WebSearch:
    def __init__(self) -> None:
        pass

    async def fetch_url(self, url: str) -> str:
        parsed = urlparse(url)
        if "wikipedia.org" in parsed.netloc and parsed.path.startswith("/wiki/"):
            title = parsed.path.replace("/wiki/", "")
            return await self.fetch_wikipedia_api(title)
        else:
            return await self.fetch_generic(url)

    async def fetch_generic(self, url: str) -> str:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MyApp/1.0)"}
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            # If we ended up at Wikipedia after a redirect, use the API instead
            final = str(resp.url)
            if "wikipedia.org" in final and "/wiki/" in final:
                title = urlparse(final).path.replace("/wiki/", "")
                return await self.fetch_wikipedia_api(title)
            return cast(str, resp.text)

    async def fetch_wikipedia_api(self, title: str) -> str:
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": True,
            "exsectionformat": "plain",
            "titles": title,
            "format": "json",
            "redirects": 1,
        }
        headers = {"User-Agent": f"MyApp/1.0 ({EMAIL})"}
        async with httpx.AsyncClient() as client:
            resp = await client.get(api_url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        pages = data["query"]["pages"]
        page = next(iter(pages.values()))
        return cast(str, page.get("extract", ""))

    async def search_urls(
        self,
        queries_string: list[str],
    ) -> list[dict[str, Any]]:
        try:
            ddgs = DDGS()
            all_work_data = []

            for query in queries_string:
                results = ddgs.text(query, max_results=1)
                all_work_data.extend(results)

            # Rename 'href' -> 'link' to match the rest of your code
            all_work_data = [
                {("link" if k == "href" else k): v for k, v in d.items()}
                for d in all_work_data
            ]

            # Fetch actual page content for each result (same as before)
            fetch_tasks = [
                self.fetch_url(result["link"])
                for result in all_work_data
                if "link" in result
            ]
            fetched_contents = await asyncio.gather(
                *fetch_tasks, return_exceptions=True
            )

            for result, content in zip(all_work_data, fetched_contents, strict=False):
                result["content"] = None if isinstance(content, Exception) else content

            return all_work_data

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
