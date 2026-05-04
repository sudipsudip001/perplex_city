import asyncio
import logging
from typing import Any

import httpx
import trafilatura
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException

from app.models.request import UserRequest
from app.models.response import Context, GeneratedResponse
from app.pipeline.deduplicator import Deduplicator
from app.pipeline.generator import Generator
from app.pipeline.query_expander import QueryExpander
from app.pipeline.web_search import WebSearch

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

router = APIRouter()


@router.post("/question")  # type: ignore[misc]
async def answer_question(query: UserRequest) -> GeneratedResponse:
    try:
        # --- Query Expansion ---
        query_expander = QueryExpander(model="gemini-2.5-flash-lite")
        queries_string = query_expander.expanded_queries(query.question)

        logger.debug(
            "Type of queries_string: %s, value: %s",
            type(queries_string),
            queries_string,
        )
        logger.debug("Expanded queries: %s", queries_string)

        async with httpx.AsyncClient(
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
            follow_redirects=True,
        ) as client:
            # --- Web Search for all URLs ---
            searcher = WebSearch()
            all_work_data = await searcher.search_urls(
                queries_string=queries_string,
            )

            # --- Deduplication ---
            deduplicator = Deduplicator()
            list_urls = [
                str(link)
                for i in all_work_data
                if i and (link := i.get("link")) is not None
            ]
            urls_to_keep = deduplicator.deduplicate(list_urls)

            logger.debug("Raw search results: %s", all_work_data)
            logger.debug("URLs found: %s", list_urls)
            logger.debug("URLs after dedup: %s", urls_to_keep)

            # --- Page Fetching ---
            items_to_fetch = [d for d in all_work_data if d.get("link") in urls_to_keep]

            logger.debug("Items to fetch: %s", len(items_to_fetch))

            async def fetch_page(data: dict[str, Any]) -> dict[str, Any] | None:
                url_link = data.get("link")
                try:
                    page_res = await client.get(url_link, timeout=10)
                    if page_res.status_code == 200:
                        clean_text = trafilatura.extract(page_res.text)
                        if clean_text:
                            logger.debug(
                                "OK extracted text from %s (%d chars)",
                                url_link,
                                len(clean_text),
                            )
                            return {
                                "title": data.get("title", ""),
                                "url": url_link,
                                "text": clean_text,
                            }
                        else:
                            logger.warning(
                                "trafilatura returned no text for %s", url_link
                            )
                    else:
                        logger.warning(
                            "Non-200 status %d for %s", page_res.status_code, url_link
                        )
                except Exception as e:
                    logger.error("Failed to fetch %s: %s", url_link, e)

                snippet = data.get("body", "").strip()
                if snippet:
                    logger.debug("Using DDG snippet for %s", url_link)
                    return {
                        "title": data.get("title", ""),
                        "url": url_link,
                        "text": snippet,
                    }
                return None

            page_results = await asyncio.gather(
                *[fetch_page(d) for d in items_to_fetch]
            )
            context_data = [r for r in page_results if r is not None]

            logger.debug(
                "Page results: %s",
                [
                    (r.get("url"), len(r.get("text", ""))) if r else None
                    for r in page_results
                ],
            )
            logger.debug("Context data count: %d", len(context_data))

            if not context_data:
                raise HTTPException(
                    status_code=500, detail="No page content could be extracted"
                )

            context_list = [
                Context(
                    title=doc["title"],
                    url=doc["url"],
                    text=doc["text"],
                )
                for doc in context_data
            ]

            # GENERATE THE ANSWER AND RESPOND BACK
            generator = Generator()
            answer = generator.generate_answer(
                question=query.question,
                context_list=context_list,
            )

            logger.debug("Generated answer: %s", answer)

            return answer

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code, detail="Serper API error"
        ) from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
