import asyncio
import logging
from typing import Any

import httpx
import trafilatura
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from rerankers import Reranker

from app.models.request import UserRequest
from app.models.response import Context, GeneratedResponse
from app.pipeline.chunker import Chunker
from app.pipeline.deduplicator import Deduplicator
from app.pipeline.generator import Generator
from app.pipeline.query_expander import QueryExpander
from app.pipeline.rank import Rank
from app.pipeline.simlar_match import SimilarMatch
from app.pipeline.web_search import WebSearch

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

router = APIRouter()
reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = Reranker(reranker_name)

chunker = Chunker(chunk_size=500, chunk_overlap=50)


@router.post("/question")  # type: ignore[misc]
async def answer_question(query: UserRequest) -> GeneratedResponse:
    try:
        # --- Query Expansion ---
        query_expander = QueryExpander(model="gemini-2.5-flash-lite")
        queries_string = query_expander.expanded_queries(query.question)

        async with httpx.AsyncClient() as client:
            # --- Web Search for all URLs ---
            searcher = WebSearch()
            all_work_data = await searcher.search_urls(
                queries_string=queries_string,
            )

            print(f"THE SEARCH WENT WELL AND RESULTED IN THIS: {all_work_data}")

            # --- Deduplication ---
            deduplicator = Deduplicator()
            list_urls = [
                str(link)
                for i in all_work_data
                if i and (link := i.get("link")) is not None
            ]
            urls_to_keep = deduplicator.deduplicate(list_urls)

            print(f"THE DEDUPLICATION WENT WELL AND HERE IS THE RESULT: {urls_to_keep}")

            # --- Page Fetching ---
            items_to_fetch = [d for d in all_work_data if d.get("link") in urls_to_keep]

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
                return None

            page_results = await asyncio.gather(
                *[fetch_page(d) for d in items_to_fetch]
            )
            context_data = [r for r in page_results if r is not None]

            if not context_data:
                raise HTTPException(
                    status_code=500, detail="No page content could be extracted"
                )

            print(f"ALL THE CONTEXT DATA HAS BEEN GENERATED {context_data}")

            context_list = [
                Context(
                    title=doc["title"],
                    url=doc["url"],
                    text=doc["text"],
                )
                for doc in context_data
            ]
            print(f"HERE'S THE FINAL PRODUCED CONTEXT_LIST: {context_list}")

            # ADD CHUNKING
            chunked_docs = chunker.chunk_documents(context_list)
            print(f"HERE'S THE FINAL PRODUCED CONTEXT_LIST: {chunked_docs}")

            # SIMLIARITY SEARCH USING BM25
            matcher = SimilarMatch(chunked_docs)
            final_results = matcher.match_similar_docs(query.question, 10)
            print(f"HERE'S THE FINAL RESULT AFTER MATCHING: {final_results}")

            # FOLLOWED BY RERANKING OF DOCUMENTS
            ranker = Rank()
            final_docs = ranker.reranked_docs(
                reranker, final_results, query.question, 3
            )
            print(f"HERE'S THE FINAL PRODUCED DOCS: {final_docs}")

            # GENERATE THE ANSWER AND RESPOND BACK
            generator = Generator()
            answer = generator.generate_answer(
                question=query.question,
                context_list=final_docs,
            )
            print(f"HERE'S THE FINAL PRODUCED ANSWER: {answer}")

            return answer

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code, detail="Serper API error"
        ) from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
