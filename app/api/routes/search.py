import asyncio
import logging
import os
from typing import Any

import httpx
import trafilatura
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from langchain_core.documents import Document
from models.request import UserRequest
from models.response import Context, GeneratedResponse
from pipeline.chunker import Chunker
from pipeline.deduplicator import Deduplicator
from pipeline.embedder import Embedder
from pipeline.generator import Generator
from pipeline.query_expander import QueryExpander
from pipeline.reranker import Ranker

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_URL = "https://google.serper.dev/search"

router = APIRouter()


@router.post("/question")  # type: ignore[misc]
async def answer_question(query: UserRequest) -> GeneratedResponse:
    try:
        # --- Query Expansion ---
        query_expander = QueryExpander(model="gemini-2.5-flash-lite")
        queries_string = query_expander.expanded_queries(query.question)

        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

        async with httpx.AsyncClient() as client:
            # --- Serper Search ---
            search_tasks = [
                client.post(SERPER_URL, json={"q": q}, headers=headers)
                for q in queries_string
            ]
            search_responses = await asyncio.gather(*search_tasks)

            # --- Collecting Results ---
            all_work_data = []
            for _, resp in enumerate(search_responses):
                resp.raise_for_status()
                results = resp.json().get("organic", [])[:5]
                all_work_data.extend(results)

            # --- Deduplication ---
            deduplicator = Deduplicator()
            list_urls = [i.get("link") for i in all_work_data]
            urls_to_keep = deduplicator.deduplicate(list_urls)

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

            # --- Chunking ---
            final_contexts = [
                Document(
                    page_content=item["text"],
                    metadata={"title": item["title"], "url": item["url"]},
                )
                for item in context_data
            ]
            chunker = Chunker(
                docs=final_contexts,
                chunk_size=400,
                chunk_overlap=40,
                embedder_model="thenlper/gte-small",
            )
            chunk_data = chunker.chunk_document()

            # --- Embedding + Vector DB ---
            embedding_model = Embedder(
                chunks=chunk_data, embedder_model="thenlper/gte-small"
            )
            vector_db = embedding_model.vector_database()

            # --- Retrieval ---
            loaded_docs = vector_db.similarity_search(query=query.question, k=3)

            # RERANKING THE DOCUMENTS
            ranker = Ranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
            reranked_docs = ranker.rerank(
                query.question,
                loaded_docs,
                num_docs_final=3,
            )

            context_list = [
                Context(
                    title=doc.metadata.get("title", "No title found"),
                    url=doc.metadata.get("url", "#"),
                    text=doc.page_content,
                )
                for doc in reranked_docs
            ]

            # GENERATE THE ANSWER AND RESPOND BACK
            generator = Generator()
            answer = generator.generate_answer(
                question=query.question,
                context_list=context_list,
            )

            return GeneratedResponse(**answer)

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code, detail="Serper API error"
        ) from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
