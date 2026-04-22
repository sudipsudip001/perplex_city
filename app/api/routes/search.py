import os

import httpx
import trafilatura
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from langchain_core.documents import Document
from models.request import UserRequest
from models.response import UserResponse
from pipeline.chunker import Chunker
from pipeline.embedder import Embedder

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
            # CONVERSION TO DOCUMENT FORMAT
            final_contexts = []
            for item in context_data:
                doc = Document(
                    page_content=item["text"],
                    metadata={
                        "title": item["title"],
                        "url": item["url"],
                    },
                )
                final_contexts.append(doc)

            # RECURSIVE CHUNKING
            chunker = Chunker(
                docs=final_contexts,
                chunk_size=400,
                chunk_overlap=40,
                embedder_model="thenlper/gte-small",
            )
            chunk_data = chunker.chunk_document()

            # CREATING VECTOR DATABASE
            embedding_model = Embedder(
                chunks=chunk_data,
                embedder_model="thenlper/gte-small",
            )
            vector_db = embedding_model.vector_database()

            # RETRIEVING THE MOST SIMILAR DOCUMENTS
            print("===> Retrieving initial documents...")
            loaded_docs = vector_db.similarity_search(query=query.question, k=3)
            return loaded_docs
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code, detail="Serper API error"
            ) from exc
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
