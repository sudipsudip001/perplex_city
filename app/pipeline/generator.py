import json
import logging
import os
import re
from typing import Any, cast

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from app.models.response import Context, Detail, GeneratedResponse

# from models.response import Context, Detail, GeneratedResponse

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

MAX_CHARS_PER_DOC = 3000
MAX_TOTAL_CHARS = 8000


class Citation(BaseModel):
    title: str
    url: str


class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation]
    actual_text: str


class Generator:
    USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer with inline citations:
"""

    def __init__(self, model: str = "gemini-2.5-flash-lite") -> None:
        self.model = model
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options={"timeout": 30000},
        )
        self.system_prompt = """You are a helpful assistant that answers questions using only the provided context.

            The JSON must have exactly this structure:
            {
            "answer": "your answer with inline citations like [1], [2]",
            "citations": [
                {"title": "source title", "url": "source url"},
                {"title": "source title", "url": "source url"}
            ]
            }

            Rules:
            - Cite sources inline using [1], [2], etc. after every claim.
            - Only include sources you actually cited inline.
            - Synthesize information in your own words.
            - If the context lacks enough information, say so in the answer field.
            - citations is a list ordered by citation number.
            - You MUST respond with ONLY a valid JSON object — no markdown, no explanation, nothing else.
        """

    def _format_contexts(self, context_list: list[Context]) -> str:
        """Format contexts into a numbered block for the prompt."""
        blocks = []
        total = 0
        for i, ctx in enumerate(context_list, start=1):
            text = ctx.text[:MAX_CHARS_PER_DOC]  # trim each doc
            total += len(text)
            if total > MAX_TOTAL_CHARS:
                break  # stop adding more docs
            blocks.append(
                f"[{i}] Title: {ctx.title}\n"
                f"    URL: {ctx.url}\n"
                f"    Content: {text}"
            )
        return "\n\n---\n\n".join(blocks)

    def _parse_json_response(self, raw: str) -> dict[str, Any]:
        """Robustly extract JSON from LLM response."""
        cleaned = raw.strip()

        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            return cast(dict[str, Any], json.loads(cleaned))
        except json.JSONDecodeError as e:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return cast(dict[str, Any], json.loads(match.group()))
            raise ValueError(f"Could not parse JSON from response: {raw[:200]}") from e

    def generate_answer(
        self, question: str, context_list: list[Context]
    ) -> GeneratedResponse:
        formatted_context = self._format_contexts(context_list)
        user_prompt = f"Context:\n{formatted_context}\n\nQuestion: {question}"

        response = self.client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=RAGResponse,
            ),
            contents=[user_prompt],
        )

        logger.debug("Raw Gemini response: %s", response.text[:500])
        data = self._parse_json_response(response.text)

        return GeneratedResponse(
            answer=data["answer"],
            citations={
                str(i + 1): Detail(title=c["title"], url=c["url"])
                for i, c in enumerate(data["citations"])
            },
            retrieved_contexts=[ctx.text for ctx in context_list],
        )
