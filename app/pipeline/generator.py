import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from models.response import Context, Detail, GeneratedResponse
from pydantic import BaseModel

load_dotenv()


class Citation(BaseModel):
    title: str
    url: str


class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation]


class Generator:
    USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer with inline citations:
"""

    def __init__(self, model: str = "gemini-2.5-flash-lite") -> None:
        self.model = model
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.system_prompt = """You are a helpful assistant that    answers questions using only the provided context.

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
        for i, ctx in enumerate(context_list, start=1):
            blocks.append(
                f"[{i}] Title: {ctx.title}\n"
                f"    URL: {ctx.url}\n"
                f"    Content: {ctx.text}"
            )
        return "\n\n---\n\n".join(blocks)

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
            ),
            contents=[user_prompt],
        )

        print(repr(response))
        print(repr(response.text))

        raw = response.text.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(raw)

        return GeneratedResponse(
            answer=data["answer"],
            citations={
                str(i + 1): Detail(title=c["title"], url=c["url"])
                for i, c in enumerate(data["citations"])
            },
        )
