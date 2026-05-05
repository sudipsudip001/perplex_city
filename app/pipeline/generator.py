import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from app.models.response import Detail, GeneratedResponse

load_dotenv()


class Citation(BaseModel):
    title: str
    url: str


class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation]


class Generator:
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context.
Rules:
- Cite sources inline using [1], [2], etc. after every claim.
- Only include sources you actually cited inline.
- Synthesize information in your own words.
- citations must be ordered by citation number.
- If the context lacks enough information, say so in the answer field.
- If the answer isn't present in the context, set answer to: THE ANSWER COULDN'T BE FOUND IN THE CONTEXT.
"""

    def __init__(self, model: str = "gemini-2.5-flash-lite") -> None:
        self.model = model
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def _format_contexts(self, context_list: list[dict[str, str]]) -> str:
        """Format contexts into a numbered block for the prompt."""
        blocks = [
            f"[{i}] Title: {ctx['title']}\n"
            f"    URL: {ctx['url']}\n"
            f"    Content: {ctx['text']}"
            for i, ctx in enumerate(context_list, start=1)
        ]
        return "\n\n---\n\n".join(blocks)

    def generate_answer(
        self, question: str, context_list: list[dict[str, str]]
    ) -> GeneratedResponse:
        formatted_context = self._format_contexts(context_list)
        user_prompt = f"Context:\n{formatted_context}\n\nQuestion: {question}"

        try:
            response = self.client.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=self.SYSTEM_PROMPT,
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=RAGResponse,
                ),
                contents=[user_prompt],
            )
            rag: RAGResponse = response.parsed
        except Exception as e:
            raise RuntimeError(f"Generation or parsing failed: {e}") from e

        return GeneratedResponse(
            answer=rag.answer,
            citations={
                str(i + 1): Detail(title=c.title, url=c.url)
                for i, c in enumerate(rag.citations)
            },
        )
