import os
import re

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
from models.response import GeneratedResponse
from pydantic import BaseModel

load_dotenv()


class Context(BaseModel):
    title: str
    url: str
    text: str


class Generator:
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the provided context.

    Rules you must follow:
    - Cite sources inline using [1], [2], etc. after every claim.
    - Every source listed in References MUST appear at least once inline. If a source was not used, do NOT include it in References.
    - Spread citations across sources — do not over-rely on a single source.
    - Synthesize information in your own words. Do not copy sentences from the context.
    - If the context does not contain enough information to answer, say so clearly.

    Format your answer as:
    <answer with inline citations>

    References:
    [1] <title> - <url>
    [2] <title> - <url>
    ..."""
    USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer with inline citations:
"""

    def __init__(self) -> None:
        self.chat_prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessagePromptTemplate.from_template(self.USER_TEMPLATE),
            ]
        )
        self._chat_model = None

    @property
    def chat_model(self) -> ChatGroq:
        if self._chat_model is None:
            self._chat_model = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                api_key=os.getenv("GROQ_API_KEY"),
            )
        return self._chat_model

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

    def _parse_citations(
        self, response: str, context_list: list[Context]
    ) -> GeneratedResponse:
        cited_indices: set[int] = {
            int(i)
            for i in re.findall(r"\d+", " ".join(re.findall(r"\[[\d,\s]+\]", response)))
        }

        citations = {
            i: context_list[i - 1] for i in cited_indices if 1 <= i <= len(context_list)
        }
        return citations

    def generate_answer(
        self, question: str, context_list: list[Context]
    ) -> GeneratedResponse:
        formatted_context = self._format_contexts(context_list)
        chain = self.chat_prompt_template | self.chat_model | StrOutputParser()

        response = chain.invoke({"context": formatted_context, "question": question})
        citations = self._parse_citations(response, context_list)

        return {
            "answer": response,
            "citations": {
                i: {"title": ctx.title, "url": ctx.url} for i, ctx in citations.items()
            },
        }
