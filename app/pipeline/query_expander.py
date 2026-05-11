import os
from datetime import date

from dotenv import load_dotenv
from fastapi import HTTPException
from google import genai
from google.api_core import exceptions
from google.genai import types
from pydantic import BaseModel

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class QueryExpansion(BaseModel):
    question1: str
    question2: str
    question3: str


class QueryExpander:
    load_dotenv()

    def __init__(self, model: str = "gemini-2.5-flash-lite") -> None:
        self._raw_prompt = """
            Role: You are an expert search query optimizer.
            Task: Given a user query, generate three expanded queries that together maximise the retrieval of all pieces of information needed to fully answer the question.
            Today's date is {current_date}.

            Guidelines for each query:
            - question1: A direct, noun-heavy rephrase that captures the exact who/what/where/when of the request. This should look like a typical search engine query that would surface a specific sentence or fact.
            - question2: A broader context query that explores background, definitions, or related structures, staying strictly within the topic.
            - question3: A complementary angle (e.g., recent updates, comparisons, responsibilities) that still directly pertains to the original subject.

            Constraints:
            - Do NOT drift from the original topic.
            - Use concise, keyword-rich, search-engine-friendly language. No full sentences or filler words.
            - If the original query implies recency (e.g., "current", "latest", "2026"), include appropriate date markers.
            - Output strictly a JSON object with keys "question1", "question2", "question3". No additional text.
        """
        self.model = model
        self.client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"timeout": 30000},
        )

    def expanded_queries(self, query: str) -> list[str]:
        prompt = self._raw_prompt.format(current_date=date.today().strftime("%Y-%m-%d"))
        final_input = f"{prompt}\n\nUser Query: {query}"
        print(final_input)
        try:
            response = self.client.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=prompt,
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=QueryExpansion,
                ),
                contents=final_input,
            )
            expanded = QueryExpansion.model_validate_json(response.text)
            return list(expanded.model_dump().values())
        except exceptions.ResourceExhausted as e:
            raise HTTPException(
                status_code=429,
                detail="Quota exceeded! Please check AI studio billing.",
            ) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
