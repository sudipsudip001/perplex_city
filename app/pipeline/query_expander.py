import os

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
        self.system_prompt = """
            Role: You are an expert search query optimizer.
            Task: Given a query, generate three expanded queries that together maximize the search coverage.
            Constraints:
                - Each question must approach the topic from different angles like why, what, when, where, how.
                - Stricly return the answer in the format mentioned.
                - Do NOT drift from the original topic.
                - Prefer specific, searchable language over vague or academic phrasing.
                - Respond only with the output. No explanation.
            Output:
                Return the output in JSON format.
                Example:
                {{
                    "question1": "first question",
                    "question2": "second question",
                    "question3": "third question",
                }}
        """
        self.model = model
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def expanded_queries(self, query: str) -> list[str]:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=QueryExpansion,
                ),
                contents=[query],
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
