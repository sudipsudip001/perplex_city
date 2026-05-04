import os

import pandas as pd
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


class Ragged:
    def __init__(
        self,
        dataset: list[dict[str, str]],
        embedder: str = "thenlper/gte-small",
    ) -> None:
        self.EMBEDDING_MODEL_NAME = embedder
        self._ragas_llm = None
        self._embedder: HuggingFaceEmbeddings | None = None
        self.dataset = Dataset.from_list(
            [
                {
                    "user_input": row["question"],
                    "retrieved_contexts": row["retrieved_contexts"],
                    "response": row["answer"],
                    "reference": row["reference_answer"],
                }
                for row in dataset
            ]
        )
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

    @property
    def ragas_llm(self) -> LangchainLLMWrapper:
        if self._ragas_llm is None:
            self._ragas_llm = LangchainLLMWrapper(
                ChatGroq(
                    # model="llama-3.3-70b-versatile",
                    model="openai/gpt-oss-120b",
                    temperature=0,
                    api_key=os.getenv("GROQ_API_KEY"),
                )
            )
        return self._ragas_llm

    @property
    def embedder(self) -> HuggingFaceEmbeddings:
        if self._embedder is None:
            self._embedder = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(model_name=self.EMBEDDING_MODEL_NAME)
            )
        return self._embedder

    def score(self) -> pd.DataFrame:
        # print("Dataset is: ", self.dataset)
        # print("Metrics is: ", self.metrics)
        # print("The LLM is: ", self.ragas_llm)
        # print("Embedder is: ", self.embedder)
        results = evaluate(
            self.dataset,
            metrics=self.metrics,
            llm=self.ragas_llm,
            embeddings=self.embedder,
        )

        return results.to_pandas()
