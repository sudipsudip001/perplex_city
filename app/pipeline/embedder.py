import torch
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class Embedder:
    def __init__(
        self,
        chunks: list[Document],
        embedder_model: str = "thenlper/gte-small",
    ):
        self.EMBEDDING_MODEL_NAME = embedder_model
        self.chunkz = chunks
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._vector_database: FAISS | None = None

    def vector_database(self) -> FAISS:
        if self._vector_database is None:
            embedder = HuggingFaceEmbeddings(
                model_name=self.EMBEDDING_MODEL_NAME,
                multi_process=False,
                model_kwargs={"device": self._device},
                encode_kwargs={"normalize_embeddings": True},
            )
            self._vector_database = FAISS.from_documents(
                self.chunkz,
                embedder,
                distance_strategy=DistanceStrategy.COSINE,
            )
        return self._vector_database
