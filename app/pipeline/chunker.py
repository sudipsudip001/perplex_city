import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.response import Context
from transformers import AutoTokenizer


class Chunker:
    MARKDOWN_SEPARATORS = ["\n#{1,6} ", "```\n", "\n\n", "\n", " ", ""]

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        embedder_model: str = "thenlper/gte-small",
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.EMBEDDING_MODEL_NAME = embedder_model

    def chunk_documents(
        self,
        knowledge_base: list[Context],
    ) -> list[dict[str, str]]:
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(self.EMBEDDING_MODEL_NAME),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
            separators=self.MARKDOWN_SEPARATORS,
        )
        seen, chunks = set(), []
        for item in knowledge_base:
            texts = text_splitter.split_text(item.text)
            for t in texts:
                if t not in seen:
                    seen.add(t)
                    chunks.append({"title": item.title, "text": t, "url": item.url})
        return chunks
