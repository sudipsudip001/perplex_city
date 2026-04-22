from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


class Chunker:
    MARKDOWN_SEPARATORS = ["\n#{1,6} ", "```\n", "\n\n", "\n", " ", ""]

    def __init__(
        self,
        docs: Document,
        chunk_size: int = 400,
        chunk_overlap: int = 40,
        embedder_model: str = "thenlper/gte-small",
    ):
        self.docs = docs
        self.CHUNK_SIZE = chunk_size
        self.CHUNK_OVERLAP = chunk_overlap
        self.EMBEDDING_MODEL_NAME = embedder_model

    def chunk_document(self) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(self.EMBEDDING_MODEL_NAME),
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            add_start_index=True,
            strip_whitespace=True,
            separators=self.MARKDOWN_SEPARATORS,
        )
        seen, chunks = set(), []
        for doc in text_splitter.split_documents(self.docs):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                chunks.append(doc)
        return chunks
