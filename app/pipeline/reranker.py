from langchain_core.documents import Document
from rerankers import Reranker


class Ranker:
    def __init__(
        self,
        reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.RERANKER_NAME = reranker_name
        self._reranker = None

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker(self.RERANKER_NAME)
        return self._reranker

    def rerank(
        self,
        query: str,
        doc_tests: list[Document],
        num_docs_final: int = 3,
    ) -> list[Document]:
        print("===> Reranking documents...")
        doc_texts = [doc.page_content for doc in doc_tests]
        rerank_results = self.reranker.rank(query, doc_texts)
        reranked_docs = []
        for res in rerank_results.results[:num_docs_final]:
            for doc in doc_tests:
                if doc.page_content == res.document.text:
                    reranked_docs.append(doc)
                    break
        relevant_docs = reranked_docs
        return relevant_docs
