from rerankers import Reranker
from app.pipeline.chunker import Chunker
from app.pipeline.query_expander import QueryExpander
from app.pipeline.deduplicator import Deduplicator
from app.pipeline.generator import Generator
from app.pipeline.rank import Rank
from app.pipeline.web_search import WebSearch

query_expander = QueryExpander(model="gemini-2.5-flash-lite")
reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = Reranker(reranker_name, device="cpu")
chunker = Chunker(chunk_size=500, chunk_overlap=50)
deduplicator = Deduplicator()
generator = Generator()
ranker = Rank()
searcher = WebSearch()
