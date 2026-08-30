import logging
from typing import Any, Dict, List, Tuple
from langchain_core.documents import Document
from config.settings import RETRIEVAL_TOP_K
from src.indexing.vector_store import get_vector_store
from src.state.state import GraphState
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


# Gecici ag veya veritabani gecikmelerine karsi 3 denemeli retry fonksiyonu
@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=0.2, max=2.0),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _fetch_documents_with_retry(query: str, top_k: int) -> List[Tuple[Document, float]]:
    vector_store = get_vector_store()
    return await vector_store.asimilarity_search_with_score(query=query, k=top_k)


async def retrieve_node(state: GraphState) -> Dict[str, Any]:
    # Guncel arama sorgusunu alip vektor tabanindan aday havuzunu toplar
    query: str = state["question"]
    logger.info(
        f"Executing async vector retrieval for query: '{query}' with Top-K: {RETRIEVAL_TOP_K}"
    )

    # 3 basarisiz denemenin ardindan hata yakalanmaz, fail-fast yaklasimiyla firlatilir
    results_with_scores = await _fetch_documents_with_retry(
        query=query, top_k=RETRIEVAL_TOP_K
    )

    documents: List[Document] = []
    for doc, score in results_with_scores:
        doc.metadata["retrieval_score"] = float(score)
        documents.append(doc)

    logger.info(f"Retrieved {len(documents)} candidate documents.")
    for idx, doc in enumerate(documents[:5]):
        content_preview = doc.page_content[:250].replace('\n', ' ')
        score = doc.metadata.get("retrieval_score", 0.0)
        logger.info("Retrieved Doc [%d] - Score: %.4f - Content: %s...", idx, score, content_preview)

    log_entry: Dict[str, Any] = {
        "node": "retrieve_node",
        "query_used": query,
        "retrieved_doc_count": len(documents),
        "success": True,
    }

    return {
        "documents": documents,
        "audit_logs": [log_entry],
    }