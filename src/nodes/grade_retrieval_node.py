import logging
from typing import Any, Dict, List
from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document
from config.settings import RERANK_SCORE_THRESHOLD, RERANK_TOP_N
from src.state.state import GraphState

logger = logging.getLogger(__name__)

# FlashRank modelini process basinda tek seferlik yukler (default: ms-marco-TinyBERT-L-2-v2 veya MiniLM)
ranker_instance = Ranker()


async def grade_retrieval_node(state: GraphState) -> Dict[str, Any]:
    # Aday dokumanlari Cross-Encoder ile puanlayip esik ustundeki en iyi dokumanlari secer
    query: str = state["question"]
    documents: List[Document] = state.get("documents", [])

    logger.info(
        f"Grading {len(documents)} documents with Cross-Encoder for query: '{query}'"
    )

    # Retrieval bos donmusse rewrite dongusunu tetiklemek icin is_relevant False donulur
    if not documents:
        logger.warning("No input documents to grade.")
        return {
            "documents": [],
            "is_relevant": False,
            "audit_logs": [
                {
                    "node": "grade_documents_node",
                    "graded_doc_count": 0,
                    "passed_doc_count": 0,
                    "is_relevant": False,
                }
            ],
        }

    # LangChain Document nesnelerini FlashRank passage formatina donusturme
    passages = [
        {"id": idx, "text": doc.page_content, "metadata": doc.metadata}
        for idx, doc in enumerate(documents)
    ]

    rerank_request = RerankRequest(query=query, passages=passages)
    rerank_results = ranker_instance.rerank(rerank_request)
    logger.info("--- Top 5 FlashRank Scores ---")
    for idx, item in enumerate(rerank_results[:5]):
        raw_score: float = float(item["score"])
        metadata: Dict[str, Any] = item.get("metadata", {})
        qdrant_score: float = float(metadata.get("retrieval_score", 0.0))
        content_preview: str = item["text"][:120].replace("\n", " ")
        logger.info(
            "Rank [%d] - Cross-Encoder Score: %.6f | Qdrant Cosine: %.4f | Content: %s...",
            idx,
            raw_score,
            qdrant_score,
            content_preview,
        )

    filtered_documents: List[Document] = []
    for item in rerank_results:
        score: float = float(item["score"])
        # Belirlenen esik uzerindeki dokumanlar kabul edilir
        if score >= RERANK_SCORE_THRESHOLD:
            doc_metadata = item.get("metadata", {})
            doc_metadata["rerank_score"] = score

            doc = Document(page_content=item["text"], metadata=doc_metadata)
            filtered_documents.append(doc)

    # En yuksek skorlu ilk N dokuman secilir
    selected_documents = filtered_documents[:RERANK_TOP_N]
    is_relevant: bool = len(selected_documents) > 0

    logger.info(
        f"Cross-Encoder grading completed. {len(selected_documents)}/{len(documents)} passed the threshold ({RERANK_SCORE_THRESHOLD})."
    )

    log_entry: Dict[str, Any] = {
        "node": "grade_documents_node",
        "graded_doc_count": len(documents),
        "passed_doc_count": len(selected_documents),
        "is_relevant": is_relevant,
        "top_scores": [
            doc.metadata.get("rerank_score") for doc in selected_documents
        ],
    }

    return {
        "documents": selected_documents,
        "is_relevant": is_relevant,
        "audit_logs": [log_entry],
    }