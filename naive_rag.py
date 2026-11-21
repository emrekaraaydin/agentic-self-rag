import os

os.environ["ORT_DISABLE_TELEMETRY"] = "1"

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from evaluate_batch import (
    check_fallback_status,
    check_target_doc_retrieved,
    extract_context_texts,
    load_dataset,
)
from src.chains.factory import get_base_llm
from src.indexing.vector_store import get_vector_store
from src.prompts.system_prompts import GENERATOR_SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("naive_rag")

NAIVE_TOP_K = 8
DATASET_PATH = Path("golden_dataset_100.json")
OUTPUT_PATH = Path("naive_rag_evaluation_results.json")


def _format_nomic_query(query: str) -> str:
    """Apply the search_query prefix required by nomic-embed-text."""
    cleaned_query = query.strip()
    if cleaned_query.startswith("search_query:"):
        return cleaned_query
    return f"search_query: {cleaned_query}"


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=0.2, max=2.0),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _fetch_documents_with_retry(
    query: str,
    top_k: int,
) -> List[Tuple[Document, float]]:
    """Query Qdrant vector store with retry on transient errors."""
    vector_store = get_vector_store()
    formatted_query = _format_nomic_query(query)
    return await vector_store.asimilarity_search_with_score(
        query=formatted_query,
        k=top_k,
    )


async def retrieve_documents(
    question: str,
    top_k: int = NAIVE_TOP_K,
) -> Tuple[List[Document], List[Dict[str, Any]]]:
    """Retrieve Top-K unique documents directly from Qdrant without reranking."""
    cleaned_question = question.strip()
    logger.info("Retrieving Top-%d documents from Qdrant for query: '%s'", top_k, cleaned_question)

    scored_docs = await _fetch_documents_with_retry(query=cleaned_question, top_k=top_k)
    scored_docs.sort(key=lambda item: float(item[1]), reverse=True)

    seen_ids: Set[str] = set()
    selected_documents: List[Document] = []

    for doc, score in scored_docs:
        chunk_id: str = (
            doc.metadata.get("chunk_id")
            or doc.metadata.get("id")
            or doc.page_content
        )
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            doc.metadata["retrieval_score"] = float(score)
            selected_documents.append(doc)
        if len(selected_documents) >= top_k:
            break

    sources: List[Dict[str, Any]] = [
        {
            "chunk_id": doc.metadata.get("chunk_id")
            or doc.metadata.get("id")
            or f"doc_{idx}",
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page"),
            "retrieval_score": doc.metadata.get("retrieval_score"),
        }
        for idx, doc in enumerate(selected_documents)
    ]

    logger.info("Retrieved %d unique documents from Qdrant.", len(selected_documents))
    return selected_documents, sources


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=0.5, max=3.0),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def generate_answer(question: str, documents: List[Document]) -> str:
    """Generate a response using the same GENERATOR_SYSTEM_PROMPT and LLM settings."""
    formatted_context = "\n\n---\n\n".join(
        f"Document [{idx + 1}]:\n{doc.page_content}"
        for idx, doc in enumerate(documents)
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GENERATOR_SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\n{retry_notes}"),
        ]
    )
    chain = prompt | get_base_llm() | StrOutputParser()

    return await chain.ainvoke(
        {
            "context": formatted_context,
            "question": question,
            "retry_notes": "",
        }
    )


async def run_naive_rag(question: str) -> Dict[str, Any]:
    """Execute the single-pass Naive RAG pipeline (retrieve k=8 -> generate)."""
    documents, sources = await retrieve_documents(question, top_k=NAIVE_TOP_K)
    generation = await generate_answer(question, documents)

    audit_logs: List[Dict[str, Any]] = [
        {
            "node": "naive_retrieve",
            "queries_used": [question.strip()],
            "retrieved_doc_count": len(documents),
            "top_k": NAIVE_TOP_K,
            "success": len(documents) > 0,
        },
        {
            "node": "naive_generate",
            "input_question": question,
            "context_doc_count": len(documents),
            "generated_char_length": len(generation),
            "generated_text": generation,
            "used_retry_notes": False,
            "success": True,
        },
    ]

    return {
        "question": question,
        "documents": documents,
        "sources": sources,
        "generation": generation,
        "audit_logs": audit_logs,
    }


def _print_evaluation_summary(metrics: Dict[str, Any]) -> None:
    """Print summary statistics matching evaluate_batch.py."""
    completed = metrics["completed"]
    avg_latency = (metrics["total_latency"] / completed) if completed > 0 else 0.0

    fb_expected = metrics["fallback_expected"]
    fb_tp = metrics["fallback_true_positive"]
    fb_fp = metrics["fallback_false_positive"]
    fb_recall = (fb_tp / fb_expected * 100) if fb_expected > 0 else 0.0

    doc_eligible = metrics["target_doc_eligible"]
    doc_hits = metrics["target_doc_hits"]
    hit_rate = (doc_hits / doc_eligible * 100) if doc_eligible > 0 else 0.0

    separator = "=" * 60
    print(f"\n{separator}")
    print("             NAIVE RAG BATCH EVALUATION SUMMARY")
    print(separator)
    print(f"Total Test Cases      : {metrics['total']}")
    print(f"Successful Runs       : {completed}")
    print(f"Failed Runs           : {metrics['failed_executions']}")
    print(f"Average Latency       : {avg_latency:.2f}s")
    print("-" * 60)
    print(f"Target Doc Hit Rate   : {hit_rate:.2f}% ({doc_hits}/{doc_eligible})")
    print(f"Fallback Recall       : {fb_recall:.2f}% ({fb_tp}/{fb_expected})")
    print(f"Fallback False Alerts : {fb_fp}")
    print(f"{separator}\n")


async def run_naive_batch_evaluation(
    dataset_path: Path = DATASET_PATH,
    output_path: Path = OUTPUT_PATH,
) -> None:
    """Run batch evaluation over golden_dataset_100.json using Naive RAG."""
    dataset = load_dataset(dataset_path)
    total_queries = len(dataset)
    logger.info("Loaded %d test items from %s", total_queries, dataset_path)

    evaluation_records: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {
        "total": total_queries,
        "completed": 0,
        "failed_executions": 0,
        "target_doc_hits": 0,
        "target_doc_eligible": 0,
        "fallback_expected": 0,
        "fallback_true_positive": 0,
        "fallback_false_positive": 0,
        "total_latency": 0.0,
    }

    for idx, item in enumerate(dataset, start=1):
        query_id = item.get("id", f"case_{idx}")
        category = item.get("category", "unknown")
        raw_question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        target_doc = item.get("target_doc", "")

        logger.info("[%d/%d] Running ID: %s | Category: %s", idx, total_queries, query_id, category)

        start_time = time.perf_counter()
        execution_error: Optional[str] = None
        result: Dict[str, Any] = {}

        try:
            result = await run_naive_rag(raw_question)
            latency_sec = round(time.perf_counter() - start_time, 3)
            metrics["completed"] += 1
            metrics["total_latency"] += latency_sec
        except Exception as err:
            latency_sec = round(time.perf_counter() - start_time, 3)
            execution_error = str(err)
            metrics["failed_executions"] += 1
            logger.error("Execution error on ID %s: %s", query_id, execution_error, exc_info=True)

        generated_answer = result.get("generation") or ""
        retrieved_documents = result.get("documents") or []
        retrieved_sources = result.get("sources") or []
        audit_logs = result.get("audit_logs") or []
        contexts = extract_context_texts(retrieved_documents)

        fallback_data = check_fallback_status(generated_answer, audit_logs)
        is_fallback_case = category == "fallback"

        if is_fallback_case:
            metrics["fallback_expected"] += 1
            if fallback_data["is_fallback"]:
                metrics["fallback_true_positive"] += 1
        elif fallback_data["is_fallback"]:
            metrics["fallback_false_positive"] += 1

        hit_target_doc = False
        if not is_fallback_case and target_doc:
            metrics["target_doc_eligible"] += 1
            hit_target_doc = check_target_doc_retrieved(
                target_doc=target_doc,
                sources=retrieved_sources,
                documents=retrieved_documents,
            )
            if hit_target_doc:
                metrics["target_doc_hits"] += 1

        record = {
            "id": query_id,
            "domain": item.get("domain", ""),
            "category": category,
            "subcategory": item.get("subcategory", ""),
            "literature_dimension": item.get("literature_dimension", ""),
            "original_question": raw_question,
            "transformed_question": raw_question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "contexts": contexts,
            "context_count": len(contexts),
            "sources": retrieved_sources,
            "target_doc": target_doc,
            "target_doc_hit": hit_target_doc if not is_fallback_case else None,
            "source_reference": item.get("source_reference", ""),
            "latency_sec": latency_sec,
            "query_rewrite_retry_count": 0,
            "hallucination_retry_count": 0,
            "fallback_status": fallback_data,
            "execution_success": execution_error is None,
            "execution_error": execution_error,
            "audit_logs": audit_logs,
        }
        evaluation_records.append(record)

        if idx % 5 == 0 or idx == total_queries:
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(evaluation_records, file, ensure_ascii=False, indent=2)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(evaluation_records, file, ensure_ascii=False, indent=2)

    logger.info("Batch execution finished. Exported %d records to %s", len(evaluation_records), output_path)
    _print_evaluation_summary(metrics)


async def _run_single_question(question: str) -> None:
    """Run a single query and print retrieved sources and answer."""
    result = await run_naive_rag(question)
    print(f"\nQuestion: {question}")
    print(f"\nRetrieved Sources ({len(result['sources'])}):")
    for idx, src in enumerate(result["sources"], start=1):
        print(
            f"  [{idx}] source={src['source']} | page={src['page']} | "
            f"score={src['retrieval_score']:.4f} | chunk_id={src['chunk_id']}"
        )
    print(f"\nAnswer:\n{result['generation']}\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Naive RAG pipeline & batch evaluation")
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Optional single question to run instead of full batch evaluation.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to the evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path to write the evaluation results JSON file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.question:
        asyncio.run(_run_single_question(args.question))
    else:
        asyncio.run(run_naive_batch_evaluation(dataset_path=args.dataset, output_path=args.output))
