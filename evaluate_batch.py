import os

os.environ["ORT_DISABLE_TELEMETRY"] = "1"

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("batch_evaluator")

DATASET_PATH = Path("golden_dataset.json")
OUTPUT_PATH = Path("evaluation_results.json")

FALLBACK_MSG_NO_DOCS = (
    "I could not find any relevant documents in the knowledge base to answer your question."
)
FALLBACK_MSG_HALLUCINATION = (
    "A verified answer could not be generated from the available context "
    "due to factual consistency constraints."
)
VALID_FALLBACK_MESSAGES = {
    FALLBACK_MSG_NO_DOCS.strip(),
    FALLBACK_MSG_HALLUCINATION.strip(),
}


def load_dataset(file_path: Path) -> List[Dict[str, Any]]:
    # Veri seti dosyasının varlığını doğrula ve diskten oku
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_context_texts(documents: List[Any]) -> List[str]:
    # Gelen doküman yapısından ham metinleri ayıkla
    contexts: List[str] = []
    for doc in documents:
        if hasattr(doc, "page_content"):
            contexts.append(doc.page_content)
        elif isinstance(doc, dict) and "page_content" in doc:
            contexts.append(doc["page_content"])
        else:
            contexts.append(str(doc))
    return contexts


def check_fallback_status(
    answer: str,
    audit_logs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    # Fallback node tetiklenme durumunu ve üretilen metni doğrula
    node_triggered = False
    trigger_reason: Optional[str] = None
    retry_count = 0

    for log in audit_logs:
        if isinstance(log, dict) and log.get("node") == "fallback_node":
            node_triggered = True
            trigger_reason = log.get("reason")
            retry_count = log.get("retry_count", 0)
            break

    message_matched = answer.strip() in VALID_FALLBACK_MESSAGES
    is_fallback = node_triggered or message_matched

    return {
        "is_fallback": is_fallback,
        "node_triggered": node_triggered,
        "message_matched": message_matched,
        "reason": trigger_reason,
        "retry_count": retry_count,
    }


def check_target_doc_retrieved(
    target_doc: str,
    sources: List[Dict[str, Any]],
    documents: List[Any],
) -> bool:
    # Hedef dokümanın retrieval / rerank aşamasından geçip geçmediğini denetle
    if not target_doc:
        return False

    for source_info in sources:
        if isinstance(source_info, dict) and source_info.get("source") == target_doc:
            return True

    for doc in documents:
        metadata = getattr(doc, "metadata", {}) if not isinstance(doc, dict) else doc.get("metadata", {})
        if metadata.get("source") == target_doc:
            return True

    return False


async def run_batch_evaluation() -> None:
    # Grafiği derle ve veri setini yükle
    logger.info("Initializing graph application...")
    app = build_graph()
    dataset = load_dataset(DATASET_PATH)
    total_queries = len(dataset)
    logger.info("Loaded %d test items from %s", total_queries, DATASET_PATH)

    evaluation_records: List[Dict[str, Any]] = []

    # Metrik takip sayaçları
    metrics = {
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

        # GraphState şemasına birebir uyumlu başlangıç durumu
        initial_state = {
            "original_question": raw_question,
            "question": raw_question,
            "is_relevant": None,
            "documents": [],
            "generation": None,
            "has_hallucination": None,
            "retrieval_retry_count": 0,
            "generation_hallucination_retry_count": 0,
            "hallucination_reasoning": None,
            "audit_logs": [],
            "sources": [],
        }

        start_time = time.perf_counter()
        execution_error: Optional[str] = None
        final_state: Dict[str, Any] = {}

        try:
            final_state = await app.ainvoke(initial_state)
            latency_sec = round(time.perf_counter() - start_time, 3)
            metrics["completed"] += 1
            metrics["total_latency"] += latency_sec
        except Exception as err:
            latency_sec = round(time.perf_counter() - start_time, 3)
            execution_error = str(err)
            metrics["failed_executions"] += 1
            logger.error("Execution error on ID %s: %s", query_id, execution_error, exc_info=True)

        generated_answer = final_state.get("generation") or ""
        retrieved_documents = final_state.get("documents") or []
        retrieved_sources = final_state.get("sources") or []
        transformed_question = final_state.get("question", raw_question)
        audit_logs = final_state.get("audit_logs") or []
        contexts = extract_context_texts(retrieved_documents)

        # Fallback analizi
        fallback_data = check_fallback_status(generated_answer, audit_logs)
        is_fallback_case = category == "fallback"

        if is_fallback_case:
            metrics["fallback_expected"] += 1
            if fallback_data["is_fallback"]:
                metrics["fallback_true_positive"] += 1
        elif fallback_data["is_fallback"]:
            metrics["fallback_false_positive"] += 1

        # Retrieval isabet kontrolü
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

        # Ragas ve benchmark formatına uygun kayıt yapısı
        record = {
            "id": query_id,
            "domain": item.get("domain", ""),
            "category": category,
            "original_question": raw_question,
            "transformed_question": transformed_question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "contexts": contexts,
            "context_count": len(contexts),
            "sources": retrieved_sources,
            "target_doc": target_doc,
            "target_doc_hit": hit_target_doc if not is_fallback_case else None,
            "source_reference": item.get("source_reference", ""),
            "latency_sec": latency_sec,
            "retrieval_retry_count": final_state.get("retrieval_retry_count", 0),
            "hallucination_retry_count": final_state.get("generation_hallucination_retry_count", 0),
            "fallback_status": fallback_data,
            "execution_success": execution_error is None,
            "execution_error": execution_error,
            "audit_logs": audit_logs,
        }
        evaluation_records.append(record)

    # Sonuçları diske yaz
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(evaluation_records, file, ensure_ascii=False, indent=2)

    logger.info("Batch execution finished. Exported %d records to %s", len(evaluation_records), OUTPUT_PATH)

    # Konsol özet raporu
    _print_evaluation_summary(metrics)


def _print_evaluation_summary(metrics: Dict[str, Any]) -> None:
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
    print("                 BATCH EVALUATION SUMMARY")
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


if __name__ == "__main__":
    asyncio.run(run_batch_evaluation())