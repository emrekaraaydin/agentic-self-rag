"""Comparison report generator and 80-query RAGAS dataset exporter for Agentic vs. Naive RAG."""

import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

os.environ["ORT_DISABLE_TELEMETRY"] = "1"

from langchain_core.documents import Document

from src.nodes.grade_hallucination_node import _detect_clean_refusal, _evaluate_with_hhem


# --- Paths & Semantic Refusal Patterns ---
ROOT = Path(__file__).resolve().parent
AGENTIC_PATH = ROOT / "evaluation_results.json"
NAIVE_PATH = ROOT / "naive_rag_evaluation_results.json"
RAGAS_AGENTIC_OUT = ROOT / "evaluation_results_ragas_80.json"
RAGAS_NAIVE_OUT = ROOT / "naive_rag_evaluation_results_ragas_80.json"
REPORT_OUT = ROOT / "agentic_vs_naive_karsilastirma_raporu.md"

REFUSAL_EXTRA_PHRASES = (
    "knowledge base does not contain",
    "knowledge base lacks",
    "does not provide information",
    "does not contain information",
    "cannot answer",
    "no relevant documents",
    "not mentioned in the provided",
    "context does not mention",
    "context does not provide",
    "insufficient information",
    "not contain enough information",
    "not available in the provided",
)


# --- Metric & RAGAS Export Helpers ---
def is_semantic_refusal(text: str) -> bool:
    """Return True if an answer cleanly refuses or reports missing context."""
    if not text or not text.strip():
        return True
    if _detect_clean_refusal(text):
        return True
    low = text.strip().lower()
    return any(p in low for p in REFUSAL_EXTRA_PHRASES)


def export_ragas_80(records: List[Dict[str, Any]], out_path: Path) -> None:
    """Export the 80 in-scope evaluation rows in RAGAS input schema format."""
    ragas_rows = []
    for r in records:
        if r.get("category") == "fallback":
            continue
        ragas_rows.append(
            {
                "id": r["id"],
                "domain": r.get("domain", "legal_compliance"),
                "category": r["category"],
                "subcategory": r.get("subcategory", ""),
                "user_input": r["original_question"],
                "response": r["generated_answer"],
                "retrieved_contexts": r.get("contexts", []),
                "reference": r["ground_truth"],
                "latency_sec": r.get("latency_sec", 0.0),
            }
        )
    out_path.write_text(json.dumps(ragas_rows, ensure_ascii=False, indent=2), encoding="utf-8")


def target_doc_list(target_doc: Union[str, List[str]]) -> List[str]:
    """Normalize single or multi-document target_doc specifications into a list."""
    if not target_doc:
        return []
    if isinstance(target_doc, list):
        return target_doc
    return [target_doc]


def compute_context_purity(rec: Dict[str, Any]) -> float:
    """Compute fraction of retrieved source chunks belonging to the target document(s)."""
    targets = set(target_doc_list(rec.get("target_doc", "")))
    sources = rec.get("sources", [])
    if not targets or not sources:
        return 0.0
    hits = sum(1 for s in sources if isinstance(s, dict) and s.get("source") in targets)
    return hits / len(sources)


def compute_partial_hit(rec: Dict[str, Any]) -> bool:
    """Return True if at least one target document appears among retrieved sources."""
    targets = set(target_doc_list(rec.get("target_doc", "")))
    sources = {s.get("source") for s in rec.get("sources", []) if isinstance(s, dict) and s.get("source")}
    return bool(targets & sources)


def get_final_hhem_score(rec: Dict[str, Any]) -> float | None:
    """Extract the terminal verifier entailment score recorded in audit_logs."""
    scores = [
        log.get("hhem_score")
        for log in rec.get("audit_logs", [])
        if isinstance(log, dict) and log.get("node") == "grade_hallucination_node" and log.get("hhem_score") is not None
    ]
    return float(scores[-1]) if scores else None


def evaluate_naive_hhem(naive_records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Score non-refusal Naive RAG answers using the local NLI/HHEM evaluator."""
    scores: Dict[str, float] = {}
    for rec in naive_records:
        ans = rec.get("generated_answer") or ""
        contexts = rec.get("contexts") or []
        if not ans.strip() or not contexts or is_semantic_refusal(ans):
            continue
        docs = [Document(page_content=c) for c in contexts]
        _, best_score, _ = _evaluate_with_hhem(docs, ans, rec.get("original_question", ""))
        scores[rec["id"]] = float(best_score)
    return scores


def check_sycophancy_rejection(ans: str) -> bool:
    """Detect whether an answer explicitly refutes a false legal premise."""
    low = (ans or "").strip().lower()
    if not low:
        return False
    neg_markers = (
        "no,", "no.", "incorrect", "not correct", "false", "that is not",
        "actually,", "however,", "contrary to", "does not fix", "does not require",
        "eliminated", "explicitly excludes", "not applicable", "shall not impose",
        "prohibits", "except where", "does not apply", "not classified as",
        "cannot skip", "shall not contact", "forward-looking", "prospective",
        "only personal data of natural persons", "lower",
    )
    return any(m in low for m in neg_markers)


def latency_stats(vals: List[float]) -> Dict[str, float]:
    """Compute mean, median, min, max, and P95 latency statistics."""
    if not vals:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0}
    sorted_v = sorted(vals)
    p95_idx = min(len(sorted_v) - 1, int(0.95 * len(sorted_v)))
    return {
        "mean": statistics.mean(sorted_v),
        "median": statistics.median(sorted_v),
        "min": sorted_v[0],
        "max": sorted_v[-1],
        "p95": sorted_v[p95_idx],
    }


# --- System-Level Aggregation ---
def summarize_system(records: List[Dict[str, Any]], hhem_map: Dict[str, float] | None = None) -> Dict[str, Any]:
    """Aggregate benchmark metrics across literature categories, PDFs, and fallbacks."""
    in_scope = [r for r in records if r["category"] != "fallback"]
    fallback = [r for r in records if r["category"] == "fallback"]
    ood = [r for r in fallback if r.get("subcategory") == "out_of_domain"]
    id_unans = [r for r in fallback if r.get("subcategory") == "in_domain_unanswerable"]

    lat_all = latency_stats([r["latency_sec"] for r in records if r.get("execution_success")])
    lat_warm = latency_stats([r["latency_sec"] for r in records[1:] if r.get("execution_success")])

    hit_rate = sum(1 for r in in_scope if r.get("target_doc_hit")) / len(in_scope)
    purity_avg = statistics.mean(compute_context_purity(r) for r in in_scope)
    ctx_all = statistics.mean(r.get("context_count", 0) for r in records)
    ctx_in = statistics.mean(r.get("context_count", 0) for r in in_scope)

    fb_struct = sum(1 for r in fallback if r.get("fallback_status", {}).get("is_fallback"))
    fb_sem = sum(
        1
        for r in fallback
        if r.get("fallback_status", {}).get("is_fallback") or is_semantic_refusal(r.get("generated_answer", ""))
    )
    fb_leak = len(fallback) - fb_sem

    ood_struct = sum(1 for r in ood if r.get("fallback_status", {}).get("is_fallback"))
    ood_sem = sum(
        1
        for r in ood
        if r.get("fallback_status", {}).get("is_fallback") or is_semantic_refusal(r.get("generated_answer", ""))
    )
    id_struct = sum(1 for r in id_unans if r.get("fallback_status", {}).get("is_fallback"))
    id_sem = sum(
        1
        for r in id_unans
        if r.get("fallback_status", {}).get("is_fallback") or is_semantic_refusal(r.get("generated_answer", ""))
    )

    in_scope_fp_fb = sum(1 for r in in_scope if r.get("fallback_status", {}).get("is_fallback"))
    in_scope_refusals = sum(
        1
        for r in in_scope
        if r.get("fallback_status", {}).get("is_fallback") or is_semantic_refusal(r.get("generated_answer", ""))
    )

    # Category & subcategory breakdowns
    by_cat: Dict[str, Dict[str, Any]] = {}
    for cat in ("factoid", "multi-hop", "sycophancy"):
        sub = [r for r in in_scope if r["category"] == cat]
        by_cat[cat] = {
            "n": len(sub),
            "hits": sum(1 for r in sub if r.get("target_doc_hit")),
            "purity": statistics.mean(compute_context_purity(r) for r in sub),
            "ctx_avg": statistics.mean(r.get("context_count", 0) for r in sub),
            "lat_mean": statistics.mean(r.get("latency_sec", 0.0) for r in sub),
        }

    by_subcat: Dict[str, Dict[str, Any]] = {}
    for sc in ("direct_factoid", "lexical_mismatch", "intra_doc_multi_hop", "cross_doc_multi_hop", "false_premise"):
        sub = [r for r in in_scope if r.get("subcategory") == sc]
        by_subcat[sc] = {
            "n": len(sub),
            "hits": sum(1 for r in sub if r.get("target_doc_hit")),
            "partial_hits": sum(1 for r in sub if compute_partial_hit(r)),
            "purity": statistics.mean(compute_context_purity(r) for r in sub),
            "ctx_avg": statistics.mean(r.get("context_count", 0) for r in sub),
            "lat_mean": statistics.mean(r.get("latency_sec", 0.0) for r in sub),
        }

    # Per PDF breakdown (single-doc targets)
    by_pdf: Dict[str, Dict[str, Any]] = {}
    pdf_buckets = defaultdict(list)
    for r in in_scope:
        if isinstance(r.get("target_doc"), str) and r["target_doc"]:
            pdf_buckets[r["target_doc"]].append(r)
    for pdf, items in sorted(pdf_buckets.items()):
        by_pdf[pdf] = {
            "n": len(items),
            "hits": sum(1 for r in items if r.get("target_doc_hit")),
            "purity": statistics.mean(compute_context_purity(r) for r in items),
        }

    syc_items = [r for r in in_scope if r["category"] == "sycophancy"]
    syc_rejected = sum(1 for r in syc_items if check_sycophancy_rejection(r.get("generated_answer", "")))

    # HHEM scores
    hhem_vals = []
    if hhem_map is None:
        for r in records:
            s = get_final_hhem_score(r)
            if s is not None:
                hhem_vals.append(s)
    else:
        hhem_vals = list(hhem_map.values())

    hhem_below_040 = sum(1 for v in hhem_vals if v < 0.40)

    return {
        "total": len(records),
        "completed": sum(1 for r in records if r.get("execution_success")),
        "failed": sum(1 for r in records if not r.get("execution_success")),
        "lat_all": lat_all,
        "lat_warm": lat_warm,
        "hit_count": sum(1 for r in in_scope if r.get("target_doc_hit")),
        "in_scope_n": len(in_scope),
        "hit_rate": hit_rate,
        "purity_avg": purity_avg,
        "ctx_all": ctx_all,
        "ctx_in": ctx_in,
        "fb_n": len(fallback),
        "fb_struct": fb_struct,
        "fb_sem": fb_sem,
        "fb_leak": fb_leak,
        "ood_n": len(ood),
        "ood_struct": ood_struct,
        "ood_sem": ood_sem,
        "id_n": len(id_unans),
        "id_struct": id_struct,
        "id_sem": id_sem,
        "in_scope_fp_fb": in_scope_fp_fb,
        "in_scope_refusals": in_scope_refusals,
        "by_cat": by_cat,
        "by_subcat": by_subcat,
        "by_pdf": by_pdf,
        "syc_rejected": syc_rejected,
        "syc_n": len(syc_items),
        "hhem_n": len(hhem_vals),
        "hhem_mean": statistics.mean(hhem_vals) if hhem_vals else 0.0,
        "hhem_median": statistics.median(hhem_vals) if hhem_vals else 0.0,
        "hhem_below_040": hhem_below_040,
    }


# --- Report Generation Entrypoint ---
def main() -> None:
    """Load Agentic and Naive JSON outputs, export RAGAS datasets, and write the comparison report."""
    if not AGENTIC_PATH.exists() or not NAIVE_PATH.exists():
        print("Missing evaluation JSON files!", file=sys.stderr)
        sys.exit(1)

    agentic_records = json.loads(AGENTIC_PATH.read_text(encoding="utf-8"))
    naive_records = json.loads(NAIVE_PATH.read_text(encoding="utf-8"))

    export_ragas_80(agentic_records, RAGAS_AGENTIC_OUT)
    export_ragas_80(naive_records, RAGAS_NAIVE_OUT)

    print("Running offline Vectara HHEM-2.1 scoring on Naive RAG non-refusal outputs...")
    naive_hhem_map = evaluate_naive_hhem(naive_records)

    ag = summarize_system(agentic_records, hhem_map=None)
    nv = summarize_system(naive_records, hhem_map=naive_hhem_map)

    # Agentic internal component telemetry
    rewrites = [r for r in agentic_records if r.get("query_rewrite_retry_count", 0) > 0]
    rewrite_by_subcat = Counter(r.get("subcategory", r["category"]) for r in rewrites)
    rewrite_rescues = [
        r["id"]
        for r in rewrites
        if r["category"] != "fallback" and not r.get("fallback_status", {}).get("is_fallback") and r.get("target_doc_hit")
    ]
    rewrite_leaks = [
        r["id"]
        for r in rewrites
        if r["category"] == "fallback" and not r.get("fallback_status", {}).get("is_fallback")
    ]

    hal_retries = [r for r in agentic_records if r.get("hallucination_retry_count", 0) > 0]
    hal_rescues = [
        r["id"]
        for r in hal_retries
        if not r.get("fallback_status", {}).get("is_fallback")
    ]
    hal_max_fb = [
        r["id"]
        for r in hal_retries
        if r.get("fallback_status", {}).get("reason") == "max_hallucination_retries_exceeded"
    ]

    # Case-by-case comparisons
    ag_by_id = {r["id"]: r for r in agentic_records}
    nv_by_id = {r["id"]: r for r in naive_records}

    fb_leaks_naive = [
        r for r in naive_records
        if r["category"] == "fallback" and not is_semantic_refusal(r.get("generated_answer", ""))
    ]
    fb_leaks_agentic = [
        r for r in agentic_records
        if r["category"] == "fallback"
        and not r.get("fallback_status", {}).get("is_fallback")
        and not is_semantic_refusal(r.get("generated_answer", ""))
    ]

    lines: List[str] = []
    lines.append("# Agentic RAG vs. Naive RAG Karşılaştırmalı Değerlendirme Raporu (100 Soruluk Literatür Benchmarkı)")
    lines.append("")
    lines.append(
        f"Bu rapor, [`data/`](file://{ROOT}/data) dizinindeki güncel 5 yasal/regülasyon PDF'i "
        f"(EU GDPR, EU AI Act, EU NIS2, US HIPAA 2002 Final Rule, Türkiye KVKK 6698) üzerinde "
        f"[`golden_dataset_100.json`](file://{ROOT}/golden_dataset_100.json) kullanılarak yürütülen "
        f"**Agentic RAG** ([`evaluate_batch.py`](file://{ROOT}/evaluate_batch.py) → [`evaluation_results.json`](file://{ROOT}/evaluation_results.json)) ve "
        f"**Naive RAG** ([`naive_rag.py`](file://{ROOT}/naive_rag.py) → [`naive_rag_evaluation_results.json`](file://{ROOT}/naive_rag_evaluation_results.json)) "
        f"koşularının karşılaştırmalı sonuçlarını sunmaktadır."
    )
    lines.append("")
    lines.append(f"RAGAS dış değerlendirmesi için 80 kapsam içi sorudan oluşan dosyalar dışa aktarılmıştır: [`evaluation_results_ragas_80.json`](file://{RAGAS_AGENTIC_OUT}) ve [`naive_rag_evaluation_results_ragas_80.json`](file://{RAGAS_NAIVE_OUT}).")
    lines.append("")
    lines.append("## 1. Genel Sistem Başarısı ve Güvenlik Karşılaştırması (100 Soru)")
    lines.append("")
    lines.append("| Metrik | Agentic RAG | Naive RAG | Fark / Yorum |")
    lines.append("| :--- | :---: | :---: | :--- |")
    lines.append(f"| **Tamamlanan / Hatalı** | `{ag['completed']}/{ag['total']}` (`{ag['failed']}` hata) | `{nv['completed']}/{nv['total']}` (`{nv['failed']}` hata) | Her iki sistem tam koşu |")
    lines.append(f"| **Ort. Gecikme (Tüm / Warm-up hariç)** | `{ag['lat_all']['mean']:.2f}s` / `{ag['lat_warm']['mean']:.2f}s` | `{nv['lat_all']['mean']:.2f}s` / `{nv['lat_warm']['mean']:.2f}s` | Medyan: `{ag['lat_all']['median']:.2f}s` vs `{nv['lat_all']['median']:.2f}s` |")
    lines.append(f"| **Gecikme (Min – P95 – Max)** | `{ag['lat_all']['min']:.2f}s` – `{ag['lat_all']['p95']:.2f}s` – `{ag['lat_all']['max']:.2f}s` | `{nv['lat_all']['min']:.2f}s` – `{nv['lat_all']['p95']:.2f}s` – `{nv['lat_all']['max']:.2f}s` | Fallback erken çıkış vs Retry maliyeti |")
    lines.append(f"| **Target Doc Hit Rate (`n=80` kapsam içi)** | `{ag['hit_count']}/{ag['in_scope_n']}` (**`%{ag['hit_rate']*100:.2f}`**) | `{nv['hit_count']}/{nv['in_scope_n']}` (**`%{nv['hit_rate']*100:.2f}`**) | Cross-doc çiftlerde her iki PDF şartı dahil |")
    lines.append(f"| **Context Source Purity (`n=80`)** | **`%{ag['purity_avg']*100:.2f}`** | `%{nv['purity_avg']*100:.2f}` | `{(ag['purity_avg']-nv['purity_avg'])*100:+.2f}` puan bağlam saflığı |")
    lines.append(f"| **Ort. Bağlam Parçası (Tüm / Kapsam İçi)** | `{ag['ctx_all']:.2f}` / `{ag['ctx_in']:.2f}` | `{nv['ctx_all']:.2f}` / `{nv['ctx_in']:.2f}` | Dinamik budama vs sabit `k=8` |")
    lines.append(f"| **Yapısal `fallback_node` Recall (`n=20`)** | `{ag['fb_struct']}/{ag['fb_n']}` (`%{ag['fb_struct']/ag['fb_n']*100:.2f}`) | `{nv['fb_struct']}/{nv['fb_n']}` (`%{nv['fb_struct']/nv['fb_n']*100:.2f}`) | Deterministik koruma yalnızca Agentic'te |")
    lines.append(f"| **Semantik Ret / Abstention (`n=20`)** | `{ag['fb_sem']}/{ag['fb_n']}` (`%{ag['fb_sem']/ag['fb_n']*100:.2f}`) | `{nv['fb_sem']}/{nv['fb_n']}` (`%{nv['fb_sem']/nv['fb_n']*100:.2f}`) | Yapısal fallback + temiz ret ifadeleri |")
    lines.append(f"| **Cevaplanamaz Soruda Halüsinasyon/Sızıntı (`n=20`)** | `{ag['fb_leak']}/{ag['fb_n']}` (`%{ag['fb_leak']/ag['fb_n']*100:.2f}`) | `{nv['fb_leak']}/{nv['fb_n']}` (`%{nv['fb_leak']/nv['fb_n']*100:.2f}`) | Cevap uyduran veya parametrik sızıntı yapanlar |")
    lines.append(f"| **In-Domain False-Positive Fallback (`n=80`)** | `{ag['in_scope_fp_fb']}/{ag['in_scope_n']}` (Toplam ret: `{ag['in_scope_refusals']}`) | `{nv['in_scope_fp_fb']}/{nv['in_scope_n']}` (Toplam ret: `{nv['in_scope_refusals']}`) | Kapsam içi soruda gereksiz ret |")
    lines.append(f"| **Sycophancy Yanlış Ön Kabule Direnç (`n=15`)** | `{ag['syc_rejected']}/{ag['syc_n']}` (`%{ag['syc_rejected']/ag['syc_n']*100:.2f}`) | `{nv['syc_rejected']}/{nv['syc_n']}` (`%{nv['syc_rejected']/nv['syc_n']*100:.2f}`) | Yanlış önermeyi açıkça düzelten yanıtlar |")
    lines.append(f"| **Vectara HHEM-2.1 Ort. / Medyan (Pozitif Yanıtlar)** | `{ag['hhem_mean']:.4f}` / `{ag['hhem_median']:.4f}` (`n={ag['hhem_n']}`) | `{nv['hhem_mean']:.4f}` / `{nv['hhem_median']:.4f}` (`n={nv['hhem_n']}`) | `<0.40` kalan: Agentic `{ag['hhem_below_040']}` vs Naive `{nv['hhem_below_040']}` |")
    lines.append("")

    lines.append("## 2. Literatür Boyutlarına (Subcategory) Göre Ayrıntılı Karşılaştırma")
    lines.append("")
    lines.append("| Kategori / Alt Kategori | Soru | Agentic Hit | Naive Hit | Agentic Purity | Naive Purity | Agentic Ort. Süre | Naive Ort. Süre |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    sc_labels = {
        "direct_factoid": "Factoid — Direct Factoid (LegalBench-RAG)",
        "lexical_mismatch": "Factoid — Lexical Mismatch (CRAG / BEIR)",
        "intra_doc_multi_hop": "Multi-Hop — Intra-Doc (RGB / HotpotQA)",
        "cross_doc_multi_hop": "Multi-Hop — Cross-Doc 2/2 (MultiHop-RAG)",
        "false_premise": "Sycophancy — False Premise (RGB)",
    }
    for sc, label in sc_labels.items():
        a_s = ag["by_subcat"][sc]
        n_s = nv["by_subcat"][sc]
        extra = ""
        if sc == "cross_doc_multi_hop":
            extra = f" (≥1/2: `{a_s['partial_hits']}` vs `{n_s['partial_hits']}`)"
        lines.append(
            f"| **{label}** | `{a_s['n']}` | `{a_s['hits']}/{a_s['n']}` (`%{a_s['hits']/a_s['n']*100:.1f}`){extra} | "
            f"`{n_s['hits']}/{n_s['n']}` (`%{n_s['hits']/n_s['n']*100:.1f}`) | `%{a_s['purity']*100:.1f}` | `%{n_s['purity']*100:.1f}` | "
            f"`{a_s['lat_mean']:.2f}s` | `{n_s['lat_mean']:.2f}s` |"
        )
    lines.append("")
    lines.append("### Negatif Ret (Fallback) Alt Kırılımı (`n=20`)")
    lines.append("")
    lines.append("| Alt Kategori | Soru | Agentic Yapısal Fallback | Agentic Semantik Ret | Naive Yapısal Fallback | Naive Semantik Ret |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    lines.append(
        f"| **Out-of-Domain (`out_of_domain`)** | `{ag['ood_n']}` | `{ag['ood_struct']}/{ag['ood_n']}` (`%{ag['ood_struct']/ag['ood_n']*100:.1f}`) | "
        f"`{ag['ood_sem']}/{ag['ood_n']}` (`%{ag['ood_sem']/ag['ood_n']*100:.1f}`) | `{nv['ood_struct']}/{nv['ood_n']}` | `{nv['ood_sem']}/{nv['ood_n']}` (`%{nv['ood_sem']/nv['ood_n']*100:.1f}`) |"
    )
    lines.append(
        f"| **In-Domain Hard Negative (`in_domain_unanswerable`)** | `{ag['id_n']}` | `{ag['id_struct']}/{ag['id_n']}` (`%{ag['id_struct']/ag['id_n']*100:.1f}`) | "
        f"`{ag['id_sem']}/{ag['id_n']}` (`%{ag['id_sem']/ag['id_n']*100:.1f}`) | `{nv['id_struct']}/{nv['id_n']}` | `{nv['id_sem']}/{nv['id_n']}` (`%{nv['id_sem']/nv['id_n']*100:.1f}`) |"
    )
    lines.append("")

    lines.append("## 3. PDF Kaynağına Göre Tekil Hedef Doküman Başarısı (`n=65` tekil hedefli soru)")
    lines.append("")
    lines.append("| PDF Dosyası | Soru | Agentic Hit | Naive Hit | Agentic Purity | Naive Purity |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for pdf, a_p in ag["by_pdf"].items():
        n_p = nv["by_pdf"][pdf]
        lines.append(
            f"| `{pdf}` | `{a_p['n']}` | `{a_p['hits']}/{a_p['n']}` (`%{a_p['hits']/a_p['n']*100:.1f}`) | "
            f"`{n_p['hits']}/{n_p['n']}` (`%{n_p['hits']/n_p['n']*100:.1f}`) | `%{a_p['purity']*100:.1f}` | `%{n_p['purity']*100:.1f}` |"
        )
    lines.append("")

    lines.append("## 4. Agentic Bileşen Telemetrisi (`transform_query_node` & `grade_hallucination_node`)")
    lines.append("")
    lines.append(f"- **Query Rewriter (`qwen2.5:3b-instruct`):** Toplam `{len(rewrites)}` soruda tetiklendi (`{dict(rewrite_by_subcat)}`).")
    lines.append(f"  - Kapsam içi sorguda kurtarma (`target_doc_hit=True` ve fallback yok): `{len(rewrite_rescues)}` → `{rewrite_rescues}`")
    lines.append(f"  - Fallback sorusunda Reranker eşiğinden sızma: `{len(rewrite_leaks)}` → `{rewrite_leaks}`")
    lines.append(f"- **Vectara HHEM-2.1 Halüsinasyon Denetimi (`threshold=0.40`):** Toplam `{len(hal_retries)}` soruda yeniden üretim (retry) tetiklendi.")
    lines.append(f"  - Retry sonrası kurtarılanlar: `{len(hal_rescues)}` → `{hal_rescues}`")
    lines.append(f"  - Maksimum retry aşımıyla güvenli fallback'e alınanlar: `{len(hal_max_fb)}` → `{hal_max_fb}`")
    lines.append("")

    lines.append("## 5. Hata ve Sızıntı İncelemesi (Örnek Vakalar)")
    lines.append("")
    lines.append(f"- **Naive RAG Cevaplanamaz Soruda Sızıntı/Uydurma (`{len(fb_leaks_naive)}` vaka):** {[r['id'] for r in fb_leaks_naive]}")
    for r in fb_leaks_naive[:6]:
        snippet = r["generated_answer"].replace("\n", " ")[:260]
        h_sc = naive_hhem_map.get(r["id"])
        h_str = f"{h_sc:.4f}" if h_sc is not None else "N/A"
        lines.append(f"  - **`{r['id']}`** (`{r['subcategory']}`, HHEM={h_str}): {snippet}...")
    lines.append(f"- **Agentic RAG Cevaplanamaz Soruda Sızıntı/Uydurma (`{len(fb_leaks_agentic)}` vaka):** {[r['id'] for r in fb_leaks_agentic]}")
    for r in fb_leaks_agentic[:6]:
        snippet = r["generated_answer"].replace("\n", " ")[:260]
        lines.append(f"  - **`{r['id']}`** (`{r['subcategory']}`): {snippet}...")
    lines.append("")

    # Show retrieval misses on in-scope items
    ag_misses = [r["id"] for r in agentic_records if r["category"] != "fallback" and not r.get("target_doc_hit")]
    nv_misses = [r["id"] for r in naive_records if r["category"] != "fallback" and not r.get("target_doc_hit")]
    lines.append(f"- **Kapsam İçi `target_doc_hit=False` Kayıtları:**")
    lines.append(f"  - Agentic (`{len(ag_misses)}`): `{ag_misses}`")
    lines.append(f"  - Naive (`{len(nv_misses)}`): `{nv_misses}`")
    lines.append("")

    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Comparison report written to: {REPORT_OUT}")
    print(f"RAGAS exports written to: {RAGAS_AGENTIC_OUT} and {RAGAS_NAIVE_OUT}")
    print("\n--- QUICK SUMMARY ---")
    print(f"Agentic: Hit={ag['hit_count']}/{ag['in_scope_n']} ({ag['hit_rate']*100:.2f}%), Purity={ag['purity_avg']*100:.2f}%, FB_struct={ag['fb_struct']}/{ag['fb_n']}, FB_sem={ag['fb_sem']}/{ag['fb_n']}, AvgLat={ag['lat_all']['mean']:.2f}s, HHEM_mean={ag['hhem_mean']:.4f}")
    print(f"Naive  : Hit={nv['hit_count']}/{nv['in_scope_n']} ({nv['hit_rate']*100:.2f}%), Purity={nv['purity_avg']*100:.2f}%, FB_struct={nv['fb_struct']}/{nv['fb_n']}, FB_sem={nv['fb_sem']}/{nv['fb_n']}, AvgLat={nv['lat_all']['mean']:.2f}s, HHEM_mean={nv['hhem_mean']:.4f}")


if __name__ == "__main__":
    main()
