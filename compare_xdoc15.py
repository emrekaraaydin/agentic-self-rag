"""Three-way comparison script across Naive RAG, Agentic v1, and Agentic v2 on 15 cross-document queries."""

import json
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

os.environ["ORT_DISABLE_TELEMETRY"] = "1"

from compare_agentic_vs_naive import compute_context_purity, is_semantic_refusal

# --- Input & Output Paths ---
ROOT = Path(__file__).resolve().parent
NAIVE_PATH = ROOT / "naive_rag_evaluation_results.json"
AGENTIC_V1_PATH = ROOT / "evaluation_results.json"
AGENTIC_V2_PATH = ROOT / "evaluation_results_xdoc15.json"
REPORT_OUT = ROOT / "xdoc15_agentic_v2_karsilastirma.md"


# --- Cross-Document Scoring Helpers ---
def balanced_both_ge2(rec: Dict[str, Any]) -> bool:
    """Return True when both required statutory PDFs appear at least twice in retrieved sources."""
    targets = rec.get("target_doc") or []
    if not isinstance(targets, list) or len(targets) != 2:
        return False
    counts = Counter(s.get("source") for s in rec.get("sources", []) if isinstance(s, dict))
    return all(counts[t] >= 2 for t in targets)


def answer_status(rec: Dict[str, Any]) -> str:
    """Classify an answer as fallback, full_refusal, partial_answer_missing_part, or complete."""
    if rec.get("fallback_status", {}).get("is_fallback"):
        return f"fallback ({rec['fallback_status'].get('reason')})"
    ans = rec.get("generated_answer", "")
    low = ans.lower()
    if is_semantic_refusal(ans):
        return "full_refusal"
    partial_phrases = (
        "does not contain",
        "do not contain",
        "does not provide",
        "do not provide",
        "not mentioned in the provided",
        "not directly mentioned",
        "not available in",
    )
    if any(p in low for p in partial_phrases):
        return "partial_answer_missing_part"
    return "complete_comparative_answer"


def get_hhem_scores(rec: Dict[str, Any]) -> List[float]:
    """Extract all verifier scores logged during hallucination grading retries."""
    return [
        float(l["hhem_score"])
        for l in rec.get("audit_logs", [])
        if isinstance(l, dict) and l.get("node") == "grade_hallucination_node" and l.get("hhem_score") is not None
    ]


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute hit rate, balanced 2/2 count, purity, response status counts, and latency metrics."""
    hits = sum(1 for r in records if r.get("target_doc_hit"))
    bal = sum(1 for r in records if balanced_both_ge2(r))
    pur = statistics.mean(compute_context_purity(r) for r in records)
    ctx = statistics.mean(r.get("context_count", 0) for r in records)
    lats = [r.get("latency_sec", 0.0) for r in records]
    statuses = Counter(answer_status(r) for r in records)
    complete = statuses["complete_comparative_answer"]
    partial = statuses["partial_answer_missing_part"]
    refusals = statuses["full_refusal"]
    fallbacks = sum(v for k, v in statuses.items() if k.startswith("fallback"))
    return {
        "n": len(records),
        "hits": hits,
        "balanced": bal,
        "purity": pur,
        "ctx": ctx,
        "lat_mean": statistics.mean(lats),
        "lat_median": statistics.median(lats),
        "complete": complete,
        "partial": partial,
        "refusals": refusals,
        "fallbacks": fallbacks,
    }


# --- Three-Way Comparison Report Entrypoint ---
def main() -> None:
    """Generate the Markdown comparison report for Naive, Agentic v1, and Agentic v2 on cross_doc_multi_hop."""
    nv_all = json.loads(NAIVE_PATH.read_text(encoding="utf-8"))
    v1_all = json.loads(AGENTIC_V1_PATH.read_text(encoding="utf-8"))
    v2 = json.loads(AGENTIC_V2_PATH.read_text(encoding="utf-8"))

    nv = [r for r in nv_all if r.get("subcategory") == "cross_doc_multi_hop"]
    v1 = [r for r in v1_all if r.get("subcategory") == "cross_doc_multi_hop"]

    nv_map = {r["id"]: r for r in nv}
    v1_map = {r["id"]: r for r in v1}
    v2_map = {r["id"]: r for r in v2}

    s_nv = summarize(nv)
    s_v1 = summarize(v1)
    s_v2 = summarize(v2)

    lines: List[str] = []
    lines.append("# `cross_doc_multi_hop` (15 Soru) Üçlü Kıyaslama: Naive RAG vs. Agentic RAG v1 vs. Agentic RAG v2")
    lines.append("")
    lines.append(
        f"Bu rapor, [`golden_dataset_100.json`](file://{ROOT}/golden_dataset_100.json) içindeki "
        "15 adet iki-mevzuat karşılaştırma (`cross_doc_multi_hop`: `xdoc_01`–`xdoc_15`) sorusu üzerinde:"
    )
    lines.append(f"- **Naive RAG** ([`naive_rag_evaluation_results.json`](file://{ROOT}/naive_rag_evaluation_results.json))")
    lines.append(f"- **Agentic RAG v1 (Tek Sorgu)** ([`evaluation_results.json`](file://{ROOT}/evaluation_results.json))")
    lines.append(f"- **Agentic RAG v2 (Decompose + Balanced Rerank + Self-Reflective Retrieval + Sentence HHEM)** ([`evaluation_results_xdoc15.json`](file://{ROOT}/evaluation_results_xdoc15.json))")
    lines.append("sonuçlarını birebir kıyaslamaktadır.")
    lines.append("")
    lines.append("## 1. Özet Karşılaştırma Tablosu (`n = 15`)")
    lines.append("")
    lines.append("| Metrik | Naive RAG | Agentic RAG v1 | Agentic RAG v2 (Yeni) |")
    lines.append("| :--- | :---: | :---: | :---: |")
    lines.append(f"| **2/2 Target Doc Hit Rate** | `{s_nv['hits']}/15` (**`%{s_nv['hits']/15*100:.1f}`**) | `{s_v1['hits']}/15` (`%{s_v1['hits']/15*100:.1f}`) | **`{s_v2['hits']}/15` (`%{s_v2['hits']/15*100:.1f}`)** |")
    lines.append(f"| **Dengeli Çift-Doküman (`her iki PDF ≥ 2 parça`)** | `{s_nv['balanced']}/15` (`%{s_nv['balanced']/15*100:.1f}`) | `{s_v1['balanced']}/15` (`%{s_v1['balanced']/15*100:.1f}`) | **`{s_v2['balanced']}/15` (`%{s_v2['balanced']/15*100:.1f}`)** |")
    lines.append(f"| **Context Source Purity** | `%{s_nv['purity']*100:.2f}` | `%{s_v1['purity']*100:.2f}` | **`%{s_v2['purity']*100:.2f}`** |")
    lines.append(f"| **Ort. Bağlam Sayısı** | `{s_nv['ctx']:.2f}` | `{s_v1['ctx']:.2f}` | `{s_v2['ctx']:.2f}` |")
    lines.append(f"| **Tam Karşılaştırmalı Yanıt (`complete`)** | `{s_nv['complete']}/15` | `{s_v1['complete']}/15` | **`{s_v2['complete']}/15`** |")
    lines.append(f"| **Kısmi Eksik Bildiren Yanıt (`partial`)** | `{s_nv['partial']}/15` | `{s_v1['partial']}/15` | `{s_v2['partial']}/15` |")
    lines.append(f"| **Tam Ret (`full_refusal`) / `fallback_node`** | `{s_nv['refusals']}/15` / `{s_nv['fallbacks']}/15` | `{s_v1['refusals']}/15` / `{s_v1['fallbacks']}/15` | `{s_v2['refusals']}/15` / `{s_v2['fallbacks']}/15` |")
    lines.append(f"| **Ort. / Medyan Gecikme** | `{s_nv['lat_mean']:.2f}s` / `{s_nv['lat_median']:.2f}s` | `{s_v1['lat_mean']:.2f}s` / `{s_v1['lat_median']:.2f}s` | `{s_v2['lat_mean']:.2f}s` / `{s_v2['lat_median']:.2f}s` |")
    lines.append("")

    # Telemetry
    decomp_multi = sum(1 for r in v2 if r.get("is_multi_part"))
    refl_items = [r for r in v2 if r.get("retrieval_reflection_retry_count", 0) > 0]
    hal_items = [r for r in v2 if r.get("hallucination_retry_count", 0) > 0]
    v2_hhem = [get_hhem_scores(r)[-1] for r in v2 if get_hhem_scores(r)]
    lines.append("## 2. Agentic RAG v2 Bileşen Telemetrisi")
    lines.append("")
    lines.append(f"- **Query Decomposition (`decompose_query_node`):** `{decomp_multi}/15` soruda çoklu alt sorgu planı üretildi.")
    lines.append(f"- **Self-Reflective Re-Retrieval (`reflect_missing_context_node`):** `{len(refl_items)}/15` soruda tetiklendi → `{[r['id'] for r in refl_items]}`.")
    lines.append(f"- **Sentence-Level HHEM-2.1 (`threshold=0.35`):** `{len(hal_items)}/15` soruda yeniden üretim denendi → `{[r['id'] for r in hal_items]}`; ortalama HHEM: `{statistics.mean(v2_hhem):.4f}`, medyan: `{statistics.median(v2_hhem):.4f}`.")
    lines.append("")

    lines.append("## 3. Soru Bazlı Ayrıntılı Karşılaştırma (`xdoc_01` – `xdoc_15`)")
    lines.append("")
    lines.append("| ID | Hedef Çift | Naive Hit & Durum | Agentic v1 Hit & Durum | Agentic v2 Hit & Durum | Agentic v2 Parça Dağılımı | v2 HHEM / Süre |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :---: |")

    for i in range(1, 16):
        qid = f"xdoc_{i:02d}"
        rn, r1, r2 = nv_map[qid], v1_map[qid], v2_map[qid]
        t_short = " + ".join(t.replace("_EN_TXT.pdf", "").replace(".pdf", "") for t in r2["target_doc"])
        c2 = Counter(s.get("source", "").replace("_EN_TXT.pdf", "").replace(".pdf", "") for s in r2.get("sources", []))
        h2 = get_hhem_scores(r2)
        h_str = "→".join(f"{s:.2f}" for s in h2) if h2 else "-"
        lines.append(
            f"| `{qid}` | `{t_short}` | `{rn['target_doc_hit']}` ({answer_status(rn)}) | "
            f"`{r1['target_doc_hit']}` ({answer_status(r1)}) | **`{r2['target_doc_hit']}`** ({answer_status(r2)}) | "
            f"`{dict(c2)}` | `{h_str}` / `{r2['latency_sec']:.1f}s` |"
        )
    lines.append("")

    lines.append("## 4. Yanıt Kalitesi Karşılaştırma Örnekleri")
    lines.append("")
    for i in range(1, 16):
        qid = f"xdoc_{i:02d}"
        rn, r1, r2 = nv_map[qid], v1_map[qid], v2_map[qid]
        lines.append(f"### `{qid}` — Sub-queries: `{r2.get('sub_queries')}`")
        lines.append(f"- **Ground Truth:** {r2['ground_truth']}")
        lines.append(f"- **Naive RAG (`hit={rn['target_doc_hit']}`):** {rn['generated_answer'].replace(chr(10), ' ')[:360]}")
        lines.append(f"- **Agentic v1 (`hit={r1['target_doc_hit']}`):** {r1['generated_answer'].replace(chr(10), ' ')[:360]}")
        lines.append(f"- **Agentic v2 (`hit={r2['target_doc_hit']}`):** {r2['generated_answer'].replace(chr(10), ' ')[:460]}")
        lines.append("")

    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written {REPORT_OUT}")
    print("Naive     :", s_nv)
    print("Agentic v1:", s_v1)
    print("Agentic v2:", s_v2)


if __name__ == "__main__":
    main()
