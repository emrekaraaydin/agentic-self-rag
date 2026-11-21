"""Google Colab (A100 40GB) uyumlu, kilitlenmeye dayanıklı (checkpoint'li) yerel RAGAS scripti.

Colab üzerinde çalıştırma adımları:

1. Paket kurulumu (Colab'ın pandas sürümüyle çakışmaması için pandas'ı listeye eklemeyin):
   %pip install -qU ragas langchain-ollama langchain-core langchain-community datasets
   (Kurulum sonrası üst menüden Runtime -> Restart session yapın.)

2. Ollama kurulumu ve arka planda başlatma:
   !curl -fsSL https://ollama.com/install.sh | sh
   import os, subprocess, time
   os.environ["OLLAMA_NUM_PARALLEL"] = "2"
   os.environ["OLLAMA_MAX_LOADED_MODELS"] = "2"
   subprocess.Popen(["ollama", "serve"])
   time.sleep(5)

3. Modellerin indirilmesi (A100 40GB VRAM için):
   - Judge LLM: qwen2.5:32b (~19 GB VRAM, JSON/talimat sadakati en yüksek açık kaynak hakem)
   - Embedding: bge-m3 (~1.2 GB VRAM, AnswerRelevancy & AnswerCorrectness için güçlü)
   !ollama pull qwen2.5:32b
   !ollama pull bge-m3

4. evaluation_results_ragas_80.json dosyasını yükleyip çalıştırın:
   !python evaluate_ragas_colab.py
"""
import json
import os
import urllib.request
from pathlib import Path

import pandas as pd
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

try:
    from ragas.metrics import ResponseRelevancy as RelevancyMetric
except ImportError:
    from ragas.metrics import AnswerRelevancy as RelevancyMetric


INPUT_PATH = Path(os.getenv("RAGAS_INPUT_PATH", "evaluation_results_ragas_80.json"))
OUTPUT_CSV = Path(os.getenv("RAGAS_OUTPUT_CSV", "ragas_results_80.csv"))
OUTPUT_SUMMARY = Path(os.getenv("RAGAS_OUTPUT_SUMMARY", "ragas_summary.json"))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "qwen2.5:32b")
EMBED_MODEL = os.getenv("RAGAS_EMBED_MODEL", "bge-m3")
BATCH_SIZE = int(os.getenv("RAGAS_BATCH_SIZE", "5"))

# OpenAI fallback hatalarını önlemek için dummy anahtar
os.environ.setdefault("OPENAI_API_KEY", "sk-ollama-local")


def preload_ollama_models(base_url: str, judge_model: str, embed_model: str) -> None:
    """Modelleri VRAM'e önceden yükler ve keep_alive=-1 ile kilitler (cold-start/swap kilitlenmesini önler)."""
    for endpoint, payload in [
        ("/api/generate", {"model": judge_model, "prompt": "", "keep_alive": -1}),
        ("/api/embed", {"model": embed_model, "input": "warmup", "keep_alive": -1}),
    ]:
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                resp.read()
        except Exception as exc:
            print(f"[Uyarı] Model ön-yükleme başarısız ({endpoint}): {exc}")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Girdi dosyası bulunamadı: {INPUT_PATH}")

    records = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    print(f"Yüklenen kapsam içi soru sayısı: {len(records)}")

    skipped = [r["id"] for r in records if not r.get("retrieved_contexts")]
    valid_records = [r for r in records if r.get("retrieved_contexts")]
    if skipped:
        print(f"Boş retrieved_contexts nedeniyle RAGAS dışı bırakılanlar ({len(skipped)}): {skipped}")

    # Checkpoint / kaldığı yerden devam desteği
    done_ids = set()
    existing_df = pd.DataFrame()
    if OUTPUT_CSV.exists():
        existing_df = pd.read_csv(OUTPUT_CSV)
        if "id" in existing_df.columns:
            done_ids = set(existing_df["id"].astype(str).tolist())
            print(f"Mevcut checkpoint bulundu: {len(done_ids)} kayıt zaten tamamlanmış, kalanlar işlenecek.")

    pending_records = [r for r in valid_records if r["id"] not in done_ids]

    if pending_records:
        print(f"Modeller VRAM'e ön-yükleniyor: {JUDGE_MODEL} + {EMBED_MODEL}...")
        preload_ollama_models(OLLAMA_BASE_URL, JUDGE_MODEL, EMBED_MODEL)

        llm = ChatOllama(
            model=JUDGE_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.0,
            num_ctx=8192,
            keep_alive=-1,
            client_kwargs={"timeout": 300.0},
        )
        embeddings = OllamaEmbeddings(
            model=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
            keep_alive=-1,
            client_kwargs={"timeout": 120.0},
        )

        evaluator_llm = LangchainLLMWrapper(llm)
        evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

        metrics = [
            Faithfulness(),
            AnswerCorrectness(),
            LLMContextRecall(),
            LLMContextPrecisionWithReference(),
            RelevancyMetric(),
        ]

        # Kilitlenmeyi önlemek için düşük eşzamanlılık (max_workers=2) ve yüksek timeout
        run_config = RunConfig(
            timeout=300,
            max_retries=5,
            max_wait=30,
            max_workers=2,
        )

        total_batches = (len(pending_records) + BATCH_SIZE - 1) // BATCH_SIZE
        for b_idx in range(0, len(pending_records), BATCH_SIZE):
            batch = pending_records[b_idx : b_idx + BATCH_SIZE]
            batch_no = (b_idx // BATCH_SIZE) + 1
            print(f"\n[Batch {batch_no}/{total_batches}] Değerlendiriliyor: {[r['id'] for r in batch]}")

            eval_dataset = EvaluationDataset.from_list(
                [
                    {
                        "user_input": r["user_input"],
                        "response": r["response"],
                        "retrieved_contexts": r["retrieved_contexts"],
                        "reference": r["reference"],
                    }
                    for r in batch
                ]
            )

            result = evaluate(
                dataset=eval_dataset,
                metrics=metrics,
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                run_config=run_config,
                raise_exceptions=False,
            )

            batch_df = result.to_pandas()
            meta_df = pd.DataFrame(
                [
                    {
                        "id": r["id"],
                        "domain": r.get("domain"),
                        "category": r["category"],
                        "latency_sec": r.get("latency_sec"),
                    }
                    for r in batch
                ]
            )
            merged_batch = pd.concat([meta_df, batch_df], axis=1)

            existing_df = (
                pd.concat([existing_df, merged_batch], ignore_index=True)
                if not existing_df.empty
                else merged_batch
            )
            existing_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
            print(f"[Batch {batch_no}/{total_batches}] Checkpoint kaydedildi -> {OUTPUT_CSV} (Toplam: {len(existing_df)})")

    final_df = pd.read_csv(OUTPUT_CSV)
    metric_cols = [
        c
        for c in final_df.columns
        if c not in {"id", "domain", "category", "user_input", "response", "retrieved_contexts", "reference"}
        and pd.api.types.is_numeric_dtype(final_df[c])
    ]

    overall = {col: round(float(final_df[col].mean(skipna=True)), 4) for col in metric_cols}
    by_category = (
        final_df.groupby("category")[metric_cols]
        .mean()
        .round(4)
        .to_dict(orient="index")
    )

    summary = {
        "judge_model": JUDGE_MODEL,
        "embed_model": EMBED_MODEL,
        "evaluated_count": len(final_df),
        "skipped_empty_context": skipped,
        "overall": overall,
        "by_category": by_category,
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== RAGAS GENEL SONUÇLAR ===")
    for k, v in overall.items():
        print(f"{k:35s}: {v:.4f}")

    print("\n=== KATEGORİ BAZLI ORTALAMALAR ===")
    print(final_df.groupby("category")[metric_cols].mean().round(4).to_string())
    print(f"\nTamamlandı -> {OUTPUT_CSV} ve {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
