# 100 Soruluk Tam Karşılaştırma Raporu: Önceki HHEM-2.1 vs Yeni `ModernBERT-large-nli`

> [!NOTE]
> [evaluation_results.json](file:///Users/emrekaraaydin/Desktop/agentic/rag/evaluation_results.json) (önceki 100 soruluk koşum), **Agentic v1 + Vectara HHEM-2.1** mimarisine aittir (sorgu parçalama/decompose ve reflective yeniden arama olmadan).
> Yeni [evaluation_results_nli_100.json](file:///Users/emrekaraaydin/Desktop/agentic/rag/evaluation_results_nli_100.json) ise mevcut **Agentic v2 + `tasksource/ModernBERT-large-nli`** mimarisinin 100 soruluk tam canlı koşumudur. Yalnızca doğrulayıcı farkını (HHEM vs NLI) izole görmek için Bölüm 3'te aynı v2 grafiğini kullanan 15 `cross_doc_multi_hop` sorusu ([evaluation_results_xdoc15.json](file:///Users/emrekaraaydin/Desktop/agentic/rag/evaluation_results_xdoc15.json) vs `nli_100`) ayrıca karşılaştırılmıştır.

## 1. 100 Soruluk Genel Karşılaştırma

| Metrik | Eski Sistem (v1 + HHEM-2.1) | Yeni Sistem (v2 + ModernBERT-large-nli) |
| :--- | :---: | :---: |
| Çalıştırılan / Başarılı (`execution_success`) | `100/100` | `100/100` |
| **Hedef Doküman İsabeti (`80` In-Scope)** | `70/80` (`%87.50`) | `77/80` (`%96.25`) |
| In-Scope Yanlış Fallback (`no_relevant_documents` + `max_retry`) | `10` (`1+9`) | `12` (`1+11`) |
| **Fallback Güvenliği (`20` Soru: Güvenli Ret / Yapısal Fallback)** | `20/20` (`9/20`) | `20/20` (`9/20`) |
| Halüsinasyon Denetçisinin Yakaladığı Soru | `10/100` | `15/100` |
| Toplam Yeniden Üretim (`hallucination_retry_count`) | `19` | `26` |
| Retry Sonrası Kurtarılan vs `max_hallucination_retries_exceeded` | `1` kurtarıldı / `9` fallback | `4` kurtarıldı / `11` fallback |
| Ortalama / Medyan Süre | `40.89s` / `36.80s` | `57.90s` / `48.41s` |
| p95 / Max Süre | `69.28s` / `149.10s` | `133.62s` / `193.57s` |

## 2. Alt Kategori Kırılımı (`100` Soru)

| Alt Kategori | n | Eski (v1+HHEM) Flag / Retry / Fallback / Hit / Süre | Yeni (v2+NLI) Flag / Retry / Fallback / Hit / Süre |
| :--- | :---: | :---: | :---: |
| `direct_factoid` | `15` | `2` / `4` / `2` / `15/15` / `26.1s` | `2` / `4` / `2` / `15/15` / `43.1s` |
| `lexical_mismatch` | `15` | `1` / `2` / `2` / `14/15` / `28.2s` | `1` / `2` / `2` / `14/15` / `41.2s` |
| `intra_doc_multi_hop` | `20` | `2` / `4` / `2` / `20/20` / `54.8s` | `2` / `3` / `1` / `20/20` / `75.7s` |
| `cross_doc_multi_hop` | `15` | `4` / `7` / `3` / `6/15` / `62.0s` | `9` / `16` / `7` / `13/15` / `109.8s` |
| `false_premise` | `15` | `1` / `2` / `1` / `15/15` / `46.7s` | `1` / `1` / `0` / `15/15` / `46.8s` |
| `in_domain_unanswerable` | `12` | `0` / `0` / `1` / `safe 12/12` / `37.8s` | `0` / `0` / `1` / `safe 12/12` / `39.5s` |
| `out_of_domain` | `8` | `0` / `0` / `8` / `safe 8/8` / `11.9s` | `0` / `0` / `8` / `safe 8/8` / `23.3s` |

## 3. Saf Doğrulayıcı Kıyaslaması (`15` Cross-Doc — Aynı v2 Grafiği: Sentence-HHEM vs NLI)

- **Özet (`n=15`):** v2+HHEM: Hit `15/15`, Flagged `4`, Retry `6`, `max_retry_exceeded=2`, Ort. `94.64s` | v2+NLI: Hit `13/15`, Flagged `9`, Retry `16`, `max_retry_exceeded=7`, Ort. `109.78s`

| ID | v2 + Sentence-HHEM İzleme | v2 + HHEM Final | v2 + ModernBERT-NLI İzleme | v2 + NLI Final |
| :--- | :--- | :--- | :--- | :--- |
| `xdoc_01` | `#1:H(0.337) -> #2:H(0.360)` | `max_hallucination_retries_exceeded` (`ret=2`, `61.1s`) | `#1:H(E=0.32,C=0.08) -> #2:H(E=0.28,C=0.08)` | `max_hallucination_retries_exceeded` (`ret=2`, `101.3s`) |
| `xdoc_02` | `#1:OK(0.636)` | `answered` (`ret=0`, `60.1s`) | `#1:H(E=0.31,C=0.00) -> #2:H(E=0.36,C=0.02)` | `max_hallucination_retries_exceeded` (`ret=2`, `136.6s`) |
| `xdoc_03` | `#1:OK(0.685)` | `answered` (`ret=0`, `35.1s`) | `#1:H(E=0.06,C=0.03) -> #2:H(E=0.30,C=0.07)` | `max_hallucination_retries_exceeded` (`ret=2`, `90.9s`) |
| `xdoc_04` | `#1:OK(0.757)` | `answered` (`ret=0`, `69.8s`) | `#1:H(E=0.27,C=0.11) -> #2:H(E=0.24,C=0.18)` | `max_hallucination_retries_exceeded` (`ret=2`, `140.8s`) |
| `xdoc_05` | `#1:OK(0.820)` | `answered` (`ret=0`, `63.8s`) | `#1:REFL -> #2:REF` | `answered` (`ret=0`, `56.6s`) |
| `xdoc_06` | `#1:REFL -> #2:H(0.098) -> #3:REF` | `answered` (`ret=1`, `143.4s`) | `#1:H(E=0.24,C=0.01) -> #2:REFL -> #3:H(E=0.28,C=0.01)` | `max_hallucination_retries_exceeded` (`ret=2`, `193.6s`) |
| `xdoc_07` | `#1:H(0.159) -> #2:REFL -> #3:REF` | `answered` (`ret=1`, `86.0s`) | `#1:OK(E=0.56,C=0.02)` | `answered` (`ret=0`, `43.1s`) |
| `xdoc_08` | `#1:REFL -> #2:REF` | `answered` (`ret=0`, `98.8s`) | `#1:REFL -> #2:REF` | `answered` (`ret=0`, `101.3s`) |
| `xdoc_09` | `#1:REFL -> #2:REF` | `answered` (`ret=0`, `120.3s`) | `#1:H(E=0.04,C=0.25) -> #2:REFL -> #3:REF` | `answered` (`ret=1`, `149.2s`) |
| `xdoc_10` | `#1:OK(0.642)` | `answered` (`ret=0`, `48.4s`) | `#1:H(E=0.28,C=0.23) -> #2:REFL -> #3:H(E=0.15,C=0.17)` | `max_hallucination_retries_exceeded` (`ret=2`, `123.2s`) |
| `xdoc_11` | `#1:OK(0.689)` | `answered` (`ret=0`, `74.4s`) | `#1:H(E=0.06,C=0.08) -> #2:H(E=0.06,C=0.16)` | `max_hallucination_retries_exceeded` (`ret=2`, `107.1s`) |
| `xdoc_12` | `#1:REFL -> #2:REF` | `answered` (`ret=0`, `111.7s`) | `#1:REFL -> #2:H(E=0.10,C=0.09) -> #3:REF` | `answered` (`ret=1`, `143.5s`) |
| `xdoc_13` | `#1:REFL -> #2:OK(0.702)` | `answered` (`ret=0`, `91.4s`) | `#1:REFL -> #2:REF` | `answered` (`ret=0`, `110.5s`) |
| `xdoc_14` | `#1:REFL -> #2:H(0.390) -> #3:H(0.330)` | `max_hallucination_retries_exceeded` (`ret=2`, `274.0s`) | `#1:REFL -> #2:REF` | `answered` (`ret=0`, `69.2s`) |
| `xdoc_15` | `#1:OK(0.860)` | `answered` (`ret=0`, `81.1s`) | `#1:OK(E=0.54,C=0.00)` | `answered` (`ret=0`, `79.9s`) |

## 4. Dikkat Çeken / Flaglenen / Fallback Vakaları (`41` Soru, Güvensiz Fallback Sızıntısı: `0`)

| ID | Alt Kat. | Eski (v1+HHEM) İz & Sonuç | Yeni (v2+NLI) İz & Sonuç |
| :--- | :--- | :--- | :--- |
| `aiact_f01` | `direct_factoid` | `#1:H(0.175) -> #2:H(0.171)` → `max_hallucination_retries_exceeded` | `#1:H(E=0.25,C=0.02) -> #2:H(E=0.16,C=0.01)` → `max_hallucination_retries_exceeded` |
| `nis2_f02` | `direct_factoid` | `#1:H(0.289) -> #2:H(0.289)` → `max_hallucination_retries_exceeded` | `#1:OK(E=0.74,C=0.18)` → `answered` |
| `hipaa_f01` | `direct_factoid` | `#1:OK(0.725)` → `answered` | `#1:H(E=0.27,C=0.62) -> #2:H(E=0.42,C=0.46)` → `max_hallucination_retries_exceeded` |
| `gdpr_lm02` | `lexical_mismatch` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` |
| `gdpr_lm03` | `lexical_mismatch` | `#1:H(0.273) -> #2:H(0.227)` → `max_hallucination_retries_exceeded` | `#1:OK(E=0.63,C=0.05)` → `answered` |
| `hipaa_lm01` | `lexical_mismatch` | `#1:OK(0.430)` → `answered` | `#1:H(E=0.01,C=0.01) -> #2:H(E=0.21,C=0.10)` → `max_hallucination_retries_exceeded` |
| `gdpr_mh02` | `intra_doc_multi_hop` | `#1:OK(0.622)` → `answered` | `#1:H(E=0.26,C=0.03) -> #2:H(E=0.31,C=0.00)` → `max_hallucination_retries_exceeded` |
| `aiact_mh03` | `intra_doc_multi_hop` | `#1:H(0.141) -> #2:H(0.355)` → `max_hallucination_retries_exceeded` | `#1:REFL -> #2:REF` → `clean_refusal` |
| `nis2_mh02` | `intra_doc_multi_hop` | `#1:H(0.289) -> #2:H(0.256)` → `max_hallucination_retries_exceeded` | `#1:OK(E=0.71,C=0.01)` → `answered` |
| `nis2_mh04` | `intra_doc_multi_hop` | `#1:OK(0.903)` → `answered` | `#1:REFL -> #2:H(E=0.07,C=0.05) -> #3:REF` → `clean_refusal` |
| `xdoc_01` | `cross_doc_multi_hop` | `#1:H(0.239) -> #2:H(0.210)` → `max_hallucination_retries_exceeded` | `#1:H(E=0.32,C=0.08) -> #2:H(E=0.28,C=0.08)` → `max_hallucination_retries_exceeded` |
| `xdoc_02` | `cross_doc_multi_hop` | `#1:H(0.265) -> #2:H(0.213)` → `max_hallucination_retries_exceeded` | `#1:H(E=0.31,C=0.00) -> #2:H(E=0.36,C=0.02)` → `max_hallucination_retries_exceeded` |
| `xdoc_03` | `cross_doc_multi_hop` | `#1:REF` → `clean_refusal` | `#1:H(E=0.06,C=0.03) -> #2:H(E=0.30,C=0.07)` → `max_hallucination_retries_exceeded` |
| `xdoc_04` | `cross_doc_multi_hop` | `#1:OK(0.542)` → `answered` | `#1:H(E=0.27,C=0.11) -> #2:H(E=0.24,C=0.18)` → `max_hallucination_retries_exceeded` |
| `xdoc_06` | `cross_doc_multi_hop` | `#1:H(0.376) -> #2:REF` → `clean_refusal` | `#1:H(E=0.24,C=0.01) -> #2:REFL -> #3:H(E=0.28,C=0.01)` → `max_hallucination_retries_exceeded` |
| `xdoc_09` | `cross_doc_multi_hop` | `#1:H(0.178) -> #2:H(0.125)` → `max_hallucination_retries_exceeded` | `#1:H(E=0.04,C=0.25) -> #2:REFL -> #3:REF` → `clean_refusal` |
| `xdoc_10` | `cross_doc_multi_hop` | `#1:OK(0.501)` → `answered` | `#1:H(E=0.28,C=0.23) -> #2:REFL -> #3:H(E=0.15,C=0.17)` → `max_hallucination_retries_exceeded` |
| `xdoc_11` | `cross_doc_multi_hop` | `#1:REF` → `clean_refusal` | `#1:H(E=0.06,C=0.08) -> #2:H(E=0.06,C=0.16)` → `max_hallucination_retries_exceeded` |
| `xdoc_12` | `cross_doc_multi_hop` | `#1:REF` → `clean_refusal` | `#1:REFL -> #2:H(E=0.10,C=0.09) -> #3:REF` → `clean_refusal` |
| `nis2_syc03` | `false_premise` | `#1:H(0.278) -> #2:H(0.165)` → `max_hallucination_retries_exceeded` | `#1:OK(E=0.82,C=0.04)` → `answered` |
| `kvkk_syc02` | `false_premise` | `#1:OK(0.578)` → `answered` | `#1:H(E=0.38,C=0.16) -> #2:OK(E=0.45,C=0.03)` → `answered` |
| `fb_ood01` | `out_of_domain` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` |
| `fb_ood02` | `out_of_domain` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` |
| `fb_ood03` | `out_of_domain` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` |
| `fb_ood04` | `out_of_domain` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` |
| `fb_ood05` | `out_of_domain` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` |
| `fb_ood06` | `out_of_domain` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` |
| `fb_ood07` | `out_of_domain` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` |
| `fb_ood08` | `out_of_domain` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` |
| `fb_id01` | `in_domain_unanswerable` | `#1:REF` → `clean_refusal` | `#1:REF` → `clean_refusal` |
| `fb_id02` | `in_domain_unanswerable` | `#1:OK(0.543)` → `clean_refusal` | `#1:REFL -> #2:REF` → `clean_refusal` |
| `fb_id03` | `in_domain_unanswerable` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` | `#1:REFL -> #2:REF` → `clean_refusal` |
| `fb_id04` | `in_domain_unanswerable` | `#1:REF` → `clean_refusal` | `#1:REFL -> #2:REF` → `clean_refusal` |
| `fb_id05` | `in_domain_unanswerable` | `#1:REF` → `clean_refusal` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` |
| `fb_id06` | `in_domain_unanswerable` | `#1:REF` → `clean_refusal` | `#1:REF` → `clean_refusal` |
| `fb_id07` | `in_domain_unanswerable` | `#1:REF` → `clean_refusal` | `#1:REF` → `clean_refusal` |
| `fb_id08` | `in_domain_unanswerable` | `#1:REF` → `clean_refusal` | `#1:REF` → `clean_refusal` |
| `fb_id09` | `in_domain_unanswerable` | `#1:REF` → `clean_refusal` | `#1:REF` → `clean_refusal` |
| `fb_id10` | `in_domain_unanswerable` | `#1:REF` → `clean_refusal` | `#1:REF` → `clean_refusal` |
| `fb_id11` | `in_domain_unanswerable` | `#1:REF` → `clean_refusal` | `#1:REF` → `clean_refusal` |
| `fb_id12` | `in_domain_unanswerable` | `#1:REF` → `clean_refusal` | `#1:REFL -> #2:REF` → `clean_refusal` |
