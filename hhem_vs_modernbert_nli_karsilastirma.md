# Önceki HHEM-2.1 vs Yeni `ModernBERT-large-nli` Karşılaştırma Raporu (`n = 35` Katmanlı Alt Küme)

> [!NOTE]
> `evaluation_results.json` (önceki **100 soruluk** koşum), **Agentic v1 + Vectara HHEM-2.1** mimarisine aittir (sorgu parçalama/decompose ve reflective yeniden arama olmadan).
> Yeni [evaluation_results_nli_35.json](file:///Users/emrekaraaydin/Desktop/agentic/rag/evaluation_results_nli_35.json) ise **Agentic v2 + `tasksource/ModernBERT-large-nli`** mimarisiyle aynı 35 soruda canlı çalıştırılmıştır. Yalnızca doğrulayıcı farkını (HHEM vs NLI) izole görmek için Bölüm 3'te aynı v2 grafiğini kullanan 7 `xdoc` sorusu (`evaluation_results_xdoc15.json` vs `nli_35`) ayrıca karşılaştırılmıştır.

## 1. 35 Soruluk Genel Karşılaştırma (Tüm Alt Kategorilerden Katmanlı Örneklem)

| Metrik | Eski Sistem (v1 + HHEM-2.1) | Yeni Sistem (v2 + ModernBERT-large-nli) |
| :--- | :---: | :---: |
| Çalıştırılan / Başarılı (`execution_success`) | `35/35` | `35/35` |
| **Hedef Doküman İsabeti (`28` In-Scope)** | `27/28` (`%96.4`) | `27/28` (`%96.4`) |
| In-Scope Yanlış Fallback (Toplam = NoDocs + MaxRetry) | `9` (`0+9`) | `8` (`0+8`) |
| **Fallback Güvenliği (`7` Soru: Güvenli Ret / Yapısal Fallback)** | `7/7` (`3/7`) | `7/7` (`4/7`) |
| Halüsinasyon Denetçisinin Yakaladığı Soru | `10/35` | `10/35` |
| Toplam Yeniden Üretim (`hallucination_retry_count`) | `19` | `18` |
| Retry Sonrası Kurtarılan vs `max_hallucination_retries_exceeded` | `1` kurtarıldı / `9` fallback | `2` kurtarıldı / `8` fallback |
| Ortalama / Medyan Süre | `52.08s` / `46.57s` | `57.31s` / `43.30s` |
| p95 / Max Süre | `123.23s` / `149.10s` | `132.84s` / `170.01s` |

## 2. Alt Kategori Bazlı Dağılım (`7` Alt Kategori)

| Alt Kategori | n | Eski Flag / Retry / Hit / Ort. Süre | Yeni Flag / Retry / Hit / Ort. Süre |
| :--- | :---: | :---: | :---: |
| `direct_factoid` | `5` | `2` flag / `4` ret / `5/5` / `31.6s` | `2` flag / `4` ret / `5/5` / `35.0s` |
| `lexical_mismatch` | `5` | `1` flag / `2` ret / `5/5` / `31.2s` | `1` flag / `2` ret / `5/5` / `41.8s` |
| `intra_doc_multi_hop` | `5` | `2` flag / `4` ret / `5/5` / `77.2s` | `0` flag / `0` ret / `5/5` / `65.5s` |
| `cross_doc_multi_hop` | `7` | `4` flag / `7` ret / `6/7` / `85.8s` | `6` flag / `11` ret / `6/7` / `118.0s` |
| `false_premise` | `6` | `1` flag / `2` ret / `6/6` / `48.9s` | `1` flag / `1` ret / `6/6` / `42.0s` |
| `in_domain_unanswerable` | `4` | `0` flag / `0` ret / `safe 4/4` / `46.9s` | `0` flag / `0` ret / `safe 4/4` / `34.7s` |
| `out_of_domain` | `3` | `0` flag / `0` ret / `safe 3/3` / `13.7s` | `0` flag / `0` ret / `safe 3/3` / `25.8s` |

## 3. Saf Doğrulayıcı Kıyaslaması (`7` Cross-Doc — Aynı v2 Grafiği: Sentence-HHEM vs NLI)

- **Özet (`n=7`):** v2+HHEM: Hit `7/7`, Flagged `2`, Retry `3`, `max_retry_exceeded=1`, Ort. Süre `80.98s` | v2+NLI: Hit `6/7`, Flagged `6`, Retry `11`, `max_retry_exceeded=5`, Ort. Süre `118.00s`

| ID | v2 + Sentence-HHEM İzleme | v2 + HHEM Final Durum | v2 + ModernBERT-NLI İzleme | v2 + NLI Final Durum |
| :--- | :--- | :--- | :--- | :--- |
| `xdoc_01` | `#1:H(0.337) -> #2:H(0.360)` | `max_hallucination_retries_exceeded` (`ret=2`, `61.1s`) | `#1:H(E=0.32,C=0.08) -> #2:H(E=0.28,C=0.08)` | `max_hallucination_retries_exceeded` (`ret=2`, `82.6s`) |
| `xdoc_02` | `#1:OK(0.636)` | `answered` (`ret=0`, `60.1s`) | `#1:H(E=0.31,C=0.00) -> #2:H(E=0.36,C=0.02)` | `max_hallucination_retries_exceeded` (`ret=2`, `132.8s`) |
| `xdoc_04` | `#1:OK(0.757)` | `answered` (`ret=0`, `69.8s`) | `#1:H(E=0.27,C=0.11) -> #2:H(E=0.24,C=0.18)` | `max_hallucination_retries_exceeded` (`ret=2`, `126.5s`) |
| `xdoc_05` | `#1:OK(0.820)` | `answered` (`ret=0`, `63.8s`) | `#1:REFL -> #2:REF` | `answered` (`ret=0`, `53.9s`) |
| `xdoc_06` | `#1:REFL -> #2:H(0.098) -> #3:REF` | `answered` (`ret=1`, `143.4s`) | `#1:H(E=0.24,C=0.01) -> #2:REFL -> #3:H(E=0.28,C=0.01)` | `max_hallucination_retries_exceeded` (`ret=2`, `170.0s`) |
| `xdoc_09` | `#1:REFL -> #2:REF` | `answered` (`ret=0`, `120.3s`) | `#1:H(E=0.04,C=0.25) -> #2:REFL -> #3:REF` | `answered` (`ret=1`, `143.3s`) |
| `xdoc_10` | `#1:OK(0.642)` | `answered` (`ret=0`, `48.4s`) | `#1:H(E=0.28,C=0.23) -> #2:REFL -> #3:H(E=0.15,C=0.17)` | `max_hallucination_retries_exceeded` (`ret=2`, `117.0s`) |

## 4. Farklılaşan, Flaglenen ve Fallback Sorularının Cevap Kontrolü

| ID | Alt Kat. | Eski (v1+HHEM) İz & Sonuç | Yeni (v2+NLI) İz & Sonuç |
| :--- | :--- | :--- | :--- |
| `aiact_f01` | `direct_factoid` | `#1:H(0.175) -> #2:H(0.171)` → `max_hallucination_retries_exceeded` | `#1:H(E=0.25,C=0.02) -> #2:H(E=0.16,C=0.01)` → `max_hallucination_retries_exceeded` |
| `nis2_f02` | `direct_factoid` | `#1:H(0.289) -> #2:H(0.289)` → `max_hallucination_retries_exceeded` | `#1:OK(E=0.74,C=0.18)` → `answered` |
| `hipaa_f01` | `direct_factoid` | `#1:OK(0.725)` → `answered` | `#1:H(E=0.27,C=0.62) -> #2:H(E=0.42,C=0.46)` → `max_hallucination_retries_exceeded` |
| `gdpr_lm03` | `lexical_mismatch` | `#1:H(0.273) -> #2:H(0.227)` → `max_hallucination_retries_exceeded` | `#1:OK(E=0.63,C=0.05)` → `answered` |
| `hipaa_lm01` | `lexical_mismatch` | `#1:OK(0.430)` → `answered` | `#1:H(E=0.01,C=0.01) -> #2:H(E=0.21,C=0.10)` → `max_hallucination_retries_exceeded` |
| `aiact_mh03` | `intra_doc_multi_hop` | `#1:H(0.141) -> #2:H(0.355)` → `max_hallucination_retries_exceeded` | `#1:REFL -> #2:REF` → `clean_refusal` |
| `nis2_mh02` | `intra_doc_multi_hop` | `#1:H(0.289) -> #2:H(0.256)` → `max_hallucination_retries_exceeded` | `#1:OK(E=0.71,C=0.01)` → `answered` |
| `xdoc_01` | `cross_doc_multi_hop` | `#1:H(0.239) -> #2:H(0.210)` → `max_hallucination_retries_exceeded` | `#1:H(E=0.32,C=0.08) -> #2:H(E=0.28,C=0.08)` → `max_hallucination_retries_exceeded` |
| `xdoc_02` | `cross_doc_multi_hop` | `#1:H(0.265) -> #2:H(0.213)` → `max_hallucination_retries_exceeded` | `#1:H(E=0.31,C=0.00) -> #2:H(E=0.36,C=0.02)` → `max_hallucination_retries_exceeded` |
| `xdoc_04` | `cross_doc_multi_hop` | `#1:OK(0.542)` → `answered` | `#1:H(E=0.27,C=0.11) -> #2:H(E=0.24,C=0.18)` → `max_hallucination_retries_exceeded` |
| `xdoc_06` | `cross_doc_multi_hop` | `#1:H(0.376) -> #2:REF` → `clean_refusal` | `#1:H(E=0.24,C=0.01) -> #2:REFL -> #3:H(E=0.28,C=0.01)` → `max_hallucination_retries_exceeded` |
| `xdoc_09` | `cross_doc_multi_hop` | `#1:H(0.178) -> #2:H(0.125)` → `max_hallucination_retries_exceeded` | `#1:H(E=0.04,C=0.25) -> #2:REFL -> #3:REF` → `clean_refusal` |
| `xdoc_10` | `cross_doc_multi_hop` | `#1:OK(0.501)` → `answered` | `#1:H(E=0.28,C=0.23) -> #2:REFL -> #3:H(E=0.15,C=0.17)` → `max_hallucination_retries_exceeded` |
| `nis2_syc03` | `false_premise` | `#1:H(0.278) -> #2:H(0.165)` → `max_hallucination_retries_exceeded` | `#1:OK(E=0.82,C=0.04)` → `answered` |
| `kvkk_syc02` | `false_premise` | `#1:OK(0.578)` → `answered` | `#1:H(E=0.38,C=0.16) -> #2:OK(E=0.45,C=0.03)` → `answered` |
| `fb_id01` | `in_domain_unanswerable` | `#1:REF` → `clean_refusal` | `#1:REF` → `clean_refusal` |
| `fb_id02` | `in_domain_unanswerable` | `#1:OK(0.543)` → `clean_refusal` | `#1:REFL -> #2:REF` → `clean_refusal` |
| `fb_id04` | `in_domain_unanswerable` | `#1:REF` → `clean_refusal` | `#1:REFL -> #2:REF` → `clean_refusal` |
| `fb_id05` | `in_domain_unanswerable` | `#1:REF` → `clean_refusal` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` |
| `fb_ood01` | `out_of_domain` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` |
| `fb_ood02` | `out_of_domain` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` |
| `fb_ood03` | `out_of_domain` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` | `Verifier çalışmadı (no_relevant_documents)` → `no_relevant_documents` |
