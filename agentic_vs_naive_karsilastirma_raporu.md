# Agentic RAG vs. Naive RAG Karşılaştırmalı Değerlendirme Raporu (100 Soruluk Literatür Benchmarkı)

Bu rapor, [`data/`](file:///Users/emrekaraaydin/Desktop/agentic/rag/data) dizinindeki güncel 5 yasal/regülasyon PDF'i (EU GDPR, EU AI Act, EU NIS2, US HIPAA 2002 Final Rule, Türkiye KVKK 6698) üzerinde [`golden_dataset_100.json`](file:///Users/emrekaraaydin/Desktop/agentic/rag/golden_dataset_100.json) kullanılarak yürütülen **Agentic RAG** ([`evaluate_batch.py`](file:///Users/emrekaraaydin/Desktop/agentic/rag/evaluate_batch.py) → [`evaluation_results.json`](file:///Users/emrekaraaydin/Desktop/agentic/rag/evaluation_results.json)) ve **Naive RAG** ([`naive_rag.py`](file:///Users/emrekaraaydin/Desktop/agentic/rag/naive_rag.py) → [`naive_rag_evaluation_results.json`](file:///Users/emrekaraaydin/Desktop/agentic/rag/naive_rag_evaluation_results.json)) koşularının karşılaştırmalı sonuçlarını sunmaktadır.

RAGAS dış değerlendirmesi için 80 kapsam içi sorudan oluşan dosyalar dışa aktarılmıştır: [`evaluation_results_ragas_80.json`](file:///Users/emrekaraaydin/Desktop/agentic/rag/evaluation_results_ragas_80.json) ve [`naive_rag_evaluation_results_ragas_80.json`](file:///Users/emrekaraaydin/Desktop/agentic/rag/naive_rag_evaluation_results_ragas_80.json).

## 1. Genel Sistem Başarısı ve Güvenlik Karşılaştırması (100 Soru)

| Metrik | Agentic RAG | Naive RAG | Fark / Yorum |
| :--- | :---: | :---: | :--- |
| **Tamamlanan / Hatalı** | `100/100` (`0` hata) | `100/100` (`0` hata) | Her iki sistem tam koşu |
| **Ort. Gecikme (Tüm / Warm-up hariç)** | `40.89s` / `41.04s` | `35.78s` / `35.99s` | Medyan: `36.80s` vs `32.17s` |
| **Gecikme (Min – P95 – Max)** | `8.99s` – `74.87s` – `149.10s` | `12.03s` – `66.80s` – `84.79s` | Fallback erken çıkış vs Retry maliyeti |
| **Target Doc Hit Rate (`n=80` kapsam içi)** | `70/80` (**`%87.50`**) | `73/80` (**`%91.25`**) | Cross-doc çiftlerde her iki PDF şartı dahil |
| **Context Source Purity (`n=80`)** | **`%83.91`** | `%85.47` | `-1.56` puan bağlam saflığı |
| **Ort. Bağlam Parçası (Tüm / Kapsam İçi)** | `7.20` / `7.90` | `8.00` / `8.00` | Dinamik budama vs sabit `k=8` |
| **Yapısal `fallback_node` Recall (`n=20`)** | `9/20` (`%45.00`) | `0/20` (`%0.00`) | Deterministik koruma yalnızca Agentic'te |
| **Semantik Ret / Abstention (`n=20`)** | `19/20` (`%95.00`) | `20/20` (`%100.00`) | Yapısal fallback + temiz ret ifadeleri |
| **Cevaplanamaz Soruda Halüsinasyon/Sızıntı (`n=20`)** | `1/20` (`%5.00`) | `0/20` (`%0.00`) | Cevap uyduran veya parametrik sızıntı yapanlar |
| **In-Domain False-Positive Fallback (`n=80`)** | `10/80` (Toplam ret: `22`) | `0/80` (Toplam ret: `15`) | Kapsam içi soruda gereksiz ret |
| **Sycophancy Yanlış Ön Kabule Direnç (`n=15`)** | `12/15` (`%80.00`) | `15/15` (`%100.00`) | Yanlış önermeyi açıkça düzelten yanıtlar |
| **Vectara HHEM-2.1 Ort. / Medyan (Pozitif Yanıtlar)** | `0.5247` / `0.5803` (`n=90`) | `0.6942` / `0.7300` (`n=65`) | `<0.40` kalan: Agentic `30` vs Naive `9` |

## 2. Literatür Boyutlarına (Subcategory) Göre Ayrıntılı Karşılaştırma

| Kategori / Alt Kategori | Soru | Agentic Hit | Naive Hit | Agentic Purity | Naive Purity | Agentic Ort. Süre | Naive Ort. Süre |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Factoid — Direct Factoid (LegalBench-RAG)** | `15` | `15/15` (`%100.0`) | `15/15` (`%100.0`) | `%85.8` | `%83.3` | `26.13s` | `26.74s` |
| **Factoid — Lexical Mismatch (CRAG / BEIR)** | `15` | `14/15` (`%93.3`) | `15/15` (`%100.0`) | `%79.2` | `%84.2` | `28.17s` | `29.95s` |
| **Multi-Hop — Intra-Doc (RGB / HotpotQA)** | `20` | `20/20` (`%100.0`) | `20/20` (`%100.0`) | `%81.9` | `%81.9` | `54.77s` | `52.22s` |
| **Multi-Hop — Cross-Doc 2/2 (MultiHop-RAG)** | `15` | `6/15` (`%40.0`) (≥1/2: `15` vs `15`) | `8/15` (`%53.3`) | `%86.7` | `%91.7` | `61.98s` | `41.96s` |
| **Sycophancy — False Premise (RGB)** | `15` | `15/15` (`%100.0`) | `15/15` (`%100.0`) | `%86.7` | `%87.5` | `46.68s` | `33.44s` |

### Negatif Ret (Fallback) Alt Kırılımı (`n=20`)

| Alt Kategori | Soru | Agentic Yapısal Fallback | Agentic Semantik Ret | Naive Yapısal Fallback | Naive Semantik Ret |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Out-of-Domain (`out_of_domain`)** | `8` | `8/8` (`%100.0`) | `8/8` (`%100.0`) | `0/8` | `8/8` (`%100.0`) |
| **In-Domain Hard Negative (`in_domain_unanswerable`)** | `12` | `1/12` (`%8.3`) | `11/12` (`%91.7`) | `0/12` | `12/12` (`%100.0`) |

## 3. PDF Kaynağına Göre Tekil Hedef Doküman Başarısı (`n=65` tekil hedefli soru)

| PDF Dosyası | Soru | Agentic Hit | Naive Hit | Agentic Purity | Naive Purity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `CELEX_32016R0679_EN_TXT.pdf` | `13` | `12/13` (`%92.3`) | `13/13` (`%100.0`) | `%66.3` | `%65.4` |
| `CELEX_32022L2555_EN_TXT.pdf` | `13` | `13/13` (`%100.0`) | `13/13` (`%100.0`) | `%92.3` | `%86.5` |
| `OJ_L_202401689_EN_TXT.pdf` | `13` | `13/13` (`%100.0`) | `13/13` (`%100.0`) | `%92.3` | `%98.1` |
| `kvkk_law_6698_en.pdf.pdf` | `13` | `13/13` (`%100.0`) | `13/13` (`%100.0`) | `%65.4` | `%70.2` |
| `privrulepd.pdf` | `13` | `13/13` (`%100.0`) | `13/13` (`%100.0`) | `%100.0` | `%100.0` |

## 4. Agentic Bileşen Telemetrisi (`transform_query_node` & `grade_hallucination_node`)

- **Query Rewriter (`qwen2.5:3b-instruct`):** Toplam `13` soruda tetiklendi (`{'lexical_mismatch': 1, 'cross_doc_multi_hop': 1, 'out_of_domain': 8, 'in_domain_unanswerable': 3}`).
  - Kapsam içi sorguda kurtarma (`target_doc_hit=True` ve fallback yok): `0` → `[]`
  - Fallback sorusunda Reranker eşiğinden sızma: `2` → `['fb_id04', 'fb_id05']`
- **Vectara HHEM-2.1 Halüsinasyon Denetimi (`threshold=0.40`):** Toplam `10` soruda yeniden üretim (retry) tetiklendi.
  - Retry sonrası kurtarılanlar: `1` → `['xdoc_06']`
  - Maksimum retry aşımıyla güvenli fallback'e alınanlar: `9` → `['aiact_f01', 'nis2_f02', 'gdpr_lm03', 'aiact_mh03', 'nis2_mh02', 'xdoc_01', 'xdoc_02', 'xdoc_09', 'nis2_syc03']`

## 5. Hata ve Sızıntı İncelemesi (Örnek Vakalar)

- **Naive RAG Cevaplanamaz Soruda Sızıntı/Uydurma (`0` vaka):** []
- **Agentic RAG Cevaplanamaz Soruda Sızıntı/Uydurma (`1` vaka):** ['fb_id02']
  - **`fb_id02`** (`in_domain_unanswerable`): The provided documents do not contain a specific mathematical regression formula for the European Data Protection Board to calculate baseline monetary fines from a company's quarterly operating margin under GDPR Article 83. The documents outline the principles...

- **Kapsam İçi `target_doc_hit=False` Kayıtları:**
  - Agentic (`10`): `['gdpr_lm02', 'xdoc_03', 'xdoc_06', 'xdoc_07', 'xdoc_08', 'xdoc_11', 'xdoc_12', 'xdoc_13', 'xdoc_14', 'xdoc_15']`
  - Naive (`7`): `['xdoc_01', 'xdoc_06', 'xdoc_08', 'xdoc_12', 'xdoc_13', 'xdoc_14', 'xdoc_15']`
