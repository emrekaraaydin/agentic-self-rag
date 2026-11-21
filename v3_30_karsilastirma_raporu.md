# Agentic RAG v3 Değerlendirme Raporu (`n = 30` Katmanlı Örneklem — 4 Sistem Kıyaslaması)

Bu rapor, [`golden_dataset_100.json`](file:///Users/emrekaraaydin/Desktop/agentic/rag/golden_dataset_100.json) içerisinden `random.Random(42)` ile katmanlı ve rastgele seçilen **30 soru** ([`golden_dataset_30_v3.json`](file:///Users/emrekaraaydin/Desktop/agentic/rag/golden_dataset_30_v3.json)) üzerinde aşağıdaki 4 sistemin gerçek `ground_truth` karşılaştırmasını sunmaktadır:
- **Naive RAG** ([`naive_rag_evaluation_results.json`](file:///Users/emrekaraaydin/Desktop/agentic/rag/naive_rag_evaluation_results.json))
- **Agentic v1 (Whole HHEM-2.1)** ([`evaluation_results.json`](file:///Users/emrekaraaydin/Desktop/agentic/rag/evaluation_results.json))
- **Agentic v2 (Decompose + Reflect + ModernBERT-NLI)** ([`evaluation_results_nli_100.json`](file:///Users/emrekaraaydin/Desktop/agentic/rag/evaluation_results_nli_100.json))
- **Agentic v3 (Hibrit BM25+Dense + Dinamik Belge Filtresi + BGE-M3 + Cümle Bazlı NLI)** ([`evaluation_results_v3_30.json`](file:///Users/emrekaraaydin/Desktop/agentic/rag/evaluation_results_v3_30.json))

---

## 1. v3 Mimari Yenilikleri (Genel / Soru-Bağımsız)

1. **Çevrimdışı Deterministik BM25 + Qdrant Hibrit Arama ([`sparse.py`](file:///Users/emrekaraaydin/Desktop/agentic/rag/src/indexing/sparse.py), [`retrieve_node.py`](file:///Users/emrekaraaydin/Desktop/agentic/rag/src/nodes/retrieve_node.py)):** Madde/bölüm numaralarını (`83(5)`, `164.514(e)`, `10^25`) koruyan BM25 seyrek vektörleri (`Modifier.IDF`) ile `nomic-embed-text` yoğun vektörleri Qdrant `Fusion.RRF` ile birleştirildi.
2. **Otomatik PDF Üs Onarımı ve Belge Profili ([`loader.py`](file:///Users/emrekaraaydin/Desktop/agentic/rag/src/indexing/loader.py), [`seed.py`](file:///Users/emrekaraaydin/Desktop/agentic/rag/src/seed/seed.py)):** PyMuPDF font bayrakları (`flags & 1`) ve `<sup>` dönüşümüyle `10 25` → `10^25` otomatik onarıldı; 5 PDF için sabit sözlük olmadan [`qdrant_catalog.json`](file:///Users/emrekaraaydin/Desktop/agentic/rag/qdrant_catalog.json) çıkarıldı ve her parçaya `[Document: <Başlık> | Page <No>]` ön eki eklendi.
3. **Katalog Bilinçli Planlayıcı + Kaynak Filtresi ([`decompose_query_node.py`](file:///Users/emrekaraaydin/Desktop/agentic/rag/src/nodes/decompose_query_node.py)):** Yalnızca çoklu soruları alt sorgulara böldü ve açıkça eşleşen belgeler için `source` filtresi (`MatchAny`) uyguladı; OOD sorularda zorla eşleştirme yapmadı (`target_sources = []`).
4. **Uzun Bağlamlı Cross-Encoder ([`grade_retrieval_node.py`](file:///Users/emrekaraaydin/Desktop/agentic/rag/src/nodes/grade_retrieval_node.py)):** FlashRank MiniLM yerine `BAAI/bge-reranker-v2-m3` (`max_length=1024`, `MIN_DOCS=4`) kullanıldı; net-negatif `transform_query_node` ve `reflect_missing_context_node` döngüleri (`MAX_RETRY=0`) kapatıldı.
5. **Cümle Bazlı Çelişki-Öncelikli ModernBERT-NLI ([`grade_hallucination_node.py`](file:///Users/emrekaraaydin/Desktop/agentic/rag/src/nodes/grade_hallucination_node.py)):** `doc_title`/`source` ve kısmi-ret bypass hataları giderildi; `contra >= 0.40 & contra > entail`, `entail >= 0.30`, `supported_ratio >= 0.60` kuralı uygulandı.

---

## 2. Genel Karşılaştırma Tablosu (`n = 30`: `25` Kapsam İçi + `5` Fallback)

| Metrik | Naive RAG | Agentic v1 (HHEM) | Agentic v2 (NLI) | **Agentic v3 (Yeni)** |
| :--- | :---: | :---: | :---: | :---: |
| **Çalışan / Hatalı** | `30/30` (`0`) | `30/30` (`0`) | `30/30` (`0`) | `30/30` (`0`) |
| **Target Doc Hit Rate (`n=25`)** | `23/25` (`%92.0`) | `23/25` (`%92.0`) | `24/25` (`%96.0`) | **`25/25` (`%100.0`)** |
| **Context Source Purity (`n=25`)** | `%78.50` | `%73.00` | `%81.90` | **`%98.00`** |
| **Ort. Bağlam Sayısı** | `8.00` | `7.20` | `5.63` | `6.77` |
| **Tam Doğru (`n=25`)** | **`16/25` (`%64.0`)** | `15/25` (`%60.0`) | `15/25` (`%60.0`) | `14/25` (`%56.0`) |
| **Kısmen Doğru / Uydurmasız (`n=25`)** | `0/25` | `1/25` (`%4.0`) | **`3/25` (`%12.0`)** | `2/25` (`%8.0`) |
| **Toplam Faydalı (`Doğru + Kısmi`)** | `16/25` (`%64.0`) | `16/25` (`%64.0`) | **`18/25` (`%72.0`)** | `16/25` (`%64.0`) |
| **Gerçek Yanlış / Halüsinasyon (`n=25`)** | `7/25` (`%28.0`) | `3/25` (`%12.0`) | **`1/25` (`%4.0`)** | `5/25` (`%20.0`) |
| **Gereksiz Ret / Fallback (`n=25`)** | **`2/25`** (`2` ret / `0` FB) | `6/25` (`1` ret / `5` FB) | `6/25` (`1` ret / `5` FB) | `4/25` (`0` ret / `4` FB) |
| **Cevap Verilenlerde Güvenilirlik** | `16/23` (`%69.6`) | `16/19` (`%84.2`) | **`18/19` (`%94.7`)** | `16/21` (`%76.2`) |
| **Fallback Güvenliği (`n=5`: Gerçek / Yapısal OOD)** | `5/5` (`0/2` yapısal) | `5/5` (`2/2` yapısal) | `5/5` (`2/2` yapısal) | **`5/5` (`2/2` yapısal)** |
| **Ort. / Medyan / P95 Süre** | **`35.4s` / `32.2s` / `66.8s`** | `44.1s` / `43.0s` / `74.9s` | `58.8s` / `55.1s` / `140.8s` | `73.3s` / `65.9s` / `137.0s` |

---

## 3. Alt Kategorilere Göre Dağılım (`D / K / H / R`)

| Alt Kategori (`n`) | Naive (`D/K/H/R`) | v1 (`D/K/H/R`) | v2 (`D/K/H/R`) | **v3 (`D/K/H/R`)** |
| :--- | :---: | :---: | :---: | :---: |
| `direct_factoid` (`4`) | `4 / 0 / 0 / 0` | `4 / 0 / 0 / 0` | `4 / 0 / 0 / 0` | `4 / 0 / 0 / 0` |
| `lexical_mismatch` (`5`) | `4 / 0 / 1 / 0` | `3 / 0 / 0 / 2` | `4 / 0 / 0 / 1` | `4 / 0 / 0 / 1` |
| `intra_doc_multi_hop` (`6`) | `3 / 0 / 3 / 0` | `3 / 1 / 2 / 0` | `2 / 3 / 0 / 1` | `3 / 1 / 2 / 0` |
| `cross_doc_multi_hop` (`5`) | `0 / 0 / 3 / 2` | `1 / 0 / 1 / 3` | `0 / 0 / 1 / 4` | `0 / 1 / 3 / 1` |
| `false_premise` (`5`) | `5 / 0 / 0 / 0` | `4 / 0 / 0 / 1` | `5 / 0 / 0 / 0` | `3 / 0 / 0 / 2` |

*(**D**: Tam Doğru, **K**: Kısmen Doğru, **H**: Yanlış/Halüsinasyon, **R**: Gereksiz Ret/Fallback)*

---

## 4. Çıkarım: Neler Büyük Ölçüde İyileşti, Neler Açık Verdi?

### Retrieval Katmanı Beklentinin Üstünde Başarılı Oldu
- **%100 Hedef Doküman İsabeti (`25/25`) ve %98.00 Bağlam Saflığı:** Çapraz-kanun kirliliği (örn. GDPR sorusuna NIS2/AI Act gelmesi, KVKK sorusuna NIS2 gelmesi) tamamen bitti.
- **`gdpr_lm02` Çözüldü:** Hem v1 hem v2'de `no_relevant_documents` hatasına düşen `<250 employees` sorusu v3'te sayfa 51'den getirildi ve 3 istisnasıyla birlikte tam doğru cevaplandı.
- **`kvkk_mh01` ve `xdoc_04` İyileşti:** `kvkk_mh01` tam 30/60/30 gün zinciriyle eksiksiz cevaplandı; `xdoc_04` NLI retry ile GDPR `20M/4%`, NIS2 `10M/2%` ve `Art 35(2)` çifte ceza yasağını doğru verdi.
- **OOD Güvenliği Hızlı ve Temiz:** `fb_ood05` ve `fb_ood07` soruları `target_sources=[]` ile filtrelenmeden arandı, reranker `0` parça geçirdi ve `13.3–17.7s` içinde deterministik olarak reddedildi.

### Sonuç Skorunu Aşağı Çeken 3 Somut Sebep (Hata Analizi)
1. **Cümle Bazlı NLI'ın 4 Doğru Cevabı Bozması veya Öldürmesi:**
   - **`hipaa_syc03`:** Taslak 0 ve 1 tamamen doğruyken NLI, `"No, that is not correct."` (4 kelimelik ret cümlesi) için `contra=0.648` verip doğru cevabı fallback'e düşürdü (çünkü bağlamda soru yok, yalnızca olumlu kural var).
   - **`hipaa_syc02`:** Doğru `"The rule explicitly excludes employment records..."` cümlesine `contra=0.726` verdi (2002 Final Rule metnindeki `"proposed to modify"` tartışmasından dolayı).
   - **`aiact_lm02`:** 2 cümlelik doğru cevabın 2. cümlesine `entail=0.228` verdiği için `supported_ratio=0.50 < 0.60` kuralından elendi.
   - **`xdoc_10`:** İlk taslak **%100 doğruydu**, NLI `"a prior application to the data controller is mandatory"` cümlesini `contra=0.724` ile reddedince (KVKK metni `"shall not contact the Board before exhausting..."` yazdığı için), yeniden üretim cümleyi `"not explicitly stated as mandatory"` şeklinde bozup halüsinasyona çevirdi.
2. **7B Üretici LLM'in (`qwen2.5:7b-instruct`) Saf Bağlamda Bile Madde Numarası Karıştırması:**
   - **`gdpr_mh01`** (`%100` GDPR sayfa 82–83 bağlamında `83(4)` ve `83(5)` cezalarını ters çevirdi), **`gdpr_mh02`** (Art 13 ve 14 zamanlamasını birleştirdi), **`xdoc_09`** (AI Act Art 86 içeriğini `"Article 50"` etiketiyle yazdı) ve **`xdoc_15`** (genel KVKK Art 5(2) şartlarını sağlık verisine karıştırdı). Yeni NLI kuralı `contra < 0.40` altında kaldığı için bu 4 hatayı geçirmiş oldu (v2'nin katı `E >= 0.40` eşiği ise bunların 3'ünü redde dönüştürmüştü).
3. **Gecikme (`73.3s`):** `BAAI/bge-reranker-v2-m3` (~560M) + `ModernBERT-large-nli` (~395M) + `qwen2.5:7b` modellerinin Apple Silicon birleşik belleğinde sırayla çalışması ve `12` retry turu ortalama süreyi v2'nin (`58.8s`) üzerine çıkardı.
