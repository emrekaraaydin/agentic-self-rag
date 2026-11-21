# Agentic RAG Assistant

This interface queries the local, self-reflective **Agentic RAG (CRAG + Self-RAG)** pipeline across five legal and regulatory compliance documents:

- **EU GDPR (`CELEX_32016R0679_EN_TXT.pdf`)**: General Data Protection Regulation
- **EU AI Act (`OJ_L_202401689_EN_TXT.pdf`)**: Regulation (EU) 2024/1689 on Artificial Intelligence
- **EU NIS2 Directive (`CELEX_32022L2555_EN_TXT.pdf`)**: Network and Information Security Directive
- **U.S. HIPAA (`privrulepd.pdf`)**: 2002 HIPAA Privacy Rule Final Rule
- **Turkey KVKK (`kvkk_law_6698_en.pdf.pdf`)**: Law No. 6698 on the Protection of Personal Data (English translation)

### Architecture Highlights
- **Query Planning & Retrieval:** `qwen2.5:7b-instruct` query decomposition + `nomic-embed-text` dense retrieval (Qdrant Top-50) + **FlashRank** cross-encoder reranking (`0.03` / `0.05`).
- **Hallucination Verification:** Whole-answer **ModernBERT-large-nli** verification (`entailment >= 0.40`, `contradiction < 0.25`) with deterministic fallback on unverified or out-of-domain queries.
- **Verifiable Sources:** Click inline `[1]`, `[2]` links to open the exact PDF page in the side panel, or click **Show Retrieved Chunks** to inspect chunk text with matched passages highlighted in bold.
