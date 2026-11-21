from app import (
    build_pdf_and_chunk_citation_plan,
    highlight_matching_chunk_passages,
)


AI_ACT_ART53_CHUNK = """EN

OJ L, 12.7.2024

2. The obligations set out in paragraph 1, points (a) and (b), shall not apply to providers of AI models that are released under a free and open-source licence that allows for the access, usage, modification, and distribution of the model, and whose parameters, including the weights, the information on the model architecture, and the information on model usage, are made publicly available. This exception shall not apply to general-purpose AI models with systemic risks.

3. Providers of general-purpose AI models shall cooperate as necessary with the Commission and the national competent authorities in the exercise of their competences and powers pursuant to this Regulation.

4. Providers of general-purpose AI models may rely on codes of practice within the meaning of Article 56 to demonstrate compliance with the obligations set out in paragraph 1 of this Article, until a harmonised standard is published. Compliance with European harmonised standards grants providers the presumption of conformity to the extent that the standards cover the obligations."""


def test_highlight_matching_chunk_passages_bolds_matched_and_keeps_others_plain():
    matched_para = (
        "The free and open-source exemption does not apply to general-purpose AI models "
        "with systemic risks, even if they are released under a free and open-source license "
        "and have publicly available parameters."
    )

    highlighted = highlight_matching_chunk_passages(AI_ACT_ART53_CHUNK, [matched_para])

    # Paragraph 2 should be highlighted in bold
    assert (
        "**2. The obligations set out in paragraph 1, points (a) and (b), shall not apply to providers of AI models that are released under a free and open-source licence that allows for the access, usage, modification, and distribution of the model, and whose parameters, including the weights, the information on the model architecture, and the information on model usage, are made publicly available.**"
        in highlighted
    )
    assert (
        "**This exception shall not apply to general-purpose AI models with systemic risks.**"
        in highlighted
    )
    # Non-matching paragraphs and metadata header must remain unbolded
    assert "\n3. Providers of general-purpose AI models shall cooperate" in highlighted
    assert "**3." not in highlighted
    assert "**4." not in highlighted
    assert "**EN**" not in highlighted
    assert highlighted.count("**") % 2 == 0


def test_highlight_matching_chunk_passages_general_context_untouched():
    untouched = highlight_matching_chunk_passages(AI_ACT_ART53_CHUNK, [])
    assert untouched == AI_ACT_ART53_CHUNK
    assert "**" not in untouched


def test_highlight_preserves_bullet_prefix_and_balances_existing_bold():
    chunk = (
        "## **Article 83**\n\n"
        "- (a) basic principles for processing, including conditions for consent under Article 5.\n"
        "- (b) miscellaneous unrelated procedural reporting timelines for supervisory audits."
    )
    para = "Fines up to 20,000,000 EUR apply to infringements of the basic principles for processing including conditions for consent."
    highlighted = highlight_matching_chunk_passages(chunk, [para])

    assert "\n- **(a) basic principles for processing, including conditions for consent under Article 5.**" in highlighted
    assert "\n- (b) miscellaneous unrelated" in highlighted
    assert highlighted.count("**") % 2 == 0


def test_build_pdf_and_chunk_citation_plan_deduplicates_pages_and_orders_matches():
    response = (
        "Providers of general-purpose AI models must draw up technical documentation under Article 53.\n\n"
        "The free and open-source exemption does not apply to general-purpose AI models with systemic risks."
    )
    sources = [
        # Two chunks from same page 86 (one matches P1, one matches nothing -> same PDF [1])
        {
            "chunk_id": "c1",
            "source": "OJ_L_202401689_EN_TXT.pdf",
            "page": 86,
            "rerank_score": 0.91,
            "content": "Article 53 obligations require providers of general-purpose AI models to draw up technical documentation.",
        },
        {
            "chunk_id": "c2",
            "source": "OJ_L_202401689_EN_TXT.pdf",
            "page": 46,
            "rerank_score": 0.89,
            "content": "Totally unrelated background recitals about digital market harmonization.",
        },
        {
            "chunk_id": "c3",
            "source": "OJ_L_202401689_EN_TXT.pdf",
            "page": 86,
            "rerank_score": 0.85,
            "content": "Further definitions regarding administrative procedures and national notifications.",
        },
        {
            "chunk_id": "c4",
            "source": "OJ_L_202401689_EN_TXT.pdf",
            "page": 85,
            "rerank_score": 0.80,
            "content": AI_ACT_ART53_CHUNK,
        },
    ]

    annotated, pdf_entries, chunk_entries = build_pdf_and_chunk_citation_plan(response, sources)

    # 3 unique (source, page) entries: p.86 (P1), p.85 (P2), p.46 (General Context)
    assert len(pdf_entries) == 3
    assert [(e["pdf_num"], e["page"], e["matched_paras"]) for e in pdf_entries] == [
        (1, 86, [1]),
        (2, 85, [2]),
        (3, 46, []),
    ]

    # Annotated paragraphs use deduplicated PDF indices
    assert "Article 53. [1]" in annotated
    assert "systemic risks. [2]" in annotated

    # Chunks keep matched first, General Context untouched
    assert len(chunk_entries) == 4
    assert [c["chunk_id"] for c in chunk_entries] == ["c1", "c4", "c2", "c3"]
    assert "**" in chunk_entries[1]["highlighted_content"]
    assert chunk_entries[2]["highlighted_content"] == sources[1]["content"]
    assert chunk_entries[3]["highlighted_content"] == sources[2]["content"]


def test_build_pdf_and_chunk_citation_plan_skips_refusal():
    refusal = "A verified answer could not be generated from the available context due to factual consistency constraints."
    sources = [{"chunk_id": "c1", "source": "CELEX_32016R0679_EN_TXT.pdf", "page": 83, "rerank_score": 0.9, "content": "Text"}]
    annotated, pdf_entries, chunk_entries = build_pdf_and_chunk_citation_plan(refusal, sources)

    assert annotated == refusal
    assert pdf_entries == []
    assert chunk_entries == []
