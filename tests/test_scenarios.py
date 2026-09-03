import json
import pytest
from src.graph import build_graph
from src.state.state import GraphState


@pytest.fixture(scope="session")
def graph_app():
    return build_graph()


def _format_execution_trace(state: GraphState) -> str:
    # Tum node gecmisini ve reasoning adimlarini okunabilir formatta toplar
    trace_lines = ["\n" + "=" * 30 + " EXECUTION TRACE " + "=" * 30]
    
    for idx, log in enumerate(state.get("audit_logs", [])):
        node_name = log.get("node", "unknown")
        trace_lines.append(f"[{idx + 1}] NODE: {node_name}")
        
        if node_name == "generate_node":
            trace_lines.append(f"    - Retry Mode: {log.get('used_retry_notes')}")
            trace_lines.append(f"    - Generation Length: {log.get('generated_char_length')}")
            trace_lines.append(f"    - Text: {log.get('generated_text')}")
        elif node_name == "grade_hallucination_node":
            trace_lines.append(f"    - Has Hallucination: {log.get('has_hallucination')}")
            trace_lines.append(f"    - Reasoning: {log.get('reasoning')}")
        
        trace_lines.append("-" * 40)
        
    trace_lines.append(f"FINAL GENERATION: {state.get('generation')}")
    trace_lines.append(f"FINAL RETRY COUNT: {state.get('generation_hallucination_retry_count')}")
    trace_lines.append(f"FINAL HALLUCINATION FLAG: {state.get('has_hallucination')}")
    trace_lines.append("=" * 77)
    
    return "\n".join(trace_lines)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_query, expected_must_contain_keyword, expect_retrieval_success",
    [
        # Senaryo 1: Moria kapisi parolasi
        ("what exact words did gandalf say to the balrog on the bridge of khazad-dum", None, True),
        ("what exact words did the witch-king say right before eowyn killed him", None, True),
        ("how many balrogs fought in the siege of gondolin according to the text", None, True),
        ("what gift did galadriel give to boromir in lothlorien", None, True),
        ("what color was the feather in tom bombadil's hat", None, True),
        ("what was the name of the horse that legolas rode during the war", None, True),
        ("where was saruman killed and who dealt the final blow", None, True),
        ("which hand and finger did gollum bite off from frodo at mount doom", None, True),
        ("why did elrond refuse to let aragorn marry arwen in rivendell", None, True),
        # # Senaryo 2: Film bilgisi / Lurtz tuzagi
        # ("how did isildur lose the ring and get killed in the anduin river", None, True),
        # # Senaryo 3: Net sayisal yas bilgisi (111 / eleventy-one)
        # ("how old bilbo baggins was when he give party and left shire", "111", True),
        # # Senaryo 4: Gollum dis varlik / Smeagol kontrolu
        # ("what happened to the knife that wounded frodo on weathertop", None, True),
        # # Senaryo 5: Kapsam disi Silmarillion sorgusu (Retrieval basarisiz olmali veya guvenli ret vermeli)
        # ("who reforged the shards of narsil into anduril and when", None, False),
    ],
)
async def test_agentic_rag_pipeline(
    graph_app,
    raw_query: str,
    expected_must_contain_keyword: str,
    expect_retrieval_success: bool,
):
    initial_state: GraphState = {
        "original_question": raw_query,
        "question": raw_query,
        "documents": [],
        "generation": None,
        "is_relevant": None,
        "has_hallucination": None,
        "retrieval_retry_count": 0,
        "generation_hallucination_retry_count": 0,
        "hallucination_reasoning": None,
        "audit_logs": [],
    }

    final_state: GraphState = await graph_app.ainvoke(initial_state)
    print(_format_execution_trace(final_state))
    print(f"\n[GENERATION]: {final_state.get('generation')}")
    trace_dump: str = _format_execution_trace(final_state)

    # 1. Uretim bos olmamali
    assert final_state["generation"] is not None and len(final_state["generation"].strip()) > 0, (
        f"Empty generation returned.\nTrace:{trace_dump}"
    )

    # 2. Retrieval esik kontrolu
    assert final_state["is_relevant"] == expect_retrieval_success, (
        f"Retrieval relevance mismatch. Expected: {expect_retrieval_success}\nTrace:{trace_dump}"
    )

    # 3. Sonsuz dongu korumasi
    assert final_state["generation_hallucination_retry_count"] <= 2, (
        f"Retry count exceeded limit.\nTrace:{trace_dump}"
    )

    # 4. Halusinasyon dogrulamasi
    assert final_state["has_hallucination"] is False, (
        f"Pipeline ended with ungrounded/hallucinated answer!\nTrace:{trace_dump}"
    )

    # 5. Anahtar kelime kontrolu
    if expected_must_contain_keyword:
        assert expected_must_contain_keyword.lower() in final_state["generation"].lower(), (
            f"Expected keyword '{expected_must_contain_keyword}' missing.\nTrace:{trace_dump}"
        )