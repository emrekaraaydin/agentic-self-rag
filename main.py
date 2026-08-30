import asyncio
import logging
from src.graph import build_graph

# Loglama seviyesini ayarla (Düğümlerin ve router'ların loglarını görmek için)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def run_test():
    # 1. Grafiği derle
    app = build_graph()

    # 2. Başlangıç State'ini hazırla
    # Soru olarak bilerek eksik/belirsiz veya veri tabanında olan bir konu ver
    test_question = "who is the boromir"
    
    initial_state = {
        "original_question": test_question,
        "question": test_question,
        "documents": [],
        "generation": None,
        "is_relevant": None,
        "has_hallucination": None,
        "is_answer_satisfactory": None,
        "retrieval_retry_count": 0,
        "generation_hallucination_retry_count": 0,
        "generation_satisfaction_retry_count": 0,
        "audit_logs": [],
    }

    logger.info("Starting graph execution with query: '%s'", test_question)

    # 3. Grafiği streaming modunda çalıştır (Düğüm düğüm izleme)
    async for output in app.astream(initial_state):
        for node_name, state_update in output.items():
            print(f"\n{'='*20} NODE: {node_name} {'='*20}")
            for key, value in state_update.items():
                if key == "audit_logs":
                    print(f"-> [Audit Log]: {value[-1]}")
                elif key == "documents":
                    print(f"-> [Documents Count]: {len(value)}")
                else:
                    print(f"-> [{key}]: {value}")


if __name__ == "__main__":
    asyncio.run(run_test())