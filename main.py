"""Single-query CLI runner for streaming node-by-node graph execution traces."""

import os
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

import asyncio
import logging
from src.graph import build_graph

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# --- Single-Query Streaming Runner ---
async def run_test():
    """Compile the graph and stream state updates for a sample compliance query."""
    # 1. Build the Agentic RAG workflow
    app = build_graph()

    # 2. Initialize GraphState with an in-corpus sample question
    test_question = "What is the maximum statutory timeframe in hours for a data controller to notify a personal data breach to the supervisory authority under Article 33?"
    
    initial_state = {
        "original_question": test_question,
        "question": test_question,
        "documents": [],
        "sources": [],
        "generation": None,
        "is_relevant": None,
        "has_hallucination": None,
        "hallucination_reasoning": None,
        "query_rewrite_retry_count": 0,
        "generation_hallucination_retry_count": 0,
        "audit_logs": [],
    }

    logger.info("Starting graph execution with query: '%s'", test_question)

    # 3. Stream node updates to inspect decisions and audit logs
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
    print(f"""
          *********************************************************************************
          *********************************************************************************
          *********************************************************************************
          *********************************************************************************
          """)
    asyncio.run(run_test())