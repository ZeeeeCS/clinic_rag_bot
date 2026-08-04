"""main.py — Clinic RAG Bot application entry point."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for path in (ROOT_DIR, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


def get_logger() -> logging.Logger:
    logger = logging.getLogger("clinic_rag_bot")
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    return logger


logger = get_logger()


def _build_safety_agent():
    from src.safety_agent import SafetyAgent

    return SafetyAgent()


def _build_retriever():
    try:
        from src.pubmed import GeneralKnowledgeRetriever
    except Exception as exc:
        logger.warning("General knowledge retriever is unavailable: %s", exc)
        return None

    return GeneralKnowledgeRetriever()


def process_request(user_text: str) -> str:
    if not user_text or not isinstance(user_text, str):
        return "Please provide a health-related question or symptom description."

    user_text = user_text.strip()
    if not user_text:
        return "Please provide a health-related question or symptom description."

    safety_agent = _build_safety_agent()
    safety_result = safety_agent.run(user_text)
    if "Immediate attention required" in safety_result:
        return safety_result

    try:
        retriever = _build_retriever()
        if retriever is not None:
            documents = retriever.retrieve(user_text, top_k=3, confidence_threshold=0.3)
            if documents:
                evidence = "; ".join(
                    f"{item.get('source') or 'source'}: {str(item.get('text', ''))[:180]}"
                    for item in documents[:2]
                )
                return (
                    "I can help with general medical information. "
                    f"Relevant context: {evidence}"
                )
    except Exception as exc:
        logger.warning("Retrieval failed for request '%s': %s", user_text, exc)

    return (
        "I can help with general medical information. "
        "Please share more detail about your symptoms or question."
    )


def run_telegram_bot() -> None:
    try:
        from bot.telegram_bot import run_bot as telegram_run_bot
    except Exception as exc:
        logger.error("Unable to start the Telegram bot: %s", exc)
        raise

    asyncio.run(telegram_run_bot())


def main() -> None:
    parser = argparse.ArgumentParser(description="Clinic RAG Bot entry point")
    parser.add_argument(
        "--query",
        help="Process a single user query and print the response",
    )
    parser.add_argument("--bot", action="store_true", help="Start the Telegram bot")
    args = parser.parse_args()

    if args.bot:
        logger.info("Starting Telegram bot...")
        run_telegram_bot()
        return

    if args.query:
        print(process_request(args.query))
        return

    print(
        "Clinic RAG Bot ready. Use --query 'your question' or --bot to start Telegram."
    )


if __name__ == "__main__":
    main()
