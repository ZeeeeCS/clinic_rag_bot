"""main.py — Clinic RAG Bot application entry point."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from bot import telegram_bot
from src.pipeline import ClinicRAGPipeline
from src.logger import Logger
import importlib
from dotenv import load_dotenv
load_dotenv()

# Initialize global components
logger = Logger()
rag_pipeline = ClinicRAGPipeline()

def process_request(user_text: str) -> str:
    """Processes a request using the established RAG pipeline."""
    if not user_text or not isinstance(user_text, str):
        return "Please provide a health-related question or symptom description."

    user_text = user_text.strip()
    if not user_text:
        return "Please provide a health-related question or symptom description."

    try:
        # The pipeline now handles safety, routing, retrieval, and generation internally
        return rag_pipeline.process(user_text)
    except Exception as e:
        logger.logger.error(f"Pipeline error: {e}")
        return "An error occurred while processing your request. Please try again later."

def run_telegram_bot() -> None:
    
    try:
        from bot.telegram_bot import run_bot as telegram_run_bot
        
        asyncio.run(telegram_run_bot())
    except Exception as exc:
        logger.logger.error("Unable to start the Telegram bot: %s", exc)
        raise

def main() -> None:
    parser = argparse.ArgumentParser(description="Clinic RAG Bot entry point")
    parser.add_argument("--query", help="Process a single user query")
    parser.add_argument("--bot", action="store_true", help="Start the Telegram bot")
    args = parser.parse_args()

    if args.bot:
        logger.logger.info("Starting Telegram bot...")
        run_telegram_bot()
        return

    if args.query:
        print(process_request(args.query))
        return

    print("Clinic RAG Bot ready. Use --query 'your question' or --bot to start Telegram.")

if __name__ == '__main__':
    main()