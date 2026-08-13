"""telegram_bot.py — thin Telegram adapter (imports pipeline, no business logic)."""
import asyncio
import os
import sys
import traceback
from pathlib import Path

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import process_request
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_text = update.message.text
    print(f"Received: {user_text}")

    try:
        bot_response = await asyncio.to_thread(process_request, user_text)
        print(f"Response: {bot_response[:200]}...")
    except Exception:
        traceback.print_exc()
        bot_response = "Sorry, an error occurred while processing your request."

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=bot_response
    )


async def run_bot():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    text_handler = MessageHandler(
        filters.TEXT & (~filters.COMMAND),
        handle_message
    )
    application.add_handler(text_handler)

    print("Telegram Bot is starting...")

    await application.initialize()
    await application.start()
    await application.updater.start_polling(drop_pending_updates=True)
    print("Bot is now polling. Send a message on Telegram!")

    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nShutting down...")
        await application.updater.stop()
        await application.stop()
        await application.shutdown()


if __name__ == '__main__':
    asyncio.run(run_bot())