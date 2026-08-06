"""telegram_bot.py — thin Telegram adapter (imports pipeline, no business logic)."""
import asyncio
import os
import sys
from pathlib import Path
import traceback

import nest_asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import process_request

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # Replace with your actual token or set it as an
# Allow nested event loops for Colab compatibility
nest_asyncio.apply()
# In telegram_bot.py, replace the handler with this:
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    print(f"Received: {user_text}")
    try:
        bot_response = await asyncio.to_thread(process_request, user_text)
        print(f"Response: {bot_response[:200]}...")
        await context.bot.send_message(chat_id=update.effective_chat.id, text=bot_response)
    except Exception:
        traceback.print_exc()
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Error! Check your terminal.")
async def run_bot():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    text_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    application.add_handler(text_handler)

    print("Telegram Bot is starting...")

    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    print("Bot is now polling. Send a message on Telegram!")


if __name__ == '__main__':
    try:
        # Get the existing loop and run the bot
        loop = asyncio.get_event_loop()
        loop.create_task(run_bot())
    except Exception as e:
        print(f"Error: {e}")