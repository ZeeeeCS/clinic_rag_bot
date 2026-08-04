"""telegram_bot.py — thin Telegram adapter (imports pipeline, no business logic)."""
import asyncio
import os

import nest_asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

from main import process_request

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN", "")

# Allow nested event loops for Colab compatibility
nest_asyncio.apply()

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message and update.message.text:
        user_text = update.message.text
        bot_response = process_request(user_text)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=bot_response)

async def run_bot():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    text_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    application.add_handler(text_handler)

    print("Telegram Bot is starting...")
    # Use initialize/start instead of run_polling for better control in notebooks
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