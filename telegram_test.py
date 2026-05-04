"""
Test script to connect to a Telegram bot and send a test message.
Requires TELEGRAM_BOT_TOKEN in the environment (see .env.example).
"""
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
if not BOT_TOKEN:
    print("Set TELEGRAM_BOT_TOKEN in your .env (see .env.example).", file=sys.stderr)
    sys.exit(1)

BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"


def get_me():
    """Verify bot token and get bot info."""
    r = requests.get(f"{BASE_URL}/getMe")
    data = r.json()
    if not data.get("ok"):
        print("Connection failed:", data.get("description", "Unknown error"))
        return None
    print("Bot connected:", data["result"]["username"])
    return data["result"]


def get_chat_id():
    """Get the most recent chat_id from updates (user must have messaged the bot first)."""
    r = requests.get(f"{BASE_URL}/getUpdates")
    data = r.json()
    if not data.get("ok"):
        print("Failed to get updates:", data.get("description"))
        return None
    updates = data.get("result", [])
    if not updates:
        print("No updates found. Send any message to your bot first, then run this script again.")
        return None
    last = updates[-1]
    msg = last.get("message") or last.get("edited_message") or {}
    chat = msg.get("chat") or {}
    cid = chat.get("id")
    if cid is None:
        print("Could not find chat_id in last update.")
        return None
    print("Using chat_id:", cid)
    return cid


def send_test_message(chat_id: int | str, text: str = "Hello from CS_Dashboard telegram_test.py") -> None:
    r = requests.post(
        f"{BASE_URL}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=30,
    )
    data = r.json()
    if not data.get("ok"):
        print("sendMessage failed:", data.get("description", data))
        return
    print("Message sent OK.")


def main():
    me = get_me()
    if not me:
        return
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if chat_id:
        send_test_message(chat_id.strip())
        return
    cid = get_chat_id()
    if cid is not None:
        send_test_message(cid)


if __name__ == "__main__":
    main()
