"""
Chat test: OpenAI-compatible for model list; LM Studio native API for chat when local
(so thinking + response stream correctly). Only prints output after prompt processing.
"""
import json
import os
import sys

import requests
from dotenv import load_dotenv
from openai import APIConnectionError, APIError, OpenAI

load_dotenv()

from hive_env import clear_hive_env_cache, get_hive_env

clear_hive_env_cache()
_he = get_hive_env()

# For local LM Studio: base without /v1, any placeholder API key
# For OpenAI: use "https://api.openai.com/v1" base and set HIVE_API_KEY
BASE_URL = (
    os.environ.get("HIVE_BASE_URL", "").strip()
    or os.environ.get("TEST_BASE_URL", "").strip()
    or _he.base_url
)
API_KEY = (
    os.environ.get("HIVE_API_KEY", "").strip()
    or os.environ.get("TEST_API_KEY", "").strip()
    or _he.api_key
)
MODEL_OVERRIDE = (os.environ.get("HIVE_MODEL") or os.environ.get("TEST_MODEL") or "").strip()
PROMPT = (
    os.environ.get("HIVE_TEST_PROMPT", "").strip()
    or os.environ.get("TEST_PROMPT", "").strip()
    or _he.test_prompt
    or "Say hello in one sentence."
)


def _is_lm_studio(base_url: str) -> bool:
    """True if this is a local LM Studio (or similar) server, not public OpenAI."""
    u = (base_url or "").strip().lower().rstrip("/")
    if "api.openai.com" in u:
        return False
    return True


def _server_base(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1")


def stream_chat_native(
    base_url: str, model: str, prompt: str, max_tokens: int = 15000, timeout: float = 360.0
) -> None:
    """Stream via LM Studio native API. Thinking + response; print only after prompt processing."""
    url = f"{_server_base(base_url)}/api/v1/chat"
    payload = {
        "model": model,
        "input": prompt,
        "stream": True,
        "max_output_tokens": max_tokens,
    }
    resp = requests.post(url, json=payload, stream=True, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = "utf-8"

    event_type = None
    prompt_done = False
    thinking_started = False
    response_started = False
    for line in resp.iter_lines(decode_unicode=True):
        if line is None:
            continue
        if line.startswith("event:"):
            event_type = line[6:].strip()
            continue
        if line.startswith("data:"):
            data_str = line[5:].strip()
            if data_str == "[DONE]" or not data_str:
                continue
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                event_type = None
                continue
            chunk_type = data.get("type") or event_type
            if chunk_type == "prompt_processing.end":
                prompt_done = True
            if not prompt_done:
                event_type = None
                continue
            if chunk_type == "reasoning.delta":
                content = data.get("content")
                if content:
                    if not thinking_started:
                        print("--- Thinking ---")
                        thinking_started = True
                    print(content, end="", flush=True)
            elif chunk_type == "message.delta":
                content = data.get("content")
                if content:
                    if not response_started:
                        if thinking_started:
                            print("\n", end="")
                        print("--- Response ---")
                        response_started = True
                    print(content, end="", flush=True)
            elif chunk_type == "chat.end":
                result = data.get("result") or data
                output = result.get("output") if isinstance(result, dict) else []
                if isinstance(output, list):
                    for item in output:
                        if isinstance(item, dict) and item.get("type") == "message":
                            msg_content = item.get("content")
                            if msg_content and not response_started:
                                if thinking_started:
                                    print("\n", end="")
                                print("--- Response ---")
                                response_started = True
                            if msg_content:
                                print(msg_content, end="", flush=True)
            event_type = None
    if response_started or thinking_started:
        print("\n---")


def stream_chat_openai(
    client: OpenAI, model: str, prompt: str, max_tokens: int = 15000
) -> None:
    """Stream via OpenAI-compatible API (thinking only if server sends reasoning_content)."""
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
    )
    thinking_started = False
    response_started = False
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None) if delta else None
        content = getattr(delta, "content", None) if delta else None
        if reasoning:
            if not thinking_started:
                print("--- Thinking ---")
                thinking_started = True
            print(reasoning, end="", flush=True)
        if content:
            if not response_started:
                if thinking_started:
                    print("\n", end="")
                print("--- Response ---")
                response_started = True
            print(content, end="", flush=True)
    if response_started or thinking_started:
        print("\n---")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    base_url = BASE_URL.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    client = OpenAI(base_url=base_url, api_key=API_KEY)

    try:
        print("Fetching models...")
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        if not model_ids:
            print("No models found. Load a model (or check API key for OpenAI).")
            return
        print("Available models:", model_ids)
        model = MODEL_OVERRIDE if MODEL_OVERRIDE in model_ids else model_ids[0]
        if MODEL_OVERRIDE and MODEL_OVERRIDE not in model_ids:
            print(f"HIVE_MODEL/TEST_MODEL {MODEL_OVERRIDE!r} not in list; using {model!r}.")

        print("\nSending prompt...")
        timeout = float(os.environ.get("HIVE_HTTP_TIMEOUT_SEC", "360"))
        max_tok = int(os.environ.get("HIVE_MAX_TOKENS", "15000"))
        if _is_lm_studio(BASE_URL):
            stream_chat_native(BASE_URL, model, PROMPT, max_tokens=max_tok, timeout=timeout)
        else:
            stream_chat_openai(client, model, PROMPT, max_tokens=max_tok)

    except APIConnectionError as e:
        print(f"Connection error: {e}. Check {BASE_URL} or API key.")
    except APIError as e:
        print(f"API error: {e}")
    except requests.RequestException as e:
        print(f"Request error: {e}")


if __name__ == "__main__":
    main()
