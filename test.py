"""
Chat test via OpenAI-compatible API (local LM Studio or public OpenAI).
Uses GET /v1/models and POST /v1/chat/completions with streaming.
"""
from openai import OpenAI
from openai import APIError, APIConnectionError


# For local LM Studio: base without /v1, any placeholder API key
# For OpenAI: use "https://api.openai.com", and set API_KEY to your key (or env OPENAI_API_KEY)
BASE_URL = "http://192.168.0.114:1234"
API_KEY = "lm-studio"
PROMPT = "Explain the hixbozon from particle physics in detail to a 5 year old"


def stream_chat(client: OpenAI, model: str, prompt: str, max_tokens: int = 15000) -> None:
    """Stream response using OpenAI-compatible chat completions (reasoning_content + content)."""
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
    base_url = BASE_URL.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    client = OpenAI(base_url=base_url, api_key=API_KEY)

    try:
        # 1. List models (GET /v1/models)
        print("Fetching models...")
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        if not model_ids:
            print("No models found. Load a model (or check API key for OpenAI).")
            return
        print("Available models:", model_ids)
        model = model_ids[0]

        # 2. Send prompt and stream result (OpenAI-compatible)
        print("\nSending prompt...")
        stream_chat(client, model, PROMPT, max_tokens=15000)

    except APIConnectionError as e:
        print(f"Connection error: {e}. Check {BASE_URL} or API key.")
    except APIError as e:
        print(f"API error: {e}")


if __name__ == "__main__":
    main()
