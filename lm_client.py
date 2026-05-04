"""LM Studio native + OpenAI-compatible chat completion (non-streaming, returns text)."""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any

import requests
from openai import APIConnectionError, OpenAI, RateLimitError

try:
    from openai import APIStatusError as _OpenAIAPIStatusError
except ImportError:
    _OpenAIAPIStatusError = ()  # type: ignore[misc,assignment]

# Rough token estimate for budgeting when the API does not return usage.
CHARS_PER_TOKEN_ESTIMATE = 4


def estimate_tokens_from_text(*parts: str) -> int:
    """Heuristic token count from prompt/response strings (for run budgets)."""
    total = sum(len(p or "") for p in parts)
    return max(0, total // CHARS_PER_TOKEN_ESTIMATE)


def is_lm_studio(base_url: str) -> bool:
    u = (base_url or "").strip().lower().rstrip("/")
    if "api.openai.com" in u:
        return False
    return True


def is_openai_api_host(base_url: str) -> bool:
    """True for the official api.openai.com host (distinct from other OpenAI-compatible providers)."""
    return "api.openai.com" in (base_url or "").lower()


def openai_max_completion_tokens_cap(model_id: str) -> int:
    """Conservative per-model ceiling on *output* tokens for chat.completions on OpenAI (env may be higher)."""
    m = (model_id or "").strip().lower()
    if not m:
        return 16_384
    if m.startswith("gpt-3.5"):
        return 4_096
    if m.startswith(("gpt-5", "o1", "o2", "o3", "o4")):
        return 128_000
    if m.startswith("computer-use"):
        return 128_000
    if m.startswith("gpt-4.1"):
        return 128_000
    if m.startswith("gpt-4o") or m.startswith("chatgpt-4o"):
        return 16_384
    if m.startswith("gpt-4-turbo") or "gpt-4-turbo" in m or m.startswith("gpt-4-1106") or m.startswith("gpt-4-0125"):
        return 16_384
    if "32k" in m and m.startswith("gpt-4"):
        return 32_768
    if m.startswith("gpt-4-"):
        return 8_192
    if m.startswith("gpt-4"):
        return 8_192
    return 16_384


def parse_openai_completion_token_ceiling_from_error(exc: BaseException) -> int | None:
    """If the API rejects max_tokens / max_completion_tokens as too large, return the allowed completion ceiling."""
    s = str(exc)
    patterns = (
        r"supports at most (\d+)\s+completion",
        r"at most (\d+)\s+completion tokens",
        r"max_tokens is too large:.*at most (\d+)\s+completion",
        r"max_completion_tokens is too large:.*at most (\d+)\s+completion",
    )
    for pat in patterns:
        mo = re.search(pat, s, re.I | re.DOTALL)
        if mo:
            return int(mo.group(1))
    return None


def clamp_openai_chat_completion_budget(model_id: str, requested: int) -> int:
    """min(requested, model ceiling); output token budget only."""
    cap = openai_max_completion_tokens_cap(model_id)
    return max(1, min(int(requested), cap))


def server_base(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1")


def collect_chat_native(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 15000,
    temperature: float | None = None,
    timeout: float = 360.0,
) -> str:
    """Stream via LM Studio native API; return assistant message text only."""
    url = f"{server_base(base_url)}/api/v1/chat"
    payload: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "stream": True,
        "max_output_tokens": max_tokens,
    }
    # LM Studio may ignore unknown fields; safe to omit when None.
    if temperature is not None:
        payload["temperature"] = float(temperature)
    resp = requests.post(url, json=payload, stream=True, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = "utf-8"

    event_type: str | None = None
    prompt_done = False
    parts: list[str] = []

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
                data: Any = json.loads(data_str)
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
            if chunk_type == "message.delta":
                content = data.get("content")
                if content:
                    parts.append(str(content))
            elif chunk_type == "chat.end":
                result = data.get("result") or data
                output = result.get("output") if isinstance(result, dict) else []
                if isinstance(output, list):
                    for item in output:
                        if isinstance(item, dict) and item.get("type") == "message":
                            msg_content = item.get("content")
                            if msg_content:
                                parts.append(str(msg_content))
            event_type = None

    return "".join(parts)


def _openai_wants_max_completion_tokens_instead(exc: BaseException) -> bool:
    """True when the server rejected max_tokens in favor of max_completion_tokens (newer OpenAI chat models)."""
    s = str(exc).lower()
    if "max_completion_tokens" not in s:
        return False
    if "max_tokens" not in s:
        return False
    if "not supported" not in s and "unsupported" not in s:
        return False
    code = getattr(exc, "status_code", None)
    return code in (400, 422, None)


def _consume_chat_completion_stream(stream: Any) -> str:
    parts: list[str] = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if not delta:
            continue
        reasoning = getattr(delta, "reasoning_content", None)
        content = getattr(delta, "content", None)
        if reasoning:
            parts.append(str(reasoning))
        if content:
            parts.append(str(content))
    return "".join(parts)


def collect_chat_openai(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 15000,
    temperature: float | None = None,
) -> str:
    """Stream OpenAI-compatible chat; return full assistant text."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    try:
        stream = client.chat.completions.create(**kwargs)
        return _consume_chat_completion_stream(stream)
    except Exception as exc:
        if not _openai_wants_max_completion_tokens_instead(exc):
            raise
        kwargs.pop("max_tokens", None)
        kwargs["max_completion_tokens"] = max_tokens
        stream = client.chat.completions.create(**kwargs)
        return _consume_chat_completion_stream(stream)


def build_prompt(system: str | None, user: str) -> str:
    if system and system.strip():
        return f"{system.strip()}\n\n{user}"
    return user


def complete(
    base_url: str,
    client: OpenAI,
    model: str,
    user: str,
    system: str | None = None,
    max_tokens: int = 15000,
    temperature: float | None = None,
    *,
    http_timeout_sec: float = 360.0,
) -> str:
    """Complete one turn; LM Studio native uses concatenated prompt string."""
    if is_lm_studio(base_url):
        prompt = build_prompt(system, user)
        return collect_chat_native(
            base_url,
            model,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=http_timeout_sec,
        )
    messages: list[dict[str, str]] = []
    if system and system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": user})
    mt = int(max_tokens)
    if is_openai_api_host(base_url):
        mt = clamp_openai_chat_completion_budget(model, mt)
    try:
        return collect_chat_openai(client, model, messages, max_tokens=mt, temperature=temperature)
    except Exception as exc:
        if not is_openai_api_host(base_url):
            raise
        lim = parse_openai_completion_token_ceiling_from_error(exc)
        if lim is None:
            raise
        adj = max(256, min(mt, lim))
        if adj >= mt:
            raise
        return collect_chat_openai(client, model, messages, max_tokens=adj, temperature=temperature)


def _is_retryable_http_error(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError)):
        return True
    if _OpenAIAPIStatusError and isinstance(exc, _OpenAIAPIStatusError):
        code = getattr(exc, "status_code", None)
        return code in (429, 502, 503, 504)
    if isinstance(exc, requests.RequestException):
        return True
    return False


def complete_with_retries(
    base_url: str,
    client: OpenAI,
    model: str,
    user: str,
    system: str | None = None,
    max_tokens: int = 15000,
    temperature: float | None = None,
    *,
    max_retries: int = 5,
    base_delay_sec: float = 1.0,
    max_delay_sec: float = 60.0,
    http_timeout_sec: float = 360.0,
) -> str:
    """Same as complete but retries on connection errors and HTTP 429."""
    last_exc: BaseException | None = None
    for attempt in range(max_retries):
        try:
            return complete(
                base_url,
                client,
                model,
                user,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                http_timeout_sec=http_timeout_sec,
            )
        except Exception as e:
            last_exc = e
            if attempt >= max_retries - 1 or not _is_retryable_http_error(e):
                raise
            delay = min(max_delay_sec, base_delay_sec * (2**attempt))
            delay *= 0.8 + 0.4 * random.random()
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def make_openai_client(base_url: str, api_key: str) -> OpenAI:
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return OpenAI(base_url=base, api_key=api_key)


def list_model_ids(client: OpenAI) -> list[str]:
    models = client.models.list()
    return [m.id for m in models.data]


def _openai_model_id_excluded_from_chat_list(low_id: str) -> bool:
    """Heuristic: /v1/models mixes embeddings, images, audio, moderation, etc."""
    needles = (
        "embedding",
        "text-similarity-",
        "text-search-",
        "text-moderation",
        "code-search-",
        "semantic-similarity",
        "moderation",
        "dall-",
        "dalle",
        "whisper",
        "-tts-",
        "tts-",
        "transcribe",
        "realtime",
        "gpt-image",
        "sora",
        "omni-moderation",
        "audio-preview",
        "speech-",
    )
    if any(n in low_id for n in needles):
        return True
    if low_id.startswith("tts-"):
        return True
    return False


def filter_openai_chat_completion_model_ids(ids: list[str]) -> list[str]:
    """Keep ids suitable for chat completions (text); drop embeddings, DALL·E, Whisper, TTS, etc."""
    kept: list[str] = []
    for raw in ids:
        mid = (raw or "").strip()
        if not mid:
            continue
        if _openai_model_id_excluded_from_chat_list(mid.lower()):
            continue
        kept.append(mid)
    return sorted(kept, key=str.lower)


def get_first_model_id(client: OpenAI) -> str:
    models = client.models.list()
    ids = [m.id for m in models.data]
    if not ids:
        raise RuntimeError("No models available from server.")
    return ids[0]
