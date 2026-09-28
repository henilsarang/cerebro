"""Day 1 - The provider: one function between the harness and the model.

Concept: an agent harness talks to a model through a single narrow call,
complete(), which takes a provider-neutral message list and returns a
provider-neutral reply. Everything Gemini-specific (URLs, wire format,
thought signatures, retries) is quarantined in this file.

Design rules:
  * Neutral in, neutral out. The rest of Cerebro never sees Gemini's JSON.
  * Standard library only: urllib, json, time, os.
  * Transient failures (rate limits, 5xx, network) are retried with
    exponential backoff; everything else fails loudly with the server's reason.
"""
import json
import os
import time
import urllib.error
import urllib.request

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3.1-pro-preview"

# Statuses worth retrying: rate limiting and transient server trouble.
RETRY_STATUSES = {429, 500, 502, 503}


def api_key():
    """Return the API key from CEREBRO_API_KEY, falling back to GEMINI_API_KEY."""
    key = os.environ.get("CEREBRO_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "No API key found. Set CEREBRO_API_KEY (or GEMINI_API_KEY) "
            "to a Gemini API key before running Cerebro."
        )
    return key


def _to_wire(messages):
    """Translate neutral messages into Gemini `contents` entries."""
    wire = []
    for m in messages:
        if m["role"] == "user":
            wire.append({"role": "user", "parts": [{"text": m["text"]}]})
        elif m["role"] == "assistant":
            parts = [{"text": m["text"]}] if m.get("text") else []
            for call in m.get("tool_calls") or []:
                part = {"functionCall": {"name": call["name"], "args": call["args"]}}
                # Gemini 3 rejects a follow-up turn unless each functionCall
                # part carries back the exact thoughtSignature it was issued.
                if call.get("signature"):
                    part["thoughtSignature"] = call["signature"]
                parts.append(part)
            wire.append({"role": "model", "parts": parts})
        elif m["role"] == "tool":
            response = {"name": m["name"], "response": {"result": m["text"]}}
            wire.append({"role": "user", "parts": [{"functionResponse": response}]})
        else:
            raise ValueError(f"unknown message role: {m['role']!r}")
    return wire


def _post(url, body, retries=5):
    """POST JSON to url and return the decoded reply, retrying transient errors."""
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    for attempt in range(retries):
        last = attempt == retries - 1
        request = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=600) as reply:
                return json.loads(reply.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            if err.code not in RETRY_STATUSES or last:
                detail = err.read().decode("utf-8", "replace")[:400]
                raise RuntimeError(f"HTTP {err.code} from model API: {detail}") from err
        except (urllib.error.URLError, TimeoutError) as err:
            if last:
                raise RuntimeError(f"network error talking to model API: {err}") from err
        time.sleep(2 ** attempt * 2)


def complete(model, system, messages, tools):
    """Send one request and return {"text", "tool_calls", "usage"}.

    tools is a list of spec dicts, each {"schema": <function declaration>},
    or empty/None for a tool-free call.
    """
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": _to_wire(messages),
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 65536},
    }
    if tools:
        body["tools"] = [{"functionDeclarations": [t["schema"] for t in tools]}]
    url = f"{API_ROOT}/{model}:generateContent?key={api_key()}"
    reply = _post(url, body)

    candidates = reply.get("candidates") or [{}]
    parts = (candidates[0].get("content") or {}).get("parts") or []
    # Thought parts are the model's private reasoning, not its answer.
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    tool_calls = [
        {"name": p["functionCall"]["name"],
         "args": p["functionCall"].get("args") or {},
         "signature": p.get("thoughtSignature")}
        for p in parts if "functionCall" in p
    ]
    meta = reply.get("usageMetadata") or {}
    usage = {"input": meta.get("promptTokenCount", 0),
             "output": meta.get("candidatesTokenCount", 0)}
    return {"text": text, "tool_calls": tool_calls, "usage": usage}
