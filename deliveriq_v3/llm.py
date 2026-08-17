import os
import requests

# ── Gemini free tier — no local model to download/host, and no manual
# JSON-parsing loop needed: Gemini has NATIVE function calling, so the
# agent just hands it a tool list and reads back a structured functionCall
# instead of us regex-parsing free text (that regex retry loop was the
# other big source of slowness with Ollama).
#
# Get a free key (no credit card) at: https://aistudio.google.com/apikey
# Then:  export GEMINI_API_KEY=your-key-here
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# flash-lite is the fastest + highest-quota free model (best for an agent
# loop that may fire 2-3 requests per user message). If you need slightly
# better reasoning and don't mind a bit more latency, set:
#   export GEMINI_MODEL=gemini-2.5-flash
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
# Tried in order if the primary model 429s (quota) OR 404s (not enabled for
# your key/region — Google's model lineup shifts fairly often). The last
# entry, "gemini-flash-latest", is Google's alias that always points at
# whatever their current fastest supported Flash model is, so this chain
# should never fully dead-end just because a specific version name changed.
GEMINI_MODEL_FALLBACKS = ["gemini-2.5-flash", "gemini-flash-latest"]

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

SYSTEM_PROMPT_DEFAULT = (
    "You are the DeliverIQ delivery assistant for a Brazilian e-commerce "
    "logistics platform. All state codes (SP, RJ, MG, BA, etc.) are "
    "Brazilian states — never rename, translate, or reinterpret one."
)


def _role_for(msg_role: str) -> str:
    """Map our simple role names to Gemini's contents roles."""
    return "model" if msg_role == "assistant" else "user"


def _messages_to_contents(messages):
    """Converts a plain [{"role","content"}, ...] list (system/user/assistant)
    into Gemini's `contents` + `systemInstruction` shape. Used for simple,
    non-tool-calling calls (see chat())."""
    system_instruction = None
    contents = []
    for m in messages:
        if m["role"] == "system":
            system_instruction = {"parts": [{"text": m["content"]}]}
            continue
        contents.append({"role": _role_for(m["role"]), "parts": [{"text": m["content"]}]})
    return system_instruction, contents


def _post(model: str, payload: dict, timeout: int = 25):
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "\u274c No GEMINI_API_KEY set.\n"
            "Fix: grab a free key (no credit card) at https://aistudio.google.com/apikey\n"
            "then run:  export GEMINI_API_KEY=your-key-here\n"
            "(Windows PowerShell:  $env:GEMINI_API_KEY=\"your-key-here\")"
        )
    url = GEMINI_URL.format(model=model) + f"?key={GEMINI_API_KEY}"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError:
        raise RuntimeError("\u274c Can't reach the Gemini API \u2014 check your internet connection.")
    except requests.exceptions.Timeout:
        raise RuntimeError("\u23f1 Gemini request timed out. Try again.")
    return resp


def generate(contents, system_instruction=None, tools=None, temperature=0.2,
             max_output_tokens=512, model=None):
    """
    Low-level call to Gemini's generateContent endpoint. Returns the raw
    `candidates[0].content` dict (still has "parts", possibly including
    functionCall parts) so callers can decide what to do with it.

    Automatically retries once on the fallback model if the primary model's
    free-tier quota is hit (HTTP 429) — keeps the app usable during bursts
    without you having to think about it.
    """
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    if system_instruction:
        payload["systemInstruction"] = system_instruction
    if tools:
        payload["tools"] = [{"functionDeclarations": tools}]
        # AUTO = model decides whether to call a tool or answer directly.
        payload["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

    primary = model or GEMINI_MODEL
    models_to_try = [primary] if model else [primary] + GEMINI_MODEL_FALLBACKS

    last_error = None
    for m in models_to_try:
        resp = _post(m, payload)
        if resp.status_code == 429:
            last_error = f"429 quota exceeded on {m}"
            continue
        if resp.status_code == 404:
            # Not available for this key/region/API version — try the next
            # model in the chain instead of failing outright.
            last_error = f"404 not found/not enabled: {m}"
            continue
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            # Safety-blocked or empty response — surface something usable.
            reason = data.get("promptFeedback", {}).get("blockReason", "empty response")
            return {"parts": [{"text": f"(Gemini returned no answer: {reason})"}]}
        return candidates[0]["content"]

    raise RuntimeError(
        f"\u274c None of these models worked for your key: {', '.join(models_to_try)}.\n"
        f"Last error: {last_error}\n"
        f"Run this to see what your key can actually use:\n"
        f'  curl "https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY"\n'
        f"then set, e.g.:  export GEMINI_MODEL=<a model name from that list>"
    )


def chat(messages, temperature: float = 0.2, **_ignored) -> str:
    """
    Simple multi-turn text call — same shape the old Ollama-backed chat()
    had ({"role": "system"/"user"/"assistant", "content": str} dicts in,
    plain text out) so anything that only needs a plain answer (no tools)
    keeps working unchanged. `json_mode` from the old signature is accepted
    and ignored: with native function calling, agent.py no longer needs to
    ask the model to hand-write JSON.
    """
    system_instruction, contents = _messages_to_contents(messages)
    content = generate(contents, system_instruction=system_instruction, temperature=temperature)
    parts = content.get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()


def call_llm(user_message: str, context: str = "") -> str:
    """Single-turn convenience call used for plain (non-agentic) RAG answers."""
    if context:
        prompt = (
            f"Context (use exactly as written, state codes are Brazilian "
            f"states \u2014 do not rename them):\n{context}\n\n"
            f"Customer question: {user_message}\n\n"
            f"Answer in a warm, friendly paragraph (roughly 2-5 sentences), "
            f"no bullet points, no markdown."
        )
    else:
        prompt = user_message

    return chat([
        {"role": "system", "content": SYSTEM_PROMPT_DEFAULT},
        {"role": "user", "content": prompt},
    ])


# ── Embeddings (used by rag.py) ──
EMBED_MODEL = "text-embedding-004"
EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:embedContent"


def embed_text(text: str):
    """Returns a list[float] embedding for `text` using Gemini's free
    embedding model — replaces the old Ollama nomic-embed-text call."""
    if not GEMINI_API_KEY:
        raise RuntimeError("No GEMINI_API_KEY set (see chat()/generate() for how to get one).")
    url = EMBED_URL + f"?key={GEMINI_API_KEY}"
    resp = requests.post(url, json={
        "model": f"models/{EMBED_MODEL}",
        "content": {"parts": [{"text": text}]},
    }, timeout=20)
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]
