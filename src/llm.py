"""Minimal multi-provider (OpenAI-compatible) chat client with on-disk cache + retry.

Routing:
  - deepseek*            -> DeepSeek   (api_key_deepseek.env, `API_KEY=...`)
  - openai/* or gpt-*    -> OpenAI direct (openai_api_key.env, bare key) if available,
                            else OpenRouter fallback. The `openai/` prefix is stripped for
                            the direct OpenAI API call, but the CACHE KEY keeps the caller's
                            original model string -> old OpenRouter cache rows stay valid.
  - everything else      -> OpenRouter (openrouter_api.env, `OPENROUTER_API_KEY=...`)
Default model = openai/gpt-4o-mini. Keys are never logged. Cache lives in
results/llm_cache.sqlite so re-runs don't re-pay.
"""
import os
import json
import math
import time
import hashlib
import sqlite3
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # E:\anti-dis
ENV_FILE = ROOT / "openrouter_api.env"
CACHE_DB = ROOT / "results" / "llm_cache.sqlite"
BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"


def load_env():
    if os.environ.get("OPENROUTER_API_KEY"):
        return
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()

from openai import OpenAI  # noqa: E402

_or_client = OpenAI(base_url=BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])

DEEPSEEK_ENV = ROOT / "api_key_deepseek.env"
OPENAI_ENV = ROOT / "openai_api_key.env"


def _read_key_file(path):
    """Read an API key from a file that is EITHER `NAME=value` lines OR a single bare key.
    Returns the first usable key string, or None. (Bare keys, e.g. `sk-proj-...`, contain no `=`.)"""
    if not path.exists():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:                      # bare key on its own line
            return line.strip('"').strip("'")
        return line.split("=", 1)[1].strip().strip('"').strip("'")   # NAME=value
    return None


_ds_key = os.environ.get("DEEPSEEK_API_KEY") or _read_key_file(DEEPSEEK_ENV)
_ds_client = OpenAI(base_url="https://api.deepseek.com", api_key=_ds_key) if _ds_key else None

_openai_key = os.environ.get("OPENAI_API_KEY") or _read_key_file(OPENAI_ENV)
_openai_client = OpenAI(api_key=_openai_key) if _openai_key else None   # base_url defaults to api.openai.com

CLAUDE_ENV = ROOT / "claude_api_key.env"
_anthropic_key = os.environ.get("ANTHROPIC_API_KEY") or _read_key_file(CLAUDE_ENV)
try:
    import anthropic as _anthropic  # noqa: E402
    _anthropic_client = _anthropic.Anthropic(api_key=_anthropic_key) if _anthropic_key else None
except ImportError:
    _anthropic, _anthropic_client = None, None

_lock = threading.Lock()


def _anthropic_complete(model, messages, temperature, max_tokens):
    """Native Anthropic SDK call for claude* models. Splits OpenAI-style `messages` into a `system` arg +
    user/assistant turns, clamps temperature to [0,1] (Anthropic's range), returns concatenated text."""
    if _anthropic_client is None:
        raise RuntimeError("Anthropic key not found (claude_api_key.env) or SDK missing")
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
    conv = [{"role": ("assistant" if m["role"] == "assistant" else "user"), "content": m["content"]}
            for m in messages if m.get("role") != "system"]
    kwargs = {"model": model, "max_tokens": max(1, max_tokens), "messages": conv}
    if system:
        kwargs["system"] = system
    if temperature is not None:
        kwargs["temperature"] = max(0.0, min(1.0, float(temperature)))
    # per-call timeout + SDK retries so a hung HTTP connection can't deadlock a worker indefinitely
    resp = _anthropic_client.with_options(timeout=60.0, max_retries=3).messages.create(**kwargs)
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def _complete(model, messages, temperature, max_tokens):
    """One completion -> text. Dispatches claude* to the native Anthropic SDK, else OpenAI-compatible."""
    if model.startswith("claude"):
        return _anthropic_complete(model, messages, temperature, max_tokens)
    client, api_model = _client_for(model)
    resp = client.chat.completions.create(model=api_model, messages=messages,
                                          temperature=temperature, max_tokens=max_tokens)
    return resp.choices[0].message.content or ""


def _client_for(model):
    """Return (client, api_model_name). api_model_name may differ from `model` (prefix stripped)."""
    if model.startswith("openrouter/"):                      # force OpenRouter (e.g. openrouter/openai/gpt-4o)
        return _or_client, model.split("/", 1)[1]
    if model.startswith("deepseek"):
        if _ds_client is None:
            raise RuntimeError("DeepSeek key not found (api_key_deepseek.env: API_KEY=...)")
        return _ds_client, model
    if model.startswith("openai/") or model.startswith("gpt-"):
        if _openai_client is not None:                       # GPT tests -> direct OpenAI key
            api_model = model.split("/", 1)[1] if model.startswith("openai/") else model
            return _openai_client, api_model
        return _or_client, model                             # fallback: OpenRouter
    return _or_client, model


def _conn():
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(CACHE_DB, timeout=30)
    c.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
    return c


def _key(model, messages, params):
    blob = json.dumps([model, messages, params], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def chat(messages, model=DEFAULT_MODEL, temperature=0.0, max_tokens=1024,
         use_cache=True, retries=5):
    """Return the assistant message text. Cached by (model, messages, params)."""
    params = {"temperature": temperature, "max_tokens": max_tokens}
    k = _key(model, messages, params)
    if use_cache:
        with _lock:
            c = _conn()
            row = c.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
            c.close()
        if row is not None:
            return row[0]
    last = None
    for attempt in range(retries):
        try:
            out = _complete(model, messages, temperature, max_tokens)
            if use_cache:
                with _lock:
                    c = _conn()
                    c.execute("INSERT OR REPLACE INTO cache (k, v) VALUES (?, ?)", (k, out))
                    c.commit()
                    c.close()
            return out
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"LLM call failed after {retries} retries: {last}")


def unsafe_prob(messages, model="openai/gpt-4o-mini", use_cache=True, retries=5):
    """Continuous P(unsafe) from first-token logprobs (unsafe-class mass / (unsafe+safe)).
    NOTE: only models with real logprobs work (gpt-4o-mini yes; deepseek returns degenerate
    p=1.0 logprobs -> unusable). The prompt must make the first token be 'unsafe' or 'safe'."""
    params = {"temperature": 0.0, "max_tokens": 1, "logprobs": True, "top_logprobs": 16}
    k = _key(model, messages, {**params, "_mode": "unsafe_prob_v1"})
    if use_cache:
        with _lock:
            c = _conn()
            row = c.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
            c.close()
        if row is not None:
            return float(row[0])
    last = None
    for attempt in range(retries):
        try:
            client, api_model = _client_for(model)
            resp = client.chat.completions.create(model=api_model, messages=messages, **params)
            top = resp.choices[0].logprobs.content[0].top_logprobs
            uns = saf = 0.0
            for t in top:
                tok = t.token.strip().lower()
                p = math.exp(t.logprob)
                if tok.startswith("uns"):
                    uns += p
                elif tok.startswith("saf"):
                    saf += p
            prob = uns / (uns + saf) if (uns + saf) > 0 else 0.5
            if use_cache:
                with _lock:
                    c = _conn()
                    c.execute("INSERT OR REPLACE INTO cache (k, v) VALUES (?, ?)", (k, str(prob)))
                    c.commit()
                    c.close()
            return prob
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"unsafe_prob failed after {retries} retries: {last}")


def sample_unsafe_prob(messages, model, k=8, temperature=1.0, use_cache=True, retries=5):
    """Continuous P(unsafe) = fraction of k independent samples (temp>0) that answer 'unsafe'.
    Works for ANY model (no logprobs needed) -> lets us use a strong model like deepseek-v4.
    Each sample is cached separately (by a _sample index NOT sent to the API), so reruns are free.
    The prompt must make the model answer with 'unsafe' or 'safe'."""
    unsafe = 0
    for s in range(k):
        kkey = _key(model, messages, {"temperature": temperature, "max_tokens": 4, "_sample": s})
        out = None
        if use_cache:
            with _lock:
                c = _conn()
                row = c.execute("SELECT v FROM cache WHERE k=?", (kkey,)).fetchone()
                c.close()
            if row is not None:
                out = row[0]
        if out is None:
            last = None
            for attempt in range(retries):
                try:
                    client, api_model = _client_for(model)
                    resp = client.chat.completions.create(
                        model=api_model, messages=messages, temperature=temperature, max_tokens=4)
                    out = (resp.choices[0].message.content or "")
                    if use_cache:
                        with _lock:
                            c = _conn()
                            c.execute("INSERT OR REPLACE INTO cache (k, v) VALUES (?, ?)", (kkey, out))
                            c.commit()
                            c.close()
                    break
                except Exception as e:  # noqa: BLE001
                    last = e
                    time.sleep(min(2 ** attempt, 30))
            else:
                raise RuntimeError(f"sample_unsafe_prob failed after {retries} retries: {last}")
        if "unsafe" in out.strip().lower():
            unsafe += 1
    return unsafe / k


def sample_one(messages, model, s, temperature=1.0, max_tokens=8, use_cache=True, retries=5, salt=None):
    """One temperature-sampled completion, cached by sample index `s` (so calling with
    s=0..k-1 gives k independent, individually-cached samples; reruns are free). `salt`
    (optional) further separates cache rows so a caller can force distinct entries even when
    `messages` are byte-identical across conditions."""
    params = {"temperature": temperature, "max_tokens": max_tokens, "_sample": s}
    if salt is not None:
        params["_salt"] = salt
    kkey = _key(model, messages, params)
    if use_cache:
        with _lock:
            c = _conn()
            row = c.execute("SELECT v FROM cache WHERE k=?", (kkey,)).fetchone()
            c.close()
        if row is not None:
            return row[0]
    last = None
    for attempt in range(retries):
        try:
            out = _complete(model, messages, temperature, max_tokens)
            if use_cache:
                with _lock:
                    c = _conn()
                    c.execute("INSERT OR REPLACE INTO cache (k, v) VALUES (?, ?)", (kkey, out))
                    c.commit()
                    c.close()
            return out
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"sample_one failed after {retries} retries: {last}")
