"""
core/usage.py — measured API cost, per run.

Every model built by core.llm.get_chat_model carries the shared TRACKER callback.
Each completed call reports the exact tokens the API billed (input, output, cache
reads, cache writes); this module prices them and adds them to the run that is
active in the calling context. A "run" is one scrape cycle or one tailored-resume
generation (core/usage.track_run / start_run), so the Scraper tab's "about $0.37"
becomes a measured figure and every generated resume gets a usage_report.json.

Why this exists: before Oct 2026 the app estimated cost (calls x a constant) and
only counted successful calls, so a run showed 21c while the Anthropic bill said
37c. Failed attempts, retries and corrective re-asks are billed too — every call
that returns a response is counted here, whether or not its answer was usable.

Prices are USD per million tokens (Anthropic price table, cached 2026-10-06).
cache read = 0.1x input, cache write = 1.25x (5-minute TTL) or 2x (1-hour TTL).
Models from other providers (Groq free tier, local Ollama) count as $0; a Claude
model missing from PRICES is counted and flagged as unpriced rather than guessed.
"""

import contextvars
import threading
from contextlib import contextmanager

from langchain_core.callbacks import BaseCallbackHandler

# (input, output) USD per 1M tokens, matched by model-id prefix (longest first).
PRICES = {
    "claude-haiku-4-5":  (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-5-5":  (0.10, 0.50),    # prompts up to 100K tokens
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5":   (2.00, 10.00),
    "claude-opus-5-5":   (4.00, 20.00),
    "claude-opus-5":     (5.00, 25.00),
    "claude-opus-4-8":   (5.00, 25.00),
    "claude-opus-4-7":   (5.00, 25.00),
    "claude-opus-4-6":   (5.00, 25.00),
}
CACHE_READ_MULT = 0.10
CACHE_WRITE_5M_MULT = 1.25
CACHE_WRITE_1H_MULT = 2.00


def price_for(model: str):
    """(input, output) per 1M tokens, 0/0 for non-Claude models, None if an unknown Claude model."""
    name = (model or "").lower()
    for prefix in sorted(PRICES, key=len, reverse=True):
        if name.startswith(prefix):
            return PRICES[prefix]
    return None if name.startswith("claude") else (0.0, 0.0)


def call_cost(model: str, input_tokens: int, output_tokens: int,
              cache_read: int = 0, cache_write_5m: int = 0, cache_write_1h: int = 0):
    """USD for one call, or None when the model is an unknown Claude model."""
    p = price_for(model)
    if p is None:
        return None
    pin, pout = p
    return (input_tokens * pin
            + output_tokens * pout
            + cache_read * pin * CACHE_READ_MULT
            + cache_write_5m * pin * CACHE_WRITE_5M_MULT
            + cache_write_1h * pin * CACHE_WRITE_1H_MULT) / 1_000_000


class RunUsage:
    """Usage accumulated for one run. Thread-safe (callbacks may fire from worker threads)."""

    def __init__(self, label: str):
        self.label = label
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def add(self, **call) -> None:
        with self._lock:
            self.calls.append(call)

    @property
    def cost_usd(self) -> float:
        return sum(c["cost_usd"] or 0.0 for c in self.calls)

    def summary(self) -> dict:
        with self._lock:
            calls = list(self.calls)
        by_model, by_node = {}, {}
        for c in calls:
            for key, bucket in ((c["model"], by_model), (c.get("node") or "other", by_node)):
                b = bucket.setdefault(key, {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                            "cache_read_tokens": 0, "cache_write_tokens": 0, "cost_usd": 0.0})
                b["calls"] += 1
                b["input_tokens"] += c["input_tokens"]
                b["output_tokens"] += c["output_tokens"]
                b["cache_read_tokens"] += c["cache_read_tokens"]
                b["cache_write_tokens"] += c["cache_write_tokens"]
                b["cost_usd"] += c["cost_usd"] or 0.0
        for bucket in (by_model, by_node):
            for b in bucket.values():
                b["cost_usd"] = round(b["cost_usd"], 5)
        return {
            "label": self.label,
            "calls": len(calls),
            "input_tokens": sum(c["input_tokens"] for c in calls),
            "output_tokens": sum(c["output_tokens"] for c in calls),
            "cache_read_tokens": sum(c["cache_read_tokens"] for c in calls),
            "cache_write_tokens": sum(c["cache_write_tokens"] for c in calls),
            "cost_usd": round(sum(c["cost_usd"] or 0.0 for c in calls), 5),
            "unpriced_calls": sum(1 for c in calls if c["cost_usd"] is None),
            "by_model": by_model,
            "by_node": by_node,
        }


_current: contextvars.ContextVar = contextvars.ContextVar("usage_run", default=None)
_UNATTRIBUTED = RunUsage("unattributed")   # calls made outside any run (e.g. one-off scoring)


def start_run(label: str) -> RunUsage:
    run = RunUsage(label)
    _current.set(run)
    return run


def current_run():
    return _current.get()


def end_run():
    run = _current.get()
    _current.set(None)
    return run


@contextmanager
def track_run(label: str):
    run = start_run(label)
    try:
        yield run
    finally:
        _current.set(None)


def _usage_from_message(message):
    """(model, input, output, cache_read, cache_write_5m, cache_write_1h) from a chat message.
    Prefers the API's own usage block (uncached input is reported separately there);
    falls back to LangChain's usage_metadata, whose input_tokens already includes cache tokens."""
    meta = getattr(message, "response_metadata", None) or {}
    model = meta.get("model_name") or meta.get("model") or ""
    raw = meta.get("usage")
    if isinstance(raw, dict) and "output_tokens" in raw:
        cache_creation = raw.get("cache_creation") or {}
        w5 = int(cache_creation.get("ephemeral_5m_input_tokens") or 0)
        w1 = int(cache_creation.get("ephemeral_1h_input_tokens") or 0)
        total_write = int(raw.get("cache_creation_input_tokens") or 0)
        if total_write > w5 + w1:
            w5 += total_write - (w5 + w1)        # untyped writes default to the 5-minute rate
        return (model, int(raw.get("input_tokens") or 0), int(raw.get("output_tokens") or 0),
                int(raw.get("cache_read_input_tokens") or 0), w5, w1)
    um = getattr(message, "usage_metadata", None)
    if um:
        details = um.get("input_token_details") or {}
        read, write = int(details.get("cache_read") or 0), int(details.get("cache_creation") or 0)
        return (model, max(0, int(um.get("input_tokens") or 0) - read - write),
                int(um.get("output_tokens") or 0), read, write, 0)
    return None


class UsageTracker(BaseCallbackHandler):
    """Attached to every model; records each completed call into the active run."""

    raise_error = False
    run_inline = True

    def __init__(self):
        self._nodes: dict = {}
        self._lock = threading.Lock()

    def on_chat_model_start(self, serialized, messages, *, run_id=None, metadata=None, **kwargs):
        node = (metadata or {}).get("langgraph_node")
        if node and run_id is not None:
            with self._lock:
                self._nodes[run_id] = node

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        with self._lock:
            node = self._nodes.pop(run_id, None)
        for generations in getattr(response, "generations", None) or []:
            for gen in generations:
                parsed = _usage_from_message(getattr(gen, "message", None))
                if not parsed:
                    continue
                model, inp, out, read, w5, w1 = parsed
                cost = call_cost(model, inp, out, read, w5, w1)
                (current_run() or _UNATTRIBUTED).add(
                    model=model or "unknown", node=node, input_tokens=inp, output_tokens=out,
                    cache_read_tokens=read, cache_write_tokens=w5 + w1, cost_usd=cost)


TRACKER = UsageTracker()
