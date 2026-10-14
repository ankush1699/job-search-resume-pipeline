"""
tests/test_usage_tracking.py

core/usage.py: measured API cost per run. The point is that every call the API
bills is counted — retries and failed attempts included — priced from the exact
tokens in the response, so a run's cost matches the Anthropic bill instead of
being an estimate (Oct 2026: the app showed 21c for a run that billed 37c).
"""
import os
import sys
import threading
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import usage


def _gen(model, inp, out, read=0, write=0, w1h=0):
    msg = SimpleNamespace(response_metadata={"model_name": model, "usage": {
        "input_tokens": inp, "output_tokens": out, "cache_read_input_tokens": read,
        "cache_creation_input_tokens": write + w1h,
        "cache_creation": {"ephemeral_5m_input_tokens": write, "ephemeral_1h_input_tokens": w1h}}},
        usage_metadata=None)
    return SimpleNamespace(generations=[[SimpleNamespace(message=msg)]])


def test_haiku_pricing_matches_the_published_table():
    # 1,000 in + 1,000 out on Haiku 4.5 = $0.001 + $0.005
    assert abs(usage.call_cost("claude-haiku-4-5-20251001", 1000, 1000) - 0.006) < 1e-9
    # Sonnet 4.6: $3 / $15 per MTok
    assert abs(usage.call_cost("claude-sonnet-4-6", 1_000_000, 1_000_000) - 18.0) < 1e-9


def test_cache_reads_and_writes_are_priced_at_their_own_rates():
    m = "claude-haiku-4-5"
    assert abs(usage.call_cost(m, 0, 0, cache_read=1_000_000) - 0.10) < 1e-9          # 0.1x input
    assert abs(usage.call_cost(m, 0, 0, cache_write_5m=1_000_000) - 1.25) < 1e-9      # 1.25x input
    assert abs(usage.call_cost(m, 0, 0, cache_write_1h=1_000_000) - 2.00) < 1e-9      # 2x input


def test_free_tier_models_cost_nothing_and_unknown_claude_models_are_flagged():
    assert usage.call_cost("openai/gpt-oss-120b", 5000, 500) == 0.0
    assert usage.call_cost("claude-future-9", 1000, 1000) is None


def test_every_call_in_a_run_is_counted_including_failed_attempts():
    tracker = usage.UsageTracker()
    with usage.track_run("scrape") as run:
        for _ in range(3):   # e.g. 3 identical failed scoring attempts — all billed
            tracker.on_llm_end(_gen("claude-haiku-4-5-20251001", 5176, 120))
        tracker.on_llm_end(_gen("claude-haiku-4-5-20251001", 660, 300, read=4516))
    s = run.summary()
    assert s["calls"] == 4 and s["unpriced_calls"] == 0
    assert s["input_tokens"] == 3 * 5176 + 660 and s["cache_read_tokens"] == 4516
    expected = (3 * usage.call_cost("claude-haiku-4-5", 5176, 120)
                + usage.call_cost("claude-haiku-4-5", 660, 300, cache_read=4516))
    assert abs(s["cost_usd"] - round(expected, 5)) < 1e-9


def test_cost_is_attributed_per_model_and_per_pipeline_node():
    tracker = usage.UsageTracker()
    with usage.track_run("tailor") as run:
        tracker.on_chat_model_start({}, [], run_id="a", metadata={"langgraph_node": "writer"})
        tracker.on_llm_end(_gen("claude-sonnet-4-6", 6000, 2500), run_id="a")
        tracker.on_chat_model_start({}, [], run_id="b", metadata={"langgraph_node": "strategist"})
        tracker.on_llm_end(_gen("claude-haiku-4-5-20251001", 4000, 800), run_id="b")
    s = run.summary()
    assert set(s["by_model"]) == {"claude-sonnet-4-6", "claude-haiku-4-5-20251001"}
    assert set(s["by_node"]) == {"writer", "strategist"}
    assert s["by_node"]["writer"]["cost_usd"] > s["by_node"]["strategist"]["cost_usd"]


def test_runs_do_not_leak_into_each_other_across_threads():
    tracker = usage.UsageTracker()
    totals = {}

    def work(name, n):
        with usage.track_run(name) as run:
            for _ in range(n):
                tracker.on_llm_end(_gen("claude-haiku-4-5", 1000, 100))
        totals[name] = run.summary()["calls"]

    threads = [threading.Thread(target=work, args=(f"r{i}", i + 2)) for i in range(4)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert totals == {"r0": 2, "r1": 3, "r2": 4, "r3": 5}


def test_calls_outside_any_run_never_crash_and_never_join_a_run():
    tracker = usage.UsageTracker()
    before = usage._UNATTRIBUTED.summary()["calls"]
    tracker.on_llm_end(_gen("claude-haiku-4-5", 100, 10))
    assert usage._UNATTRIBUTED.summary()["calls"] == before + 1
    with usage.track_run("x") as run:
        pass
    assert run.summary()["calls"] == 0


def test_langchain_usage_metadata_fallback_does_not_double_count_cache_tokens():
    msg = SimpleNamespace(response_metadata={"model_name": "claude-haiku-4-5"},
                          usage_metadata={"input_tokens": 5176, "output_tokens": 100,
                                          "input_token_details": {"cache_read": 4516, "cache_creation": 0}})
    model, inp, out, read, w5, w1 = usage._usage_from_message(msg)
    assert (inp, read, out) == (660, 4516, 100)    # uncached = total - cached


def test_get_chat_model_attaches_the_tracker():
    from core.llm import get_chat_model
    m = get_chat_model("cheap", provider_override="anthropic")
    assert usage.TRACKER in (m.callbacks or [])
    assert usage.TRACKER in (get_chat_model("quality").callbacks or [])


def test_run_history_uses_the_measured_cost_when_a_run_was_tracked():
    from scraper import run_history as rh
    tracker = usage.UsageTracker()
    rec = rh.new_record(kind="manual")
    rec["scored"]["model_calls"] = 99                  # the estimate would say 99 x $0.004
    usage.start_run("scrape")
    tracker.on_llm_end(_gen("claude-haiku-4-5", 1_000_000, 0))   # exactly $1.00
    rh.finish(rec)
    assert rec["cost_source"] == "measured" and rec["est_cost_usd"] == 1.0
    rec2 = rh.new_record(kind="manual"); rec2["scored"]["model_calls"] = 10
    rh.finish(rec2)                                    # nothing tracked -> falls back to the estimate
    assert rec2["cost_source"] == "estimated" and rec2["est_cost_usd"] == round(10 * rh.COST_PER_MODEL_CALL_USD, 3)
