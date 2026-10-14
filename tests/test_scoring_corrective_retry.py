"""
tests/test_scoring_corrective_retry.py

Regression tests for the live scoring failure of Sep 18 2026 (Google
"Software Engineer III, Full Stack", entry c6f8b8d6f4ab):

  Haiku labeled the JD seniority_level="mid" but scored seniority 5 — its
  reasoning discounted the CANDIDATE ("zero US work experience") instead of
  rating the JOB's level. _SubScoreInconsistent fired (correctly), the loop
  hopped to Groq, Groq's model emitted truncated tool-call JSON
  (tool_use_failed), Ollama wasn't running, and with no fallback left the
  old loop fell through to plain retries of Groq — the provider it had just
  declared a dead end — three times. The job ended up unscored.

New behaviour under test:
  1. An inconsistent answer gets ONE corrective re-ask on the SAME provider
     (contradiction fed back as an extra human turn; system prefix untouched
     so the prompt cache still hits) before any hop.
  2. A hop-worthy failure with nothing left in the chain raises immediately,
     naming each provider's failure — no dead-end retries, no sleeping.

Run: pytest tests/test_scoring_corrective_retry.py -v
"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import HumanMessage, SystemMessage

from core.should_apply import _build_messages, _llm_score, _CORRECTION_TEMPLATE


def _scores(tech, sen, role, spons, *, seniority_level="mid", role_type="fullstack-swe"):
    m = MagicMock()
    m.tech_score, m.seniority_score, m.role_type_score, m.sponsorship_bonus_score = tech, sen, role, spons
    m.seniority_level, m.role_type = seniority_level, role_type
    m.reasoning, m.jd_summary = "r", "s"
    m.tailoring_recommended, m.tailoring_reasoning = False, "t"
    return m


# The exact live shape: label "mid" (rubric: 20) paired with the senior-tier 5.
_LIVE_INCONSISTENT = _scores(42, 5, 15, 7, seniority_level="mid")
_LIVE_CORRECTED = _scores(42, 20, 15, 7, seniority_level="mid")

_GROQ_TOOL_USE_FAILED = Exception(
    "Error code: 400 - {'error': {'message': 'Failed to parse tool call arguments as JSON', "
    "'type': 'invalid_request_error', 'code': 'tool_use_failed', 'failed_generation': '{...'}}"
)


def test_inconsistent_answer_is_corrected_on_the_same_provider_before_any_hop():
    haiku = MagicMock()
    invoke = haiku.with_structured_output.return_value.invoke
    invoke.side_effect = [_LIVE_INCONSISTENT, _LIVE_CORRECTED]

    with patch("core.should_apply.cheap_provider", return_value="anthropic"), \
         patch("core.should_apply.next_cheap_provider") as mock_next, \
         patch("core.should_apply.get_chat_model", return_value=haiku), \
         patch("core.should_apply.time.sleep") as mock_sleep:
        result = _llm_score("some JD text")

    assert result["sub_scores"]["seniority"] == 20
    assert result["score"] == 42 + 20 + 15 + 7
    assert invoke.call_count == 2
    mock_next.assert_not_called()   # never left Haiku
    mock_sleep.assert_not_called()  # the re-ask is immediate

    # The second call carried the contradiction back to the model, after an
    # unchanged system + JD prefix (cache-friendly).
    second_messages = invoke.call_args_list[1].args[0]
    first_messages = invoke.call_args_list[0].args[0]
    assert second_messages[:2] == first_messages[:2]
    assert len(second_messages) == 3
    assert "seniority_level='mid'" in second_messages[2].content
    assert "got 5" in second_messages[2].content


def test_second_inconsistent_answer_hops_and_a_dead_chain_raises_immediately():
    """The full live sequence with the fix: Haiku wrong twice -> hop to Groq
    -> tool_use_failed -> nothing left -> raise at once, Groq called exactly
    once, no backoff sleeps, both failures named in the error."""
    haiku = MagicMock()
    haiku.with_structured_output.return_value.invoke.side_effect = [_LIVE_INCONSISTENT, _LIVE_INCONSISTENT]
    groq = MagicMock()
    groq.with_structured_output.return_value.invoke.side_effect = _GROQ_TOOL_USE_FAILED

    with patch("core.should_apply.cheap_provider", return_value="anthropic"), \
         patch("core.should_apply.next_cheap_provider", side_effect=lambda p: {"anthropic": "groq"}.get(p)), \
         patch("core.should_apply.get_chat_model", side_effect=[haiku, groq]), \
         patch("core.should_apply.time.sleep") as mock_sleep:
        try:
            _llm_score("some JD text", max_retries=3)
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            msg = str(e)

    assert groq.with_structured_output.return_value.invoke.call_count == 1
    assert haiku.with_structured_output.return_value.invoke.call_count == 2
    mock_sleep.assert_not_called()
    assert "anthropic: inconsistent scores" in msg
    assert "groq: quota exhausted / unusable tool calls" in msg
    assert "tool_use_failed" in msg


def test_transient_errors_still_use_the_normal_retry_budget():
    """A plain transient failure is not hop-worthy and not a score problem:
    it must still back off and retry the same provider as before."""
    haiku = MagicMock()
    haiku.with_structured_output.return_value.invoke.side_effect = [
        Exception("Connection reset by peer"), _LIVE_CORRECTED,
    ]
    with patch("core.should_apply.cheap_provider", return_value="anthropic"), \
         patch("core.should_apply.next_cheap_provider", return_value=None), \
         patch("core.should_apply.get_chat_model", return_value=haiku), \
         patch("core.should_apply.time.sleep") as mock_sleep:
        result = _llm_score("some JD text", max_retries=3)
    assert result["score"] == 42 + 20 + 15 + 7
    mock_sleep.assert_called_once()


def test_build_messages_correction_turn_keeps_cached_prefix_intact():
    base = _build_messages("anthropic", "STATIC", "JOB DESCRIPTION:\nx")
    corrected = _build_messages("anthropic", "STATIC", "JOB DESCRIPTION:\nx",
                                correction="seniority_level='mid' allows scores [25, 20] per the rubric, but got 5")
    assert corrected[:2] == base[:2]
    assert isinstance(corrected[0], SystemMessage)
    assert corrected[0].content[0]["cache_control"] == {"type": "ephemeral"}
    assert isinstance(corrected[2], HumanMessage)
    assert corrected[2].content == _CORRECTION_TEMPLATE.format(
        problem="seniority_level='mid' allows scores [25, 20] per the rubric, but got 5")
    assert "describe the JOB, not the candidate's fit" in corrected[2].content



# ── Malformed tool calls (Oct 2026) ─────────────────────────────────────────

def _validation_error(missing_field="reasoning"):
    from pydantic import BaseModel, ValidationError

    class M(BaseModel):
        reasoning: str

    try:
        M()
    except ValidationError as e:
        return e


def test_malformed_answer_gets_one_reask_naming_the_field_then_succeeds():
    """A real run burned 3 identical calls per job on these (temperature 0
    reproduces the same mistake). Now: one re-ask that names the field."""
    from core.should_apply import ScoringFailed  # noqa: F401  (import check)
    haiku = MagicMock()
    invoke = haiku.with_structured_output.return_value.invoke
    invoke.side_effect = [_validation_error(), _LIVE_CORRECTED]
    with patch("core.should_apply.cheap_provider", return_value="anthropic"), \
         patch("core.should_apply.next_cheap_provider") as mock_next, \
         patch("core.should_apply.get_chat_model", return_value=haiku), \
         patch("core.should_apply.time.sleep") as mock_sleep:
        result = _llm_score("some JD text")
    assert result["score"] == 42 + 20 + 15 + 7
    assert result["paid_calls"] == 2
    assert invoke.call_count == 2
    mock_next.assert_not_called()
    mock_sleep.assert_not_called()
    reask = invoke.call_args_list[1].args[0][2].content
    assert "'reasoning' field was missing" in reask and "separate field" in reask


def test_repeated_malformed_answers_stop_after_two_paid_calls_and_report_them():
    from core.should_apply import ScoringFailed
    haiku = MagicMock()
    haiku.with_structured_output.return_value.invoke.side_effect = [_validation_error(), _validation_error()]
    with patch("core.should_apply.cheap_provider", return_value="anthropic"), \
         patch("core.should_apply.next_cheap_provider", return_value=None), \
         patch("core.should_apply.get_chat_model", return_value=haiku), \
         patch("core.should_apply.time.sleep"):
        try:
            _llm_score("some JD text", max_retries=3)
            assert False, "expected ScoringFailed"
        except ScoringFailed as e:
            assert e.paid_calls == 2
            assert "malformed answers" in str(e)
    assert haiku.with_structured_output.return_value.invoke.call_count == 2   # was 3 identical calls


def test_paid_calls_count_only_anthropic():
    haiku, groq = MagicMock(), MagicMock()
    haiku.with_structured_output.return_value.invoke.side_effect = [_validation_error(), _validation_error()]
    groq.with_structured_output.return_value.invoke.return_value = _LIVE_CORRECTED
    with patch("core.should_apply.cheap_provider", return_value="anthropic"), \
         patch("core.should_apply.next_cheap_provider", side_effect=lambda p: {"anthropic": "groq"}.get(p)), \
         patch("core.should_apply.get_chat_model", side_effect=[haiku, groq]), \
         patch("core.should_apply.time.sleep"):
        result = _llm_score("some JD text")
    assert result["paid_calls"] == 2       # Groq is free tier


# ── Agent tone/consistency fixes (Oct 6 2026) ───────────────────────────────

def test_years_label_never_rounds_up_into_a_plus_claim():
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "engine")))
    from graph import compute_years_label
    assert compute_years_label("Nov 2020 - Jul 2024") == "nearly 4"   # 3y8m: the candidate's real TCS tenure
    assert compute_years_label("Jan 2020 - Mar 2024") == "4+"         # 4y2m
    assert compute_years_label("Jan 2021 - Apr 2024") == "3+"         # 3y3m


def test_dynamic_programming_is_not_a_banned_phrase_but_tells_are():
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "engine")))
    from graph import banned_phrase_hits
    assert banned_phrase_hits("Stress-tested graph traversal and dynamic programming problems") == []
    assert banned_phrase_hits("A dynamic, fast-paced team") == ["dynamic"]
    assert set(banned_phrase_hits("seamless, comprehensive end-to-end access")) == {"seamless", "comprehensive", "end-to-end"}


def test_too_many_ing_endings_get_flagged_but_a_couple_are_fine():
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "engine")))
    from graph import validate_writer_output
    master = {"experience": [{"company": "Acme", "bullets": [{"text": "x"}]}], "projects": []}
    tail = "Built the API layer for the billing service, reducing manual work for the team."
    plain = "Built the API layer for the billing service."
    many = {"tailored_bullets": [{"company_name": "Acme", "rewritten_text": tail}] * 6 +
                                [{"company_name": "Acme", "rewritten_text": plain}] * 2,
            "tailored_summary": "Engineer with 20K+ users served."}
    flagged = [f for f in validate_writer_output(many, master)["flagged_bullets"]
               if any("participial" in i for i in f["issues"])]
    assert flagged, "6 of 8 bullets ending in '-ing' clauses must be flagged"
    few = {"tailored_bullets": [{"company_name": "Acme", "rewritten_text": tail}] * 2 +
                               [{"company_name": "Acme", "rewritten_text": plain}] * 6,
           "tailored_summary": "Engineer with 20K+ users served."}
    assert not [f for f in validate_writer_output(few, master)["flagged_bullets"]
                if any("participial" in i for i in f["issues"])]
