"""
tests/test_candidate_context_from_master_data.py

The scoring prompt's CANDIDATE CONTEXT block and the profile summary's visa line used to be
hardcoded in core/should_apply.py and core/profile_cache.py. They are now read from the
`candidate_context` section of the (gitignored) master data file, so personal details such
as work authorization never ship with the code.

Run: pytest tests/test_candidate_context_from_master_data.py -v
"""
import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.should_apply import _load_candidate_context, _CANDIDATE_CONTEXT_FALLBACK, _static_prompt
from core.profile_cache import generate_profile_cache


def _write(tmp_path, data):
    p = tmp_path / "master.json"
    p.write_text(json.dumps(data))
    return str(p)


def test_context_lines_become_bullets(tmp_path):
    path = _write(tmp_path, {"candidate_context": {"scoring_context": ["First fact", "Second fact"]}})
    with patch("core.should_apply.MASTER_DATA_PATH", path):
        assert _load_candidate_context() == "- First fact\n- Second fact"


def test_blank_lines_are_skipped(tmp_path):
    path = _write(tmp_path, {"candidate_context": {"scoring_context": ["Real fact", "  ", ""]}})
    with patch("core.should_apply.MASTER_DATA_PATH", path):
        assert _load_candidate_context() == "- Real fact"


def test_falls_back_when_section_missing(tmp_path):
    path = _write(tmp_path, {"technical_skills": {"Languages": ["Python"]}})
    with patch("core.should_apply.MASTER_DATA_PATH", path):
        assert _load_candidate_context() == _CANDIDATE_CONTEXT_FALLBACK


def test_falls_back_when_file_missing_or_malformed(tmp_path):
    with patch("core.should_apply.MASTER_DATA_PATH", str(tmp_path / "nope.json")):
        assert _load_candidate_context() == _CANDIDATE_CONTEXT_FALLBACK
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with patch("core.should_apply.MASTER_DATA_PATH", str(bad)):
        assert _load_candidate_context() == _CANDIDATE_CONTEXT_FALLBACK


def test_edit_takes_effect_on_next_call(tmp_path):
    path = _write(tmp_path, {"candidate_context": {"scoring_context": ["Old"]}})
    with patch("core.should_apply.MASTER_DATA_PATH", path):
        assert "Old" in _load_candidate_context()
        _write(tmp_path, {"candidate_context": {"scoring_context": ["New"]}})
        assert "New" in _load_candidate_context()


def test_scoring_prompt_contains_the_master_data_context(tmp_path):
    path = _write(tmp_path, {"candidate_context": {"scoring_context": ["Unique marker fact 12345"]}})
    with patch("core.should_apply.MASTER_DATA_PATH", path):
        assert "CANDIDATE CONTEXT:\n- Unique marker fact 12345" in _static_prompt()


def test_code_itself_no_longer_hardcodes_work_authorization():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for rel in ("core/should_apply.py", "core/profile_cache.py"):
        text = open(os.path.join(root, rel), encoding="utf-8").read()
        assert "needs H-1B sponsorship for long-term employment" not in text
        assert "Visa: F-" not in text


def test_profile_visa_line_comes_from_master_data():
    md = {"candidate_context": {"visa_line": "Visa: example status."}, "personal_info": {}, "experience": []}
    assert "Visa: example status." in generate_profile_cache(md)


def test_profile_has_no_visa_line_when_not_provided():
    out = generate_profile_cache({"personal_info": {}, "experience": []})
    assert "Visa:" not in out
    assert out.splitlines()[:2] == ["CANDIDATE PROFILE CACHE", "======================="]


# ---- title corrections come from master data, not from code ----
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "engine")))
from graph import title_corrections, validate_cross_document_consistency  # noqa: E402

_CORR = {"company": "Acme", "short_name": "Acme", "wrong_titles_summary": ["Software Developer"],
         "wrong_titles_bullets": ["Software Developer"], "auto_replace": ["Software Developer"]}


def _master(with_corrections):
    md = {"experience": [{"company": "Acme", "title": "Software Engineer",
                          "bullets": [{"text": "Built things.", "technologies_used": ["Python"]}]}],
          "projects": [], "personal_info": {}}
    if with_corrections:
        md["candidate_context"] = {"title_corrections": [_CORR]}
    return md


def _draft(summary):
    return {"tailored_summary": summary, "tailored_bullets": [], "tailored_project_bullets": []}


def test_title_corrections_empty_when_missing():
    assert title_corrections({}) == []
    assert title_corrections({"candidate_context": {}}) == []


def test_wrong_title_flagged_when_configured():
    report = validate_cross_document_consistency(_draft("A Software Developer at Acme."), _master(True))
    assert [i for i in report["issues"] if i["type"] == "title_mismatch"]


def test_no_title_checks_when_not_configured():
    report = validate_cross_document_consistency(_draft("A Software Developer at Acme."), _master(False))
    assert not [i for i in report["issues"] if i["type"] == "title_mismatch"]


def test_code_no_longer_hardcodes_an_employer_name():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    assert "Tata Consultancy Services" not in open(os.path.join(root, "engine", "graph.py"), encoding="utf-8").read()


# ── summary brief: lead sentence and example numbers come from master data ──────────────

from engine.graph import (  # noqa: E402
    metric_examples, summary_lead, _METRIC_EXAMPLES_FALLBACK, _SUMMARY_LEAD_FALLBACK,
)


def test_summary_lead_and_metric_examples_read_from_candidate_context():
    md = {"candidate_context": {"summary_lead": "shipping the billing app (5K+ users).",
                                "metric_examples": "5K+ users, 12 APIs"}}
    assert summary_lead(md) == "shipping the billing app (5K+ users)"      # trailing period stripped: the prompt adds its own
    assert metric_examples(md) == "5K+ users, 12 APIs"


def test_summary_lead_and_metric_examples_fall_back_when_missing_or_blank():
    for md in ({}, {"candidate_context": {}}, {"candidate_context": {"summary_lead": "  ", "metric_examples": ""}}):
        assert summary_lead(md) == _SUMMARY_LEAD_FALLBACK
        assert metric_examples(md) == _METRIC_EXAMPLES_FALLBACK


def test_example_master_data_defines_both_keys():
    example = os.path.join(os.path.dirname(__file__), "..", "Ankush_Master_Data.example.json")
    with open(example) as f:
        cc = json.load(f)["candidate_context"]
    assert cc["summary_lead"].strip() and cc["metric_examples"].strip()


def test_graph_source_has_no_hardcoded_author_story_in_the_summary_brief():
    """The brief must not carry anyone's own employer or numbers; they live in master data."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "engine", "graph.py"), encoding="utf-8").read()
    for hardcoded in ("retail client", "UK life-insurance client", "e.g. 20K+ users", "e.g. 10K+ users"):
        assert hardcoded not in src, hardcoded
