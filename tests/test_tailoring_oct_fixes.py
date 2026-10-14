"""
tests/test_tailoring_oct_fixes.py

Regressions from the first real tailored run after the Oct 2026 resume overhaul
(Example Corp, Software Engineer II, Java):
  1. the summary printed its opening sentence twice (metric backstop only looked
     at the bullets selected for that JD, so a real "20K+" looked missing)
  2. "Pursuing an M.S." for a degree completed in May 2026
  3. the strongest bullet (the 20K+-user app) was dropped for a Java role
  4. a rewrite kept the source's words but ADDED a claim ("standardizing build,
     test, and release stages across the team")
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "engine")))

import graph
from graph import (enforce_bullet_traceability, fix_degree_status, summary_metric_backstop,
                   strategist_node, SelectedBullet, StrategistSelection)

MASTER = {
    "summary": ("Software engineer with nearly 4 years of production experience shipping Angular, Ionic and "
                "Spring Boot apps for a retail client with 20K+ users. Built a 6-node LangGraph "
                "job-application agent as a personal project."),
    "education": [{"degree": "M.S. Computer Engineering", "duration": "Aug 2024 - May 2026",
                   "graduation_status": "completed"}],
    "experience": [{"company": "TCS", "bullets": [
        {"text": "Shipped a production Angular and Ionic hybrid mobile app; it served 20K+ active users.", "base_priority": 1},
        {"text": "Integrated 30+ REST APIs into the Angular app and validated them in Postman."},
    ]}],
}


# 1 ─ backstop ──────────────────────────────────────────────────────────────
def test_a_real_metric_from_an_unselected_bullet_is_not_treated_as_missing():
    s = "Software engineer building Spring Boot apps for a retail platform serving 20K+ users."
    assert summary_metric_backstop(s, MASTER) == s


def test_backstop_never_inserts_a_near_copy_of_an_existing_sentence():
    s = ("Software engineer with nearly 4 years of production experience shipping Angular and Spring Boot apps "
         "for a retail client. Completed an M.S. in Computer Science at Example University.")
    out = summary_metric_backstop(s, MASTER)
    assert out.count("Software engineer with nearly 4 years") == 1


def test_backstop_still_inserts_when_the_summary_truly_has_no_metric_and_no_copy():
    s = "Backend engineer who likes clean APIs. Completed an M.S. at Example University."
    out = summary_metric_backstop(s, MASTER)
    assert "20K+" in out and out.startswith("Backend engineer who likes clean APIs.")


# 2 ─ degree status ─────────────────────────────────────────────────────────
def test_completed_degree_is_never_pursuing():
    assert fix_degree_status("Pursuing an M.S. in Computer Engineering.", MASTER) == "Completed an M.S. in Computer Engineering."
    assert fix_degree_status("Engineer, currently pursuing an M.S.", MASTER) == "Engineer, completed an M.S."
    in_progress = {**MASTER, "education": [{"duration": "Aug 2025 - May 2099", "graduation_status": "in_progress"}]}
    assert fix_degree_status("Pursuing an M.S.", in_progress) == "Pursuing an M.S."


# 3 ─ anchor bullet ─────────────────────────────────────────────────────────
def test_anchor_bullet_is_always_selected_for_its_employer():
    master = {
        "experience": [{"company": "TCS", "title": "Engineer", "resume_max_bullets": 2, "bullets": [
            {"text": "Anchor: shipped the 20K+-user app", "is_quantified": True, "base_priority": 1},
            {"text": "Debugged production issues", "is_quantified": False},
            {"text": "Wrote Spring Boot endpoints", "is_quantified": False},
        ]}],
        "projects": [{"name": "P", "domain": "x", "tech_stack": ["Python"]}],
    }
    fake = StrategistSelection(
        selected_bullets=[
            SelectedBullet(company_name="TCS", original_text="Debugged production issues", relevance_score=3, selection_reason="r"),
            SelectedBullet(company_name="TCS", original_text="Wrote Spring Boot endpoints", relevance_score=5, selection_reason="r"),
        ],
        selected_project_names=["P"],
    )
    with patch.object(graph, "invoke_with_retry", return_value=fake):
        result = strategist_node({"job_description": "Java backend role", "master_data": master})
    picked = [b["original_text"] for b in result["selected_content"]["experience"]]
    assert "Anchor: shipped the 20K+-user app" in picked
    assert "Wrote Spring Boot endpoints" in picked          # the strongest JD pick survives
    assert "Debugged production issues" not in picked       # the weakest pick made room
    assert len(picked) == 2                                  # count unchanged


# 4 ─ added content ─────────────────────────────────────────────────────────
_SRC = [{"company_name": "TCS", "original_text": "Moved releases from manual steps to GitLab CI and Jenkins pipelines across development and staging."}]


def test_the_real_embellished_rewrite_is_reverted():
    rewrite = [{"company_name": "TCS", "rewritten_text":
                "Automated CI/CD pipelines using GitLab CI and Jenkins, reducing manual deployment steps and "
                "standardizing build, test, and release stages across the team."}]
    out, reverts = enforce_bullet_traceability(rewrite, _SRC)
    assert reverts and "added too much" in reverts[0][1]
    assert out[0]["rewritten_text"] == _SRC[0]["original_text"]


def test_a_light_rewrite_with_a_verified_jd_keyword_is_kept():
    rewrite = [{"company_name": "TCS", "rewritten_text":
                "Moved releases from manual steps to CI/CD pipelines in GitLab CI and Jenkins across development and staging."}]
    out, reverts = enforce_bullet_traceability(rewrite, _SRC, allowed_terms=["CI/CD"])
    assert reverts == [] and out[0]["rewritten_text"].startswith("Moved releases")
