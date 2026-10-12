"""The offline no-LLM baseline (AUTOAPPLY_MODEL=baseline)."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from autoapply.baseline import BaselineClient, _items, match, parse_jd, write
from autoapply.cli import main
from autoapply.documents import load_cv, load_jd
from autoapply.graph import build_graph
from autoapply.grounding import run_rule_checks
from tests.conftest import SAMPLE_CV, SAMPLE_JD


@pytest.fixture
def jd():
    return parse_jd(load_jd(str(SAMPLE_JD)))


def test_parser_sections_and_injection(jd) -> None:
    assert jd.title.startswith("Supply Chain Data Analyst") and jd.company == "Nordwind Logistik GmbH"
    assert "Strong SQL and Python (pandas) skills" in jd.must_haves
    assert any("SAP EWM" in n for n in jd.nice_to_haves)
    assert jd.suspicious_instructions and not any("AI screening" in r for r in jd.must_haves + jd.nice_to_haves)


def test_items_keep_abbreviations_together() -> None:
    assert _items("Knowledge of forecasting methods (e.g. ARIMA, exponential smoothing).") == [
        "Knowledge of forecasting methods (e.g. ARIMA, exponential smoothing)."]


def test_matcher_cites_real_lines_and_stems(jd, cv) -> None:
    report = match(jd, cv.lines)
    by_req = {r.requirement: r for r in report.requirements}
    sql = by_req["Strong SQL and Python (pandas) skills"]
    assert sql.status == "met" and all(l in cv.lines for l in sql.cv_lines)
    forecasting = next(r for r in report.requirements if r.requirement.startswith("Knowledge of forecasting"))
    assert forecasting.status == "met"  # "forecasting" matched "Forecast" via stemming
    assert next(r for r in report.requirements if r.requirement.startswith("Lean")).status == "missing"
    assert 0 <= report.score <= 100


def test_writer_is_grounded_by_construction(jd, cv) -> None:
    draft = write(jd, match(jd, cv.lines), cv.lines)
    assert "Baseline draft" in draft.cover_letter
    assert draft.claims and all(cv.lines[c.cv_lines[0]] in c.sentence for c in draft.claims)
    high = [f for f in run_rule_checks(draft, cv, jd) if f.severity == "high"]
    assert high == []  # verbatim quotes cannot fail citation or number checks


def test_full_graph_offline(tmp_path, cv) -> None:
    graph = build_graph(BaselineClient(), tmp_path, InMemorySaver())
    config = {"configurable": {"thread_id": "b"}}
    result = graph.invoke({"run_id": "b", "jd_text": load_jd(str(SAMPLE_JD)), "cv_lines": cv.lines,
                           "cv_source": cv.source, "auto_revisions": 0, "human_rounds": 0}, config)
    payload = result["__interrupt__"][0].value
    assert payload["review"]["verdict"] == "pass" and payload["match"]["score"] > 50
    done = graph.invoke(Command(resume={"action": "approve"}), config)
    assert "NOT submitted" in (tmp_path / "b" / "STATUS.md").read_text(encoding="utf-8")
    assert done["decision"] == "approve"


def test_cli_uses_baseline_without_api_key(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("AUTOAPPLY_MODEL", "baseline")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("AUTOAPPLY_DATA_DIR", str(tmp_path))
    assert main(["--env-file", "none.env", "run", "--jd", str(SAMPLE_JD), "--cv", str(SAMPLE_CV),
                 "--no-interactive"]) == 0
    out = capsys.readouterr().out
    assert "with baseline" in out and "Baseline draft" in out


def test_unglue_scraped_bullets() -> None:
    from autoapply.baseline import _unglue
    text = "0-3 years in a data-driven roleStrong Excel skillsOutstanding communication | teamwork; PowerPoint"
    parts = _unglue(text).splitlines()
    assert parts == ["0-3 years in a data-driven role", "Strong Excel skills", "Outstanding communication",
                     "teamwork; PowerPoint"]


def test_no_requirements_is_not_scored_as_zero_fit() -> None:
    from autoapply.models import ParsedJD
    report = match(ParsedJD(title="x", company="y"), {"L01": "SQL"})
    assert report.requirements == [] and "not meaningful" in report.summary
