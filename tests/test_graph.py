"""Graph flow: human checkpoint, approve/revise/reject, auto-revision cap."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from autoapply.graph import build_graph
from autoapply.models import ReviewReport
from tests.conftest import HIGH_FLAG_REVIEW, FakeLLM, clean_draft, fabricated_draft

CONFIG = {"configurable": {"thread_id": "test-run"}}


def _graph(llm, tmp_path, max_auto_revisions=1):
    return build_graph(llm, tmp_path / "outputs", InMemorySaver(), max_auto_revisions)


def test_pauses_at_human_checkpoint_with_draft(tmp_path, initial_state) -> None:
    llm = FakeLLM()
    result = _graph(llm, tmp_path).invoke(initial_state, CONFIG)
    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["draft"]["cover_letter"].startswith("Dear Hiring Manager")
    assert payload["match"]["score"] == 72
    assert payload["suspicious_instructions"]  # injection surfaced to the human
    assert llm.agents_called() == ["parser", "matcher", "writer", "reviewer"]
    assert not (tmp_path / "outputs").exists()  # nothing final before approval


def test_approve_writes_files_and_never_submits(tmp_path, initial_state) -> None:
    llm = FakeLLM()
    g = _graph(llm, tmp_path)
    g.invoke(initial_state, CONFIG)
    calls_before = len(llm.calls)
    result = g.invoke(Command(resume={"action": "approve"}), CONFIG)
    assert len(llm.calls) == calls_before  # resuming the checkpoint makes no model calls
    out = tmp_path / "outputs" / "test-run"
    assert result["output_dir"] == str(out)
    assert {p.name for p in out.iterdir()} >= {"STATUS.md", "cover_letter.md", "cv_suggestions.md",
                                               "match_report.md", "review.md"}
    assert "NOT submitted" in (out / "STATUS.md").read_text(encoding="utf-8")


def test_reject_writes_nothing(tmp_path, initial_state) -> None:
    g = _graph(FakeLLM(), tmp_path)
    g.invoke(initial_state, CONFIG)
    result = g.invoke(Command(resume={"action": "reject"}), CONFIG)
    assert result["decision"] == "reject"
    assert not (tmp_path / "outputs").exists()


def test_revise_reruns_writer_with_feedback_then_pauses_again(tmp_path, initial_state) -> None:
    llm = FakeLLM(drafts=[clean_draft(), clean_draft("Revised.")])
    g = _graph(llm, tmp_path)
    g.invoke(initial_state, CONFIG)
    result = g.invoke(Command(resume={"action": "revise", "feedback": "Mention forecasting."}), CONFIG)
    assert "__interrupt__" in result
    assert "Revised." in result["__interrupt__"][0].value["draft"]["cover_letter"]
    writer_calls = [c for c in llm.calls if c["agent"] == "writer"]
    assert len(writer_calls) == 2
    assert "Mention forecasting." in writer_calls[1]["user"]
    assert "previous draft" in writer_calls[1]["user"]


def test_failed_review_triggers_one_auto_revision_then_human(tmp_path, initial_state) -> None:
    llm = FakeLLM(drafts=[fabricated_draft(), clean_draft()],
                  reviews=[HIGH_FLAG_REVIEW, ReviewReport(verdict="pass")])
    result = _graph(llm, tmp_path, max_auto_revisions=1).invoke(initial_state, CONFIG)
    assert llm.agents_called() == ["parser", "matcher", "writer", "reviewer", "writer", "reviewer"]
    assert "__interrupt__" in result
    assert "reviewer found" in [c for c in llm.calls if c["agent"] == "writer"][1]["user"]


def test_auto_revision_cap_respected(tmp_path, initial_state) -> None:
    llm = FakeLLM(drafts=[fabricated_draft()], reviews=[HIGH_FLAG_REVIEW])
    result = _graph(llm, tmp_path, max_auto_revisions=2).invoke(initial_state, CONFIG)
    assert llm.agents_called().count("writer") == 3  # first draft + 2 auto-revisions
    payload = result["__interrupt__"][0].value
    assert payload["review"]["verdict"] == "needs_changes"  # human sees the unresolved flags


def test_zero_auto_revisions_goes_straight_to_human(tmp_path, initial_state) -> None:
    llm = FakeLLM(drafts=[fabricated_draft()], reviews=[HIGH_FLAG_REVIEW])
    _graph(llm, tmp_path, max_auto_revisions=0).invoke(initial_state, CONFIG)
    assert llm.agents_called().count("writer") == 1


def test_rule_flags_merged_into_review(tmp_path, initial_state) -> None:
    llm = FakeLLM(drafts=[fabricated_draft()], reviews=[ReviewReport(verdict="pass")])
    result = _graph(llm, tmp_path, max_auto_revisions=0).invoke(initial_state, CONFIG)
    review = result["__interrupt__"][0].value["review"]
    assert review["verdict"] == "needs_changes"  # rule checks override the LLM's "pass"
    assert any(f["source"] == "rule" for f in review["flags"])
