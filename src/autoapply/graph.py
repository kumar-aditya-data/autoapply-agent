"""The LangGraph pipeline: Parser -> Matcher -> Writer -> Reviewer -> human checkpoint.

Verified against langgraph 1.2 (docs.langchain.com/oss/python/langgraph/interrupts):
``interrupt()`` pauses the graph (requires a checkpointer + thread_id), the
caller sees ``result["__interrupt__"]``, and ``Command(resume=...)`` continues.
On resume the interrupted node re-runs from its first line, so
``human_review`` calls ``interrupt()`` before doing anything else.

The graph has no "submit" step. ``finalize`` only writes files for the human
to send themselves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from autoapply.documents import CV
from autoapply.grounding import run_rule_checks
from autoapply.llm import LLMClient
from autoapply.models import Draft, MatchReport, ParsedJD, ReviewFlag, ReviewReport
from autoapply.prompts import MATCHER_SYSTEM, PARSER_SYSTEM, REVIEWER_SYSTEM, WRITER_SYSTEM
from autoapply.scamcheck import ScamScreen, rules_only
from autoapply.usage import UsageLog, format_table, summarise

Action = Literal["approve", "revise", "reject"]


class AppState(TypedDict, total=False):
    """Graph state. Pydantic outputs are stored as plain dicts so checkpoints serialise cleanly."""

    run_id: str
    jd_text: str
    screening: dict[str, Any]
    cv_lines: dict[str, str]
    cv_source: str
    parsed_jd: dict[str, Any]
    match: dict[str, Any]
    draft: dict[str, Any]
    review: dict[str, Any]
    feedback: str
    auto_revisions: int
    human_rounds: int
    decision: str
    route: str
    output_dir: str


def _cv(state: AppState) -> CV:
    return CV(lines=state["cv_lines"], source=state.get("cv_source", ""))


def _jd_block(text: str) -> str:
    return f"<job_description>\n{text}\n</job_description>"


def _feedback_from_review(review: ReviewReport) -> str:
    lines = ["The reviewer found these problems. Fix all of them:"]
    for f in review.flags:
        if f.severity in {"high", "medium"}:
            lines.append(f"- [{f.severity}] \"{f.text}\": {f.issue}")
    return "\n".join(lines)


def build_graph(llm: LLMClient, output_root: str | Path, checkpointer: Any, max_auto_revisions: int = 1,
                usage_log: UsageLog | None = None, scam_screen: ScamScreen | None = None):
    """Compile the pipeline.

    Args:
        llm: any ``LLMClient`` (real Anthropic client or a test fake).
        output_root: folder under which approved runs are written.
        checkpointer: a LangGraph checkpointer (``SqliteSaver`` in the CLI, ``InMemorySaver`` in tests).
        max_auto_revisions: Writer retries after a failed review before a human sees the draft.
        usage_log: if given, the token summary is written alongside approved outputs.
        scam_screen: trained scam classifier; without it only the red-flag rules run.
    """
    output_root = Path(output_root)

    def screen_jd(state: AppState) -> AppState:
        # Deterministic and free: runs before any model call.
        text = state["jd_text"]
        return {"screening": scam_screen.assess(text) if scam_screen else rules_only(text)}

    def parse_jd(state: AppState) -> AppState:
        jd = llm.structured(run_id=state["run_id"], agent="parser", output_model=ParsedJD,
                            system=PARSER_SYSTEM, cached_context=None, user=_jd_block(state["jd_text"]))
        return {"parsed_jd": jd.model_dump()}

    def match_cv(state: AppState) -> AppState:
        user = ("Parsed job requirements (JSON):\n" + json.dumps(state["parsed_jd"], indent=2)
                + "\n\nScore the candidate's fit.")
        match = llm.structured(run_id=state["run_id"], agent="matcher", output_model=MatchReport,
                               system=MATCHER_SYSTEM, cached_context=_cv(state).numbered(), user=user)
        return {"match": match.model_dump()}

    def write_draft(state: AppState) -> AppState:
        parts = [
            "Parsed job requirements (JSON):\n" + json.dumps(state["parsed_jd"], indent=2),
            "Fit analysis (JSON):\n" + json.dumps(state["match"], indent=2),
        ]
        if state.get("draft") and state.get("feedback"):
            parts.append("Your previous draft (JSON):\n" + json.dumps(state["draft"], indent=2))
            parts.append("Feedback to address:\n" + state["feedback"])
        parts.append("Write the draft.")
        draft = llm.structured(run_id=state["run_id"], agent="writer", output_model=Draft,
                               system=WRITER_SYSTEM, cached_context=_cv(state).numbered(), user="\n\n".join(parts))
        return {"draft": draft.model_dump()}

    def review_draft(state: AppState) -> AppState:
        cv = _cv(state)
        draft = Draft.model_validate(state["draft"])
        jd = ParsedJD.model_validate(state["parsed_jd"])
        rule_flags = run_rule_checks(draft, cv, jd)
        user = ("Draft to fact-check (JSON):\n" + json.dumps(state["draft"], indent=2)
                + "\n\nAutomated rule checks already raised (verify, do not just repeat):\n"
                + (json.dumps([f.model_dump() for f in rule_flags], indent=2) if rule_flags else "none"))
        llm_review = llm.structured(run_id=state["run_id"], agent="reviewer", output_model=ReviewReport,
                                    system=REVIEWER_SYSTEM, cached_context=cv.numbered(), user=user)
        flags: list[ReviewFlag] = rule_flags + [f.model_copy(update={"source": "llm"}) for f in llm_review.flags]
        verdict = "needs_changes" if any(f.severity == "high" for f in flags) else "pass"
        review = ReviewReport(verdict=verdict, flags=flags, notes=llm_review.notes)

        update: AppState = {"review": review.model_dump(), "route": "human_review"}
        if verdict == "needs_changes" and state.get("auto_revisions", 0) < max_auto_revisions:
            update["auto_revisions"] = state.get("auto_revisions", 0) + 1
            update["feedback"] = _feedback_from_review(review)
            update["route"] = "write_draft"
        return update

    def route_after_review(state: AppState) -> str:
        return state["route"]

    def human_review(state: AppState) -> AppState:
        # interrupt() MUST be the first call: this node re-runs from the top on resume.
        decision = interrupt({
            "run_id": state["run_id"],
            "job": {k: state["parsed_jd"].get(k) for k in ("title", "company", "location")},
            "match": {k: state["match"].get(k) for k in ("score", "recommendation", "summary", "gaps")},
            "draft": state["draft"],
            "review": state["review"],
            "suspicious_instructions": state["parsed_jd"].get("suspicious_instructions", []),
            "screening": state.get("screening", {}),
            "scam_signals": state["parsed_jd"].get("scam_signals", []),
            "instructions": "Resume with {'action': 'approve'|'revise'|'reject', 'feedback': '...'}",
        })
        action = str(decision.get("action", "")).lower()
        if action not in {"approve", "revise", "reject"}:
            raise ValueError(f"unknown review action: {action!r}")
        update: AppState = {"decision": action}
        if action == "revise":
            feedback = str(decision.get("feedback", "")).strip()
            if not feedback:
                raise ValueError("revise needs feedback text")
            update["feedback"] = "Human reviewer feedback:\n" + feedback
            update["human_rounds"] = state.get("human_rounds", 0) + 1
            update["auto_revisions"] = 0  # fresh auto-revision allowance for the new draft
        return update

    def route_after_human(state: AppState) -> str:
        return {"approve": "finalize", "revise": "write_draft"}.get(state["decision"], END)

    def finalize(state: AppState) -> AppState:
        out = output_root / state["run_id"]
        write_outputs(out, state, usage_log)
        return {"output_dir": str(out)}

    g = StateGraph(AppState)
    g.add_node("screen_jd", screen_jd)
    g.add_node("parse_jd", parse_jd)
    g.add_node("match_cv", match_cv)
    g.add_node("write_draft", write_draft)
    g.add_node("review_draft", review_draft)
    g.add_node("human_review", human_review)
    g.add_node("finalize", finalize)
    g.add_edge(START, "screen_jd")
    g.add_edge("screen_jd", "parse_jd")
    g.add_edge("parse_jd", "match_cv")
    g.add_edge("match_cv", "write_draft")
    g.add_edge("write_draft", "review_draft")
    g.add_conditional_edges("review_draft", route_after_review, ["write_draft", "human_review"])
    g.add_conditional_edges("human_review", route_after_human, ["finalize", "write_draft", END])
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def write_outputs(out: Path, state: AppState, usage_log: UsageLog | None = None) -> list[Path]:
    """Write the human-approved files. Nothing here sends or submits anything."""
    out.mkdir(parents=True, exist_ok=True)
    jd = ParsedJD.model_validate(state["parsed_jd"])
    match = MatchReport.model_validate(state["match"])
    draft = Draft.model_validate(state["draft"])
    review = ReviewReport.model_validate(state["review"])
    files = {
        "STATUS.md": (f"# {jd.title} at {jd.company}\n\nApproved by a human reviewer. **NOT submitted.**\n"
                      "autoapply never submits applications: review the files and apply yourself.\n"),
        "cover_letter.md": draft.cover_letter.strip() + "\n",
        "cv_suggestions.md": render_bullets(draft),
        "match_report.md": render_match(match),
        "review.md": render_review(review),
    }
    if usage_log is not None:
        files["usage.md"] = "# Token usage\n\n" + format_table(summarise(usage_log.records(state["run_id"]))) + "\n"
    paths = []
    for name, text in files.items():
        p = out / name
        p.write_text(text, encoding="utf-8")
        paths.append(p)
    return paths


def render_match(match: MatchReport) -> str:
    """Markdown fit report."""
    rows = "\n".join(f"| {r.requirement} | {r.status} | {', '.join(r.cv_lines) or '-'} | {r.reasoning} |"
                     for r in match.requirements)
    return (f"# Fit: {match.score}/100 ({match.recommendation})\n\n{match.summary}\n\n"
            "| Requirement | Status | CV lines | Reasoning |\n|---|---|---|---|\n" + rows + "\n\n"
            "**Strengths:** " + "; ".join(match.strengths) + "\n\n**Gaps:** " + "; ".join(match.gaps) + "\n")


def render_bullets(draft: Draft) -> str:
    """Markdown CV bullet suggestions."""
    lines = ["# CV bullet suggestions\n"]
    for b in draft.bullet_suggestions:
        lines.append(f"- **{b.cv_line}** (targets: {b.targets_requirement})\n  - Original: {b.original}\n"
                     f"  - Suggested: {b.suggested}")
    if draft.gaps_acknowledged:
        lines.append("\n## Gaps deliberately not claimed\n" + "\n".join(f"- {g}" for g in draft.gaps_acknowledged))
    return "\n".join(lines) + "\n"


def render_review(review: ReviewReport) -> str:
    """Markdown review report."""
    if not review.flags:
        return f"# Review: {review.verdict}\n\nNo issues flagged.\n{review.notes}\n"
    rows = "\n".join(f"| {f.severity} | {f.source} | {f.text} | {f.issue} |" for f in review.flags)
    return (f"# Review: {review.verdict}\n\n{review.notes}\n\n| Severity | Source | Text | Issue |\n|---|---|---|---|\n"
            + rows + "\n")
