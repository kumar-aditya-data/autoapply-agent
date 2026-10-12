"""Batch evaluation on real job postings.

Runs the full agent pipeline on each posting up to the human checkpoint, records
what happened, then *rejects* the run so nothing is ever finalised. This is an
offline quality check of the agents (fit scoring, fabrication flags, scam
signals, token cost), not an application workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from autoapply.documents import CV
from autoapply.graph import build_graph
from autoapply.llm import LLMClient
from autoapply.scamcheck import ScamScreen
from autoapply.usage import UsageLog, summarise


def evaluate_postings(postings: pd.DataFrame, cv: CV, llm: LLMClient, usage_log: UsageLog, eval_id: str,
                      scratch_dir: str | Path, max_auto_revisions: int = 1,
                      scam_screen: ScamScreen | None = None) -> pd.DataFrame:
    """Run each posting through the graph and collect one row of metrics per posting.

    Args:
        postings: rows with ``jd_text`` and, if available, ``job_id``, ``title``, ``fraudulent``.
    """
    graph = build_graph(llm, Path(scratch_dir) / "never_written", InMemorySaver(), max_auto_revisions,
                        usage_log, scam_screen)
    rows: list[dict[str, Any]] = []
    for i, (_, posting) in enumerate(postings.iterrows(), start=1):
        run_id = f"{eval_id}-{i:03d}"
        config = {"configurable": {"thread_id": run_id}}
        row: dict[str, Any] = {"run_id": run_id, "job_id": posting.get("job_id"), "title": posting.get("title"),
                               "label_fraudulent": posting.get("fraudulent")}
        try:
            result = graph.invoke({"run_id": run_id, "jd_text": posting["jd_text"], "cv_lines": cv.lines,
                                   "cv_source": cv.source, "auto_revisions": 0, "human_rounds": 0}, config)
            payload = result["__interrupt__"][0].value
            state = graph.get_state(config).values
            flags = payload["review"]["flags"]
            row.update({
                "status": "ok",
                "scam_risk": payload["screening"].get("risk"),
                "scam_probability": payload["screening"].get("probability"),
                "scam_red_flags": len(payload["screening"].get("red_flags", [])),
                "parser_scam_signals": len(payload.get("scam_signals", [])),
                "requirements_found": len(state["match"].get("requirements", [])),
                "match_score": payload["match"]["score"],
                "recommendation": payload["match"]["recommendation"],
                "review_verdict": payload["review"]["verdict"],
                "flags_high": sum(f["severity"] == "high" for f in flags),
                "flags_rule": sum(f["source"] == "rule" for f in flags),
                "flags_llm": sum(f["source"] == "llm" for f in flags),
                "auto_revisions": state.get("auto_revisions", 0),
                "claims": len(payload["draft"].get("claims", [])),
            })
            graph.invoke(Command(resume={"action": "reject"}), config)  # never finalise in an eval
        except Exception as exc:  # keep going; record the failure honestly
            row.update({"status": f"error: {type(exc).__name__}: {exc}"[:200]})
        tokens = summarise(usage_log.records(run_id))
        if tokens:
            total = tokens[-1]
            row.update({k: total[k] for k in ("calls", "input_tokens", "cache_read_input_tokens",
                                              "cache_creation_input_tokens", "output_tokens")})
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_eval(results: pd.DataFrame) -> str:
    """Markdown summary of an evaluation run."""
    ok = results[results["status"] == "ok"]
    lines = [f"# Evaluation on real job postings", "",
             f"Postings run: {len(results)} (succeeded: {len(ok)}, failed: {len(results) - len(ok)})", ""]
    if "requirements_found" in ok and (ok["requirements_found"] == 0).any():
        lines += [f"{int((ok['requirements_found'] == 0).sum())} posting(s) had no extractable requirements and are "
                  "excluded from fit metrics.", ""]
    scored = ok[ok["requirements_found"] > 0] if "requirements_found" in ok else ok
    if ok.empty:
        return "\n".join(lines) + "\n"
    lines += [
        "| Metric | Value |", "|---|---:|",
        f"| Mean match score | {scored['match_score'].mean():.1f} |",
        f"| Recommendation apply / stretch / skip | {(scored['recommendation'] == 'apply').sum()} / "
        f"{(scored['recommendation'] == 'stretch').sum()} / {(scored['recommendation'] == 'skip').sum()} |",
        f"| Drafts passing review at checkpoint | {(ok['review_verdict'] == 'pass').mean():.0%} |",
        f"| Runs needing an auto-revision | {(ok['auto_revisions'] > 0).mean():.0%} |",
        f"| Mean high-severity flags at checkpoint | {ok['flags_high'].mean():.2f} |",
        f"| Mean rule / LLM flags | {ok['flags_rule'].mean():.2f} / {ok['flags_llm'].mean():.2f} |",
    ]
    if "input_tokens" in ok:
        lines += [f"| Mean model calls per posting | {ok['calls'].mean():.1f} |",
                  f"| Mean input / cache-read / output tokens | {ok['input_tokens'].mean():,.0f} / "
                  f"{ok['cache_read_input_tokens'].mean():,.0f} / {ok['output_tokens'].mean():,.0f} |"]
    if ok["label_fraudulent"].notna().any():
        fraud = ok[ok["label_fraudulent"] == 1]
        legit = ok[ok["label_fraudulent"] == 0]
        if len(fraud):
            lines += [f"| Labelled scams: screen risk medium/high | {fraud['scam_risk'].isin(['medium', 'high']).mean():.0%} "
                      f"(n={len(fraud)}) |",
                      f"| Labelled scams: Parser reported scam signals | {(fraud['parser_scam_signals'] > 0).mean():.0%} |"]
        if len(legit):
            lines += [f"| Legit postings: screen risk medium/high (false alarms) | "
                      f"{legit['scam_risk'].isin(['medium', 'high']).mean():.0%} (n={len(legit)}) |",
                      f"| Legit postings: Parser reported scam signals | {(legit['parser_scam_signals'] > 0).mean():.0%} |"]
    return "\n".join(lines) + "\n"
