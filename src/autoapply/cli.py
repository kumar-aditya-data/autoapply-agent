"""Command-line interface.

    autoapply run --jd job.md --cv my_cv.docx          # interactive review at the checkpoint
    autoapply run --jd - --cv my_cv.docx --no-interactive   # paste JD via stdin; pause for later
    autoapply resume <run_id> --approve | --reject | --revise "make it shorter"
    autoapply usage [--run <run_id>]

    autoapply data sample --data data/raw --keywords "logistics,supply chain" --n 5   # real EMSCAD postings
    autoapply scam train --data data/raw        # train the free scam pre-screen on labelled postings
    autoapply scam check --jd job.md
    autoapply eval --data data/raw --cv my_cv.docx --n 10   # batch quality check (calls the API)

There is deliberately no command that submits an application.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import json
import uuid
from datetime import datetime
from typing import Any, Callable

import pandas as pd
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from autoapply.baseline import BaselineClient
from autoapply.config import BASELINE_ALIAS, ConfigError, Settings, load_settings
from autoapply.datasets import DatasetError, export_postings, load_emscad, select_postings
from autoapply.documents import InputError, load_cv, load_jd
from autoapply.evaluate import evaluate_postings, summarise_eval
from autoapply.graph import build_graph
from autoapply.llm import AnthropicClient, LLMClient, LLMError
from autoapply.scamcheck import ScamScreen, rules_only, train
from autoapply.usage import UsageLog, format_table, summarise

LLMFactory = Callable[[Settings, UsageLog], LLMClient]
RULE = "=" * 78


def _default_llm(settings: Settings, usage_log: UsageLog) -> LLMClient:
    if settings.model == BASELINE_ALIAS:  # offline, no API key needed
        return BaselineClient(usage_log)
    return AnthropicClient(settings, usage_log)


class Runtime:
    """Settings, usage log, checkpointer and compiled graph for one CLI invocation."""

    def __init__(self, settings: Settings, llm_factory: LLMFactory) -> None:
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.usage_log = UsageLog(settings.usage_log)
        self._conn = sqlite3.connect(settings.checkpoint_db, check_same_thread=False)
        self._llm_factory = llm_factory
        self.scam_screen = ScamScreen.load(settings.scam_model) if settings.scam_model.exists() else None
        self.graph = build_graph(llm_factory(settings, self.usage_log), settings.outputs_dir,
                                 SqliteSaver(self._conn), settings.max_auto_revisions, self.usage_log,
                                 self.scam_screen)

    def new_llm(self) -> LLMClient:
        """A fresh LLM client (used by batch evaluation)."""
        return self._llm_factory(self.settings, self.usage_log)

    def close(self) -> None:
        """Close the checkpoint database."""
        self._conn.close()


def _config(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}}


def show_checkpoint(payload: dict[str, Any]) -> None:
    """Print what the human needs to decide on."""
    job, match, draft, review = payload["job"], payload["match"], payload["draft"], payload["review"]
    print(f"\n{RULE}\nHUMAN REVIEW CHECKPOINT  (run {payload['run_id']})\n{RULE}")
    print(f"Role: {job.get('title')} at {job.get('company')} ({job.get('location') or 'n/a'})")
    print(f"Fit:  {match.get('score')}/100, recommendation: {match.get('recommendation')}")
    print(f"      {match.get('summary')}")
    if match.get("gaps"):
        print("Gaps: " + "; ".join(match["gaps"]))
    screening = payload.get("screening") or {}
    if screening:
        prob = screening.get("probability")
        print(f"Scam screen: risk {str(screening.get('risk')).upper()}"
              + (f" (model probability {prob:.0%})" if prob is not None else " (rules only; no trained model)"))
        if screening.get("top_terms"):
            print("   terms raising the score: " + ", ".join(screening["top_terms"]))
        for f in screening.get("red_flags", []):
            print(f"   ? check yourself (unvalidated rule): {f['why']}: \"{f['evidence']}\"")
    for s in payload.get("scam_signals", []):
        print(f"   ! Parser: {s}")
    if payload.get("suspicious_instructions"):
        print("\n!! The job description contains text aimed at AI systems (ignored):")
        for s in payload["suspicious_instructions"]:
            print(f"   - {s}")
    print(f"\n--- Cover letter draft ---\n{draft['cover_letter'].strip()}\n")
    print("--- CV bullet suggestions ---")
    for b in draft.get("bullet_suggestions", []):
        print(f"  [{b['cv_line']}] {b['suggested']}\n        (was: {b['original']})")
    print(f"\n--- Fabrication review: {review['verdict'].upper()} ---")
    if not review["flags"]:
        print("  No issues flagged.")
    for f in review["flags"]:
        print(f"  [{f['severity'].upper():6s}|{f['source']}] {f['text'][:90]}\n           -> {f['issue']}")
    print(RULE)


def _report_finish(result: dict[str, Any]) -> None:
    if result.get("decision") == "approve":
        print(f"\nApproved. Files written to {result.get('output_dir')}")
        print("Nothing has been submitted. Review the files and apply yourself.")
    elif result.get("decision") == "reject":
        print("\nRejected. No final files were written.")


def _drive(rt: Runtime, run_id: str, result: dict[str, Any], interactive: bool) -> int:
    """Show checkpoints and collect decisions until the run finishes or is paused."""
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        show_checkpoint(payload)
        if not interactive:
            print(f"\nPaused for human review. Resume with:\n  autoapply resume {run_id} --approve\n"
                  f"  autoapply resume {run_id} --revise \"<feedback>\"\n  autoapply resume {run_id} --reject")
            return 0
        decision = _ask()
        if decision is None:
            print(f"\nPaused. Resume later with: autoapply resume {run_id}")
            return 0
        result = rt.graph.invoke(Command(resume=decision), _config(run_id))
    _report_finish(result)
    _print_usage(rt, run_id)
    return 0


def _ask() -> dict[str, str] | None:
    while True:
        choice = input("\n[a]pprove  [r]evise  [x] reject  [p]ause > ").strip().lower()
        if choice in {"a", "approve"}:
            return {"action": "approve"}
        if choice in {"x", "reject"}:
            return {"action": "reject"}
        if choice in {"p", "pause"}:
            return None
        if choice in {"r", "revise"}:
            feedback = input("What should change? > ").strip()
            if feedback:
                return {"action": "revise", "feedback": feedback}
        print("Please choose a, r, x or p.")


def _print_usage(rt: Runtime, run_id: str) -> None:
    rows = summarise(rt.usage_log.records(run_id))
    if rows:
        print(f"\nToken usage for {run_id} (model {rt.settings.model}):\n{format_table(rows)}")


def cmd_run(args: argparse.Namespace, rt: Runtime) -> int:
    """Start a new run."""
    cv = load_cv(args.cv)
    jd_text = load_jd(args.jd)
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    print(f"Run {run_id}: parsing JD, matching CV, drafting and reviewing with {rt.settings.model}...")
    result = rt.graph.invoke({"run_id": run_id, "jd_text": jd_text, "cv_lines": cv.lines, "cv_source": cv.source,
                              "auto_revisions": 0, "human_rounds": 0}, _config(run_id))
    return _drive(rt, run_id, result, interactive=not args.no_interactive)


def cmd_resume(args: argparse.Namespace, rt: Runtime) -> int:
    """Continue a paused run."""
    config = _config(args.run_id)
    snapshot = rt.graph.get_state(config)
    if not snapshot.values:
        print(f"error: no run with id {args.run_id}", file=sys.stderr)
        return 2
    if not snapshot.next:
        print(f"Run {args.run_id} is already finished (decision: {snapshot.values.get('decision')}).")
        return 0
    if args.approve:
        decision: dict[str, str] | None = {"action": "approve"}
    elif args.reject:
        decision = {"action": "reject"}
    elif args.revise:
        decision = {"action": "revise", "feedback": args.revise}
    else:  # no flag: show the pending checkpoint and ask interactively
        show_checkpoint(snapshot.interrupts[0].value)
        decision = _ask()
        if decision is None:
            return 0
    result = rt.graph.invoke(Command(resume=decision), config)
    return _drive(rt, args.run_id, result, interactive=not args.no_interactive)


def _keywords(raw: str | None) -> list[str]:
    return [k.strip() for k in (raw or "").split(",") if k.strip()]


def cmd_data_sample(args: argparse.Namespace, rt: Runtime) -> int:
    """Export real EMSCAD postings as JD files."""
    df = load_emscad(args.data)
    fraud = {"legit": False, "fraud": True, "any": None}[args.label]
    sample = select_postings(df, _keywords(args.keywords), args.n, args.seed, fraudulent=fraud)
    for p in export_postings(sample, args.out):
        print(p)
    return 0


def cmd_scam_train(args: argparse.Namespace, rt: Runtime) -> int:
    """Train and evaluate the scam pre-screen on labelled EMSCAD postings."""
    df = load_emscad(args.data)
    print(f"Loaded {len(df):,} postings ({df['fraudulent'].sum():,} labelled fraudulent; "
          f"{df.attrs.get('duplicates_removed', 0):,} exact duplicates removed). Training...")
    screen, test = train(df, seed=args.seed)
    screen.save(rt.settings.scam_model)
    metrics_path = rt.settings.scam_model.with_name("scam_metrics.json")
    metrics_path.write_text(json.dumps(screen.metrics, indent=2), encoding="utf-8")
    test.to_csv(rt.settings.scam_model.with_name("scam_test_predictions.csv"), index=False)
    m = screen.metrics
    print(f"\nHeld-out test set: {m['n_test']:,} postings, {m['fraud_rate_test']:.1%} fraudulent")
    print(f"  Classifier PR-AUC {m['test_high_pr_auc']:.3f} | ROC-AUC {m['test_high_roc_auc']:.3f}")
    print(f"  'high' risk  (p >= {m['threshold_high']:.2f}): precision {m['test_high_precision']:.2f}, "
          f"recall {m['test_high_recall']:.2f}")
    print(f"  'medium'+    (p >= {m['threshold_medium']:.2f}): precision {m['test_medium_precision']:.2f}, "
          f"recall {m['test_medium_recall']:.2f}")
    print(f"  Red-flag rules alone: precision {m['rules_precision']:.2f}, recall {m['rules_recall']:.2f} "
          f"(flag {m['rules_flag_rate']:.1%} of postings)")
    print(f"\nSaved {rt.settings.scam_model} and {metrics_path}")
    return 0


def cmd_scam_check(args: argparse.Namespace, rt: Runtime) -> int:
    """Screen one job description without calling any model."""
    text = load_jd(args.jd)
    result = rt.scam_screen.assess(text) if rt.scam_screen else rules_only(text)
    print(json.dumps(result, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace, rt: Runtime) -> int:
    """Batch-evaluate the agents on real postings (calls the API; never finalises anything)."""
    df = load_emscad(args.data)
    parts = [select_postings(df, _keywords(args.keywords), args.n, args.seed, fraudulent=False)]
    if args.n_fraud:
        parts.append(select_postings(df, None, args.n_fraud, args.seed, fraudulent=True))
    postings = pd.concat(parts)
    cv = load_cv(args.cv)
    eval_id = f"eval-{datetime.now():%Y%m%d-%H%M%S}"
    out = rt.settings.data_dir / "eval" / eval_id
    out.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating {len(postings)} real postings with {rt.settings.model} "
          f"(about 4-6 model calls each). Nothing will be finalised or submitted.")
    results = evaluate_postings(postings, cv, rt.new_llm(), rt.usage_log, eval_id, out,
                                rt.settings.max_auto_revisions, rt.scam_screen)
    results.to_csv(out / "results.csv", index=False)
    summary = summarise_eval(results)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary + f"\nWrote {out / 'results.csv'} and {out / 'summary.md'}")
    return 0


def cmd_usage(args: argparse.Namespace, rt: Runtime) -> int:
    """Print token usage."""
    print(format_table(summarise(rt.usage_log.records(args.run))))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Argument parser."""
    p = argparse.ArgumentParser(prog="autoapply", description=__doc__.split("\n\n")[0],
                                epilog="autoapply never submits applications.")
    p.add_argument("--env-file", default=".env", help="dotenv file (default .env)")
    p.add_argument("-v", "--verbose", action="store_true", help="log every model call")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="parse a JD, match your CV, draft, review, then ask you")
    r.add_argument("--jd", required=True, help="job description file, or '-' to paste via stdin (URLs refused)")
    r.add_argument("--cv", required=True, help="your CV (.md, .txt or .docx)")
    r.add_argument("--no-interactive", action="store_true", help="pause at the checkpoint instead of prompting")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("resume", help="approve, revise or reject a paused run")
    s.add_argument("run_id")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--approve", action="store_true")
    g.add_argument("--reject", action="store_true")
    g.add_argument("--revise", metavar="FEEDBACK", help="ask the Writer to revise with this feedback")
    s.add_argument("--no-interactive", action="store_true", help="pause again at the next checkpoint")
    s.set_defaults(func=cmd_resume)

    d = sub.add_parser("data", help="work with the real EMSCAD job-postings dataset")
    dsub = d.add_subparsers(dest="data_command", required=True)
    ds = dsub.add_parser("sample", help="export real postings as JD files")
    ds.add_argument("--data", default="data/raw", help="fake_job_postings.csv, its zip, or the folder holding it")
    ds.add_argument("--keywords", help="comma-separated title/function keywords, e.g. 'logistics,supply chain'")
    ds.add_argument("--n", type=int, default=5)
    ds.add_argument("--seed", type=int, default=42)
    ds.add_argument("--label", choices=["legit", "fraud", "any"], default="legit")
    ds.add_argument("--out", default="data/real_jds")
    ds.set_defaults(func=cmd_data_sample)

    sc = sub.add_parser("scam", help="job-scam pre-screen (no API calls)")
    scsub = sc.add_subparsers(dest="scam_command", required=True)
    st = scsub.add_parser("train", help="train + evaluate on labelled EMSCAD postings")
    st.add_argument("--data", default="data/raw")
    st.add_argument("--seed", type=int, default=42)
    st.set_defaults(func=cmd_scam_train)
    sk = scsub.add_parser("check", help="screen one job description")
    sk.add_argument("--jd", required=True)
    sk.set_defaults(func=cmd_scam_check)

    e = sub.add_parser("eval", help="batch-evaluate the agents on real postings (uses the API)")
    e.add_argument("--data", default="data/raw")
    e.add_argument("--cv", required=True)
    e.add_argument("--keywords", help="comma-separated keywords for the legitimate postings")
    e.add_argument("--n", type=int, default=10, help="number of legitimate postings")
    e.add_argument("--n-fraud", type=int, default=0, help="also include this many labelled scam postings")
    e.add_argument("--seed", type=int, default=42)
    e.set_defaults(func=cmd_eval)

    u = sub.add_parser("usage", help="token usage summary")
    u.add_argument("--run", help="only this run id")
    u.set_defaults(func=cmd_usage)
    return p


def main(argv: list[str] | None = None, llm_factory: LLMFactory = _default_llm) -> int:
    """CLI entry point. ``llm_factory`` is injectable for tests."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    try:
        settings = load_settings(args.env_file)
        rt = Runtime(settings, llm_factory)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    try:
        return args.func(args, rt)
    except (InputError, LLMError, DatasetError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        rt.close()


if __name__ == "__main__":
    raise SystemExit(main())
