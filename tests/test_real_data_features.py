"""Dataset loader, scam pre-screen, batch evaluation and new CLI commands.

Uses a small SYNTHETIC file shaped like EMSCAD so tests never need the real download.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from autoapply.cli import main
from autoapply.datasets import DatasetError, build_jd_text, export_postings, load_emscad, select_postings
from autoapply.documents import load_cv
from autoapply.evaluate import evaluate_postings, summarise_eval
from autoapply.graph import build_graph
from autoapply.scamcheck import ScamScreen, red_flags, rules_only, train
from autoapply.usage import UsageLog
from tests.conftest import SAMPLE_CV, FakeLLM

LEGIT = ["Logistics Analyst", "Supply Chain Planner", "Warehouse Supervisor", "Data Analyst", "Transport Coordinator"]


def synthetic_emscad(n_legit: int = 160, n_fraud: int = 40, seed: int = 0) -> pd.DataFrame:
    """SYNTHETIC rows in the EMSCAD schema; scams use typical scam vocabulary."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_legit):
        title = LEGIT[i % len(LEGIT)]
        rows.append({"job_id": i, "title": title, "location": "US, NY, New York", "employment_type": "Full-time",
                     "company_profile": f"Acme Logistics {i} operates regional distribution centres.",
                     "description": f"We are hiring a {title} to analyse throughput, maintain dashboards in SQL "
                                    f"and Power BI, and support planning across {rng.integers(2, 9)} sites. " * 4,
                     "requirements": "Degree in supply chain or engineering; 2+ years experience; SQL; Excel.",
                     "benefits": "Pension, 25 days holiday.", "function": "Supply Chain", "fraudulent": 0})
    for i in range(n_fraud):
        rows.append({"job_id": 10_000 + i, "title": "Data Entry Clerk - Work From Home", "location": "US",
                     "employment_type": "Part-time", "company_profile": "",
                     "description": f"Earn $500 per day from home, no experience needed working from home. "
                                    f"Pay a small fee for your starter kit and send your bank account details. "
                                    f"Contact hr{i}@gmail.com on WhatsApp today. " * 3,
                     "requirements": "", "benefits": "Unlimited income", "function": "Administrative", "fraudulent": 1})
    return pd.DataFrame(rows)


@pytest.fixture
def emscad_csv(tmp_path):
    df = synthetic_emscad()
    df = pd.concat([df, df.iloc[[0]]])  # one exact duplicate to be removed
    path = tmp_path / "fake_job_postings.csv"
    df.to_csv(path, index=False)
    return path


# --- dataset ----------------------------------------------------------------------

def test_load_dedupes_and_builds_text(emscad_csv) -> None:
    df = load_emscad(emscad_csv.parent)  # folder path works too
    assert len(df) == 200 and df.attrs["duplicates_removed"] == 1
    text = df.loc[0, "jd_text"]
    assert text.startswith("Logistics Analyst") and "About the company" in text and "Requirements" in text


def test_jd_text_skips_empty_sections() -> None:
    text = build_jd_text(pd.Series({"title": "X", "company_profile": float("nan"), "description": "Do things",
                                    "requirements": "", "benefits": None}))
    assert text == "X\n\nDescription\nDo things"


def test_load_errors(tmp_path) -> None:
    with pytest.raises(DatasetError, match="kaggle datasets download"):
        load_emscad(tmp_path / "missing.csv")
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"title": ["x"]}).to_csv(bad, index=False)
    with pytest.raises(DatasetError, match="missing columns"):
        load_emscad(bad)


def test_select_and_export(emscad_csv, tmp_path) -> None:
    df = load_emscad(emscad_csv)
    sample = select_postings(df, ["supply chain", "logistics"], n=3, min_chars=100)
    assert len(sample) == 3 and (sample["fraudulent"] == 0).all()
    paths = export_postings(sample, tmp_path / "jds")
    head = paths[0].read_text(encoding="utf-8")
    assert "EMSCAD" in head and "CC0" in head and "label: legitimate" in head
    with pytest.raises(DatasetError):
        select_postings(df, ["astronaut"], min_chars=100)
    # the attribution/label header must not reach the models
    from autoapply.documents import load_jd
    loaded = load_jd(str(paths[0]))
    assert "EMSCAD" not in loaded and "label:" not in loaded and loaded.startswith(sample.iloc[0]["title"])


# --- scam screen --------------------------------------------------------------------

def test_red_flags_detect_typical_scam_text() -> None:
    flags = {f["flag"] for f in red_flags(
        "Pay the $99 fee for your training materials. Send your bank account details. "
        "Message us on WhatsApp or email recruiter@gmail.com. Earn $400 per day!")}
    assert {"upfront_payment", "financial_details", "off_platform_chat", "personal_email", "unrealistic_pay"} <= flags
    assert red_flags("Supply chain analyst, SQL and Power BI, hybrid in Hamburg.") == []


def test_rules_only_assessment() -> None:
    assert rules_only("Normal posting") == {"probability": None, "risk": "unknown", "top_terms": [],
                                            "red_flags": [], "model": False}
    tg = rules_only("Contact us on Telegram")
    assert tg["risk"] == "unknown" and tg["red_flags"][0]["flag"] == "off_platform_chat"  # rules never set risk


def test_train_save_load_and_assess(emscad_csv, tmp_path) -> None:
    screen, test = train(load_emscad(emscad_csv), seed=1)
    m = screen.metrics
    assert {"test_high_pr_auc", "test_high_precision", "rules_recall", "threshold_high"} <= set(m)
    assert 0 <= screen.threshold_medium <= screen.threshold_high <= 1
    assert len(test) == m["n_test"]
    path = screen.save(tmp_path / "m.joblib")
    loaded = ScamScreen.load(path)
    scam = loaded.assess("Earn $500 per day from home, pay a fee for your starter kit, bank account details")
    legit = loaded.assess("Supply chain planner, SQL and Power BI dashboards across distribution centres")
    assert scam["probability"] > legit["probability"]
    assert scam["risk"] in {"medium", "high"}
    assert scam["top_terms"] and set(scam["top_terms"]) <= set(
        "earn 500 per day from home pay fee for your starter kit bank account details".split()
        + ["earn 500", "500 per", "per day", "day from", "from home", "pay fee", "starter kit", "bank account",
           "account details", "fee for", "for your", "your starter", "kit bank", "home pay", "pay a"])


# --- graph + evaluation -------------------------------------------------------------

def test_screening_reaches_checkpoint_without_model(tmp_path, initial_state) -> None:
    state = {**initial_state, "jd_text": initial_state["jd_text"] + "\nEmail jobs@gmail.com to apply."}
    graph = build_graph(FakeLLM(), tmp_path, InMemorySaver())
    result = graph.invoke(state, {"configurable": {"thread_id": "t"}})
    screening = result["__interrupt__"][0].value["screening"]
    assert screening["model"] is False and screening["red_flags"][0]["flag"] == "personal_email"


def test_evaluate_postings_never_finalises(emscad_csv, tmp_path) -> None:
    df = load_emscad(emscad_csv)
    postings = pd.concat([df[df.fraudulent == 0].head(2), df[df.fraudulent == 1].head(1)])
    log = UsageLog(tmp_path / "u.jsonl")
    results = evaluate_postings(postings, load_cv(SAMPLE_CV), FakeLLM(usage_log=log), log, "ev", tmp_path)
    assert (results["status"] == "ok").all() and len(results) == 3
    assert results.loc[2, "scam_red_flags"] > 0 and results.loc[0, "scam_red_flags"] == 0
    assert (results["calls"] == 4).all()
    assert not (tmp_path / "never_written").exists()  # rejected: no outputs
    summary = summarise_eval(results)
    assert "Labelled scams" in summary and "Mean match score | 72.0" in summary


# --- CLI ----------------------------------------------------------------------------

@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOAPPLY_MODEL", "fake-model")
    monkeypatch.setenv("AUTOAPPLY_DATA_DIR", str(tmp_path / "state"))
    return tmp_path / "state"


def _factory(settings, usage_log):
    return FakeLLM(usage_log=usage_log)


def test_cli_data_scam_and_eval(env, emscad_csv, tmp_path, capsys) -> None:
    base = ["--env-file", "none.env"]
    assert main(base + ["data", "sample", "--data", str(emscad_csv), "--keywords", "planner", "--n", "2",
                        "--out", str(tmp_path / "jds")], llm_factory=_factory) == 0
    jd = next((tmp_path / "jds").glob("*.md"))
    capsys.readouterr()  # discard the exported file list
    assert main(base + ["scam", "check", "--jd", str(jd)], llm_factory=_factory) == 0
    assert json.loads(capsys.readouterr().out)["model"] is False  # no trained model yet: rules only
    assert main(base + ["scam", "train", "--data", str(emscad_csv)], llm_factory=_factory) == 0
    assert (env / "models" / "scam_screen.joblib").exists()
    assert "PR-AUC" in capsys.readouterr().out
    assert main(base + ["scam", "check", "--jd", str(jd)], llm_factory=_factory) == 0
    assert json.loads(capsys.readouterr().out)["model"] is True  # trained model now used
    assert main(base + ["eval", "--data", str(emscad_csv), "--cv", str(SAMPLE_CV), "--n", "2", "--n-fraud", "1"],
                llm_factory=_factory) == 0
    assert "Evaluation on real job postings" in capsys.readouterr().out
    assert list((env / "eval").glob("*/results.csv"))
