"""CLI end to end with a fake LLM: pause in one process, resume in another."""

from __future__ import annotations

import pytest

from autoapply.cli import main
from autoapply.usage import UsageLog
from tests.conftest import SAMPLE_CV, SAMPLE_JD, FakeLLM


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOAPPLY_MODEL", "fake-model")
    monkeypatch.setenv("AUTOAPPLY_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


def _factory(settings, usage_log: UsageLog):
    return FakeLLM(usage_log=usage_log)


def _run_id(output: str) -> str:
    line = next(l for l in output.splitlines() if "autoapply resume" in l)
    return line.split()[2]


def test_run_pause_then_resume_approve(env, capsys) -> None:
    args = ["--env-file", "none.env", "run", "--jd", str(SAMPLE_JD), "--cv", str(SAMPLE_CV), "--no-interactive"]
    assert main(args, llm_factory=_factory) == 0
    out = capsys.readouterr().out
    assert "HUMAN REVIEW CHECKPOINT" in out and "Paused for human review" in out
    run_id = _run_id(out)

    # a separate invocation (new process in real life) resumes from the SQLite checkpoint
    assert main(["--env-file", "none.env", "resume", run_id, "--approve"], llm_factory=_factory) == 0
    out = capsys.readouterr().out
    assert "Nothing has been submitted" in out
    assert "| TOTAL | 4 |" in out  # parser, matcher, writer, reviewer
    assert (env / "outputs" / run_id / "usage.md").exists()

    assert main(["--env-file", "none.env", "resume", run_id, "--approve"], llm_factory=_factory) == 0
    assert "already finished" in capsys.readouterr().out


def test_interactive_reject(env, monkeypatch, capsys) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt="": "x")
    args = ["--env-file", "none.env", "run", "--jd", str(SAMPLE_JD), "--cv", str(SAMPLE_CV)]
    assert main(args, llm_factory=_factory) == 0
    assert "Rejected. No final files" in capsys.readouterr().out
    assert not (env / "outputs").exists()


def test_url_jd_rejected(env, capsys) -> None:
    args = ["--env-file", "none.env", "run", "--jd", "https://example.com/job", "--cv", str(SAMPLE_CV)]
    assert main(args, llm_factory=_factory) == 2
    assert "never fetches" in capsys.readouterr().err


def test_missing_model_config(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.delenv("AUTOAPPLY_MODEL", raising=False)
    assert main(["--env-file", str(tmp_path / "none.env"), "usage"], llm_factory=_factory) == 2
    assert "AUTOAPPLY_MODEL" in capsys.readouterr().err


def test_unknown_run_id(env, capsys) -> None:
    assert main(["--env-file", "none.env", "resume", "nope", "--approve"], llm_factory=_factory) == 2
