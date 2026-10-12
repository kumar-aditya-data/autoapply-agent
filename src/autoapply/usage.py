"""Token-usage logging: one JSON line per model call, plus per-run summaries."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class UsageRecord:
    """Token usage for a single model call."""

    run_id: str
    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    latency_s: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


class UsageLog:
    """Append-only JSONL usage log."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: UsageRecord) -> None:
        """Write one record."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record)) + "\n")

    def records(self, run_id: str | None = None) -> list[UsageRecord]:
        """All records, optionally filtered to one run."""
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = UsageRecord(**json.loads(line))
                if run_id is None or rec.run_id == run_id:
                    out.append(rec)
        return out


TOKEN_FIELDS = ["input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens"]


def summarise(records: list[UsageRecord]) -> list[dict[str, object]]:
    """Per-agent totals (in call order) plus a TOTAL row."""
    rows: dict[str, dict[str, object]] = {}
    for r in records:
        row = rows.setdefault(r.agent, {"agent": r.agent, "calls": 0, **{f: 0 for f in TOKEN_FIELDS}, "latency_s": 0.0})
        row["calls"] = int(row["calls"]) + 1
        for f in TOKEN_FIELDS:
            row[f] = int(row[f]) + int(getattr(r, f))
        row["latency_s"] = round(float(row["latency_s"]) + r.latency_s, 1)
    out = list(rows.values())
    if out:
        total: dict[str, object] = {"agent": "TOTAL", "calls": sum(int(r["calls"]) for r in out)}
        for f in TOKEN_FIELDS:
            total[f] = sum(int(r[f]) for r in out)
        total["latency_s"] = round(sum(float(r["latency_s"]) for r in out), 1)
        out.append(total)
    return out


def format_table(rows: list[dict[str, object]]) -> str:
    """Render summary rows as a Markdown table."""
    if not rows:
        return "(no usage recorded)"
    headers = ["agent", "calls", "input", "cache read", "cache write", "output", "latency (s)"]
    keys = ["agent", "calls", *TOKEN_FIELDS, "latency_s"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"]
    for r in rows:
        cells = [f"{r[k]:,}" if isinstance(r[k], int) else str(r[k]) for k in keys]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
