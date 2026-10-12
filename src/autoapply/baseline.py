"""Offline baseline: a no-LLM stand-in for the four agents.

Enable with ``AUTOAPPLY_MODEL=baseline``. Nothing here calls an API; it does
simple, transparent processing so the whole graph (scam screen, grounding
checks, human checkpoint, outputs) runs for free, and so Claude's agents
have a baseline to be compared against:

- Parser:   section headings, bullet/sentence splitting and keyword cues.
- Matcher:  keyword overlap (crude stemming) between each requirement and the numbered CV lines.
- Writer:   a template letter that quotes matching CV lines *verbatim*, so it
            cannot fabricate, but it also cannot adapt tone or act on feedback.
- Reviewer: no LLM fact-check; only the deterministic rule checks (run by the graph).

Outputs are labelled as baseline wherever they appear.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel

from autoapply.grounding import _key_terms
from autoapply.models import (BulletSuggestion, Claim, Draft, MatchReport, ParsedJD, RequirementMatch,
                              ReviewReport)
from autoapply.scamcheck import red_flags
from autoapply.usage import UsageLog, UsageRecord

T = TypeVar("T", bound=BaseModel)
BASELINE_MODEL = "offline-baseline"

# heading prefix -> canonical section name
SECTION_HEADINGS = {
    "about the company": "about", "about us": "about", "description": "description",
    "what you will do": "responsibilities", "what you'll do": "responsibilities",
    "responsibilities": "responsibilities", "your role": "responsibilities",
    "requirements": "requirements", "what you bring": "requirements", "qualifications": "requirements",
    "your profile": "requirements", "nice to have": "nice", "preferred": "nice", "bonus": "nice",
    "benefits": "benefits", "what we offer": "benefits",
}
MATCH_STOPWORDS = {
    "another", "other", "method", "methods", "that", "them", "this", "these", "those", "such", "using",
    "related", "field", "similar", "relevant", "including", "within", "across", "strong", "very", "good",
    "team", "work", "will", "your", "have", "must", "required", "preferred", "ideally", "plus", "least",
    "minimum", "e.g.", "i.e.", "etc", "able", "both", "well", "high", "level", "based", "ideally",
}
NICE_CUES = re.compile(r"\b(plus|preferred|nice to have|desirable|bonus|advantage|ideally)\b", re.I)
REQ_CUES = re.compile(r"\b(experience|degree|skills?|knowledge|ability|proficien\w*|familiar\w*|years?)\b", re.I)
LANG_CUES = re.compile(r"\b(english|german|french|spanish|dutch|italian)\b", re.I)
AI_DIRECTED = re.compile(r"(note to (ai|automated)|\bai screening\b|language model|ignore (all |previous )?"
                         r"instructions|automated (screening|system)s?)", re.I)
SENTENCE_SPLIT = re.compile(r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<=[.;])\s+(?=[A-Z])|\s+[•·]\s+")
WORD = re.compile(r"[a-z0-9][a-z0-9+#.-]*[a-z0-9+#]|[a-z0-9]")


def _json_after(label: str, text: str) -> dict:
    start = text.find("{", text.find(label))
    return json.JSONDecoder().raw_decode(text[start:])[0]


def _sections(jd: str) -> dict[str, str]:
    """Split a posting into canonical sections; text before any heading goes under 'intro'."""
    out: dict[str, list[str]] = {"intro": []}
    current = "intro"
    for line in jd.splitlines():
        key = line.strip().strip("#*: ").lower()
        heading = next((name for prefix, name in SECTION_HEADINGS.items() if key.startswith(prefix)), None)
        if heading and len(key) <= 40:  # a short line starting with a known heading
            current = heading
            out.setdefault(current, [])
        else:
            out.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in out.items()}


CAMEL_WORDS = ["PowerPoint", "PowerBI", "JavaScript", "TypeScript", "LinkedIn", "YouTube", "SharePoint",
               "QuickBooks", "SalesForce", "GitHub", "PostgreSQL", "NetSuite", "WordPress", "HubSpot", "DevOps"]


def _unglue(text: str) -> str:
    """Restore bullet breaks lost when HTML was stripped (common in scraped postings such as EMSCAD).

    "experience in a roleStrong Excel skills" -> two lines; "titles- Support" -> two lines; "a | b" -> two lines.
    Known CamelCase product names are protected.
    """
    text = text.replace("\xa0", " ")
    for i, w in enumerate(CAMEL_WORDS):
        text = text.replace(w, f"\x00{i}\x00")
    text = re.sub(r"([a-z]{2,}[.)]?)([A-Z][a-z])", r"\1\n\2", text)
    text = re.sub(r"(?<=\w)-\s+(?=[A-Z])", "\n", text)
    text = re.sub(r"(?<=[a-z]{2})\.(?=[A-Z0-9])", ".\n", text)  # "an advantage.5+ years" -> two items
    text = re.sub(r"\s*\|\s*", "\n", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: CAMEL_WORDS[int(m.group(1))], text)


def _items(text: str, limit: int = 10) -> list[str]:
    """Bullets (one per line) or sentences, de-duplicated, trimmed, prompt-injection lines dropped."""
    text = _unglue(text)
    lines = [l for l in text.splitlines() if l.strip()]
    parts = lines if len(lines) > 1 else SENTENCE_SPLIT.split(text)
    seen, out = set(), []
    for p in parts:
        p = p.strip(" -*•\t").strip()
        if len(p) >= 12 and len(p.split()) >= 2 and p.lower() not in seen and not AI_DIRECTED.search(p):
            seen.add(p.lower())
            out.append(p[:160])
    return out[:limit]


def parse_jd(jd: str) -> ParsedJD:
    """Heuristic parse of a job description."""
    lines = [l.strip() for l in jd.splitlines() if l.strip()]
    title = lines[0].strip("# ") if lines else "Unknown role"
    sec = _sections(jd)
    location = next((l.split(":", 1)[1].strip() for l in lines if l.lower().startswith("location:")), "")
    bold = re.search(r"\*\*([^*]{2,60})\*\*", jd)
    lead = re.match(r"([A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*){0,3})", sec.get("about", ""))
    company = bold.group(1) if bold else lead.group(1) if lead else "the company"
    company = re.sub(r"^(At|The|We|Our|Welcome to)\s+", "", company.strip())  # "At Hayes-Corp" -> "Hayes-Corp"
    candidates = (_items(sec.get("requirements", ""), 14)
                  or [s for s in _items(sec.get("description", ""), 30) if REQ_CUES.search(s)])
    nice = [c for c in candidates if NICE_CUES.search(c)] + _items(sec.get("nice", ""), 6)
    must = [c for c in candidates if c not in nice]
    return ParsedJD(
        title=title, company=company.strip(), location=location,
        must_haves=must[:10], nice_to_haves=nice[:6],
        responsibilities=_items(sec.get("responsibilities") or sec.get("description") or sec.get("intro", ""), 5),
        language_requirements=[c for c in candidates if LANG_CUES.search(c)][:3],
        suspicious_instructions=[l for l in lines if AI_DIRECTED.search(l)][:3],
        scam_signals=[f"{f['why']}: \"{f['evidence']}\"" for f in red_flags(jd)],
    )


def _cv_lines(numbered: str) -> dict[str, str]:
    return dict(re.findall(r"^\[(L\d+)\] (.*)$", numbered, re.M))


def _stem(word: str) -> str:
    """Crude prefix stem so 'forecasting' ~ 'forecast' and 'warehouses' ~ 'warehouse'."""
    return word[:6] if len(word) > 5 else word


def _tokens(text: str) -> set[str]:
    return {_stem(w) for w in WORD.findall(text.lower())}


def match(jd: ParsedJD, cv: dict[str, str]) -> MatchReport:
    """Keyword overlap per requirement; must-haves weigh 3x."""
    reqs, total, got = [], 0.0, 0.0
    cv_tokens = {lid: _tokens(txt) for lid, txt in cv.items()}
    for weight, items in ((3.0, jd.must_haves), (1.0, jd.nice_to_haves)):
        for req in items:
            terms = {t.lower() for t in _key_terms(req)} - MATCH_STOPWORDS
            hits = {lid: [t for t in terms if _stem(t) in toks] for lid, toks in cv_tokens.items()}
            hits = {k: v for k, v in hits.items() if v}
            matched = {t for v in hits.values() for t in v}
            share = len(matched) / len(terms) if terms else 0.0
            status = "met" if terms and share >= 0.5 else "partial" if matched else "missing"
            best = sorted(hits, key=lambda k: -len(hits[k]))[:3]
            reasoning = (f"CV mentions {', '.join(sorted(matched))} ({len(matched)}/{len(terms)} key terms)"
                         if matched else f"No CV line mentions {', '.join(sorted(terms)) or 'its key terms'}")
            reqs.append(RequirementMatch(requirement=req, status=status, cv_lines=best, reasoning=reasoning))
            total += weight
            got += weight * {"met": 1.0, "partial": 0.5, "missing": 0.0}[status]
    if not reqs:
        return MatchReport(score=0, recommendation="skip", requirements=[],
                           summary="Baseline could not extract any requirements from this posting; "
                                   "the score is not meaningful.")
    score = round(100 * got / total)
    rec = "apply" if score >= 70 else "stretch" if score >= 45 else "skip"
    met = [r.requirement for r in reqs if r.status == "met"]
    gaps = [r.requirement for r in reqs if r.status == "missing" and r.requirement in jd.must_haves]
    return MatchReport(score=score, recommendation=rec, requirements=reqs, strengths=met[:5], gaps=gaps[:5],
                       summary=f"Baseline keyword overlap: {len(met)} of {len(reqs)} requirements fully matched.")


def write(jd: ParsedJD, report: MatchReport, cv: dict[str, str]) -> Draft:
    """Template letter quoting CV lines verbatim: grounded by construction, but rigid."""
    evidence: list[str] = []
    for r in sorted(report.requirements, key=lambda r: r.status != "met"):
        for lid in r.cv_lines:
            # skip headings such as "Professional Experience": evidence must be a real statement
            if lid not in evidence and r.status != "missing" and lid in cv and len(cv[lid].split()) >= 6:
                evidence.append(lid)
    evidence = evidence[:3]
    claims = [Claim(sentence=f"My CV notes: \"{cv[lid]}\".", cv_lines=[lid]) for lid in evidence]
    body = " ".join(c.sentence for c in claims) or "My CV does not closely match the listed requirements."
    letter = (f"Dear Hiring Manager,\n\nI would like to be considered for the {jd.title} role at {jd.company}. "
              f"Relevant to your requirements: {body}\n\nI would welcome the chance to discuss the role.\n\n"
              "[Baseline draft generated without an LLM: rewrite before sending.]")
    bullets = [BulletSuggestion(cv_line=lid, original=cv[lid], suggested=cv[lid],
                                targets_requirement=next(r.requirement for r in report.requirements if lid in r.cv_lines))
               for lid in evidence]
    return Draft(cover_letter=letter, bullet_suggestions=bullets, claims=claims, gaps_acknowledged=report.gaps)


class BaselineClient:
    """``LLMClient`` that runs the heuristics above instead of a model."""

    def __init__(self, usage_log: UsageLog | None = None) -> None:
        self.usage_log = usage_log

    def structured(self, *, run_id: str, agent: str, output_model: type[T], system: str,
                   cached_context: str | None, user: str) -> T:
        """Dispatch on the requested output model; logs a zero-token usage record."""
        if self.usage_log is not None:
            self.usage_log.append(UsageRecord(run_id=run_id, agent=agent, model=BASELINE_MODEL,
                                              input_tokens=0, output_tokens=0))
        cv = _cv_lines(cached_context or "")
        if output_model is ParsedJD:
            jd = re.search(r"<job_description>\n(.*)\n</job_description>", user, re.S)
            return parse_jd(jd.group(1) if jd else user)  # type: ignore[return-value]
        if output_model is MatchReport:
            return match(ParsedJD.model_validate(_json_after("Parsed job requirements", user)), cv)  # type: ignore[return-value]
        if output_model is Draft:
            jd = ParsedJD.model_validate(_json_after("Parsed job requirements", user))
            report = MatchReport.model_validate(_json_after("Fit analysis", user))
            return write(jd, report, cv)  # type: ignore[return-value]
        if output_model is ReviewReport:
            return ReviewReport(verdict="pass", flags=[],  # type: ignore[return-value]
                                notes="Baseline reviewer: deterministic rule checks only, no LLM fact-check.")
        raise TypeError(f"baseline cannot produce {output_model.__name__}")
