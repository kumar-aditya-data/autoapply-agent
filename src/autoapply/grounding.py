"""Deterministic grounding checks that run before the LLM Reviewer.

They catch the cheap, mechanical failures for free and cannot be talked out of
their verdict: citations to lines that don't exist, numbers that don't appear
in the cited lines, and job requirements echoed into the draft that the CV
never mentions. They are heuristics, so the LLM Reviewer still runs after them.
"""

from __future__ import annotations

import re

from autoapply.documents import CV
from autoapply.models import Draft, ParsedJD, ReviewFlag

NUMBER = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(%|k|m|\+)?", re.IGNORECASE)
STOPWORDS = {
    "and", "or", "the", "a", "an", "of", "in", "with", "to", "for", "on", "at", "years", "year",
    "experience", "knowledge", "skills", "strong", "good", "excellent", "ability", "plus",
    "working", "proficiency", "proficient", "understanding", "degree", "e.g", "etc", "including",
    # generic English that says nothing about a qualification
    "from", "have", "will", "your", "their", "they", "them", "that", "this", "these", "those", "when",
    "where", "which", "while", "with", "into", "about", "able", "also", "both", "each", "more", "most",
    "must", "such", "than", "very", "well", "work", "willingness", "relevant", "required", "requirements",
    "preferred", "advantage", "start", "finish", "complete", "clear", "desired", "minimum", "least",
}


def _numbers(text: str) -> set[str]:
    return {m.group(1).replace(",", "") for m in NUMBER.finditer(text)}


def _key_terms(requirement: str) -> list[str]:
    """Distinctive tokens from a requirement (tools, acronyms, multi-word names kept simple)."""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#./-]{1,}", requirement)
    return [t for t in tokens if t.lower() not in STOPWORDS and (len(t) > 3 or t.isupper())]


def check_citations(draft: Draft, cv: CV) -> list[ReviewFlag]:
    """Every claim and bullet must cite existing CV line IDs."""
    flags = []
    for claim in draft.claims:
        bad = [c for c in claim.cv_lines if c not in cv.lines]
        if not claim.cv_lines or bad:
            flags.append(ReviewFlag(
                text=claim.sentence,
                issue="no CV line cited" if not claim.cv_lines else f"cites non-existent CV lines: {bad}",
                severity="high", cv_lines_checked=claim.cv_lines, source="rule"))
    for b in draft.bullet_suggestions:
        if b.cv_line not in cv.lines:
            flags.append(ReviewFlag(text=b.suggested, issue=f"bullet rewrites non-existent CV line {b.cv_line}",
                                    severity="high", cv_lines_checked=[b.cv_line], source="rule"))
    return flags


def check_numbers(draft: Draft, cv: CV) -> list[ReviewFlag]:
    """Numbers in claims/bullets must appear in the lines they cite."""
    flags = []
    pairs = [(c.sentence, c.cv_lines) for c in draft.claims]
    pairs += [(b.suggested, [b.cv_line]) for b in draft.bullet_suggestions]
    for text, lines in pairs:
        cited = " ".join(cv.lines.get(c, "") for c in lines)
        missing = sorted(_numbers(text) - _numbers(cited))
        if missing:
            flags.append(ReviewFlag(text=text, issue=f"number(s) {missing} not found in cited CV lines {lines}",
                                    severity="high", cv_lines_checked=lines, source="rule"))
    return flags


def check_unsupported_requirements(draft: Draft, cv: CV, jd: ParsedJD) -> list[ReviewFlag]:
    """Flag JD requirement terms that appear in the draft but nowhere in the CV."""
    cv_text = cv.text().lower()
    draft_text = (draft.cover_letter + " " + " ".join(b.suggested for b in draft.bullet_suggestions)).lower()
    acknowledged = " ".join(draft.gaps_acknowledged).lower()
    # Naming the role or employer ("...the Digital Procurement Manager role at X") is not a claim.
    naming = f"{jd.title} {jd.company}".lower()
    flags = []
    seen: set[str] = set()
    for req in jd.must_haves + jd.nice_to_haves:
        for term in _key_terms(req):
            t = term.lower()
            if t in seen or t in cv_text or t in acknowledged or t in naming:
                continue
            if re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", draft_text):
                seen.add(t)
                flags.append(ReviewFlag(
                    text=term, issue=f"draft mentions '{term}' (from requirement '{req}') but the CV never does",
                    severity="medium", source="rule"))
    return flags


ORG_NAME = re.compile(
    r"\b((?:[A-Z][\w&.\-]*\s){0,3}(?:GmbH|AG|SE|Ltd\.?|Limited|Inc\.?|LLC|B\.V\.|Pvt\.?|PLC))(?![\w])"
)


def check_company_names(draft: Draft, cv: CV, jd: ParsedJD) -> list[ReviewFlag]:
    """Company-suffixed names in the letter must be the target company or appear in the CV."""
    known = (cv.text() + " " + jd.company).lower()
    flags = []
    for match in {m.group(1).strip() for m in ORG_NAME.finditer(draft.cover_letter)}:
        *words, suffix = match.split()
        # The regex may also capture a capitalised word before the name ("Before Parcelio GmbH"),
        # so the name is known if any trailing run of words appears in the CV or is the target.
        tails = [" ".join(words[i:]).lower() for i in range(len(words))]
        if tails and not any(t in known for t in tails):
            flags.append(ReviewFlag(text=f"{words[-1]} {suffix}",
                                    issue="company name not in the CV and not the target employer",
                                    severity="high", source="rule"))
    return flags


def run_rule_checks(draft: Draft, cv: CV, jd: ParsedJD) -> list[ReviewFlag]:
    """All deterministic checks."""
    return (check_citations(draft, cv) + check_numbers(draft, cv)
            + check_unsupported_requirements(draft, cv, jd) + check_company_names(draft, cv, jd))
