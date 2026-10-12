"""Job-scam pre-screen: runs before any model call, so it costs nothing and needs no API key.

Two layers, with different standing:

1. **A text classifier** (TF-IDF + logistic regression) trained on the real,
   labelled EMSCAD postings and evaluated on a held-out split. It alone sets the
   risk band, and it explains itself by listing the terms that pushed the score up.
2. **Red-flag rules** for common modern scam signs (fees, bank details, chat-app
   contact, free email addresses, unrealistic pay). On EMSCAD they had very low
   precision (they fire mostly on legitimate postings, and EMSCAD masks emails and
   phone numbers), so they are shown as an *unvalidated checklist* and never
   change the risk band.

The classifier is trained on posting *text only*, because that is all a user
pastes into this tool (no logo or metadata fields).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

RED_FLAGS: list[tuple[str, str, re.Pattern[str]]] = [
    ("upfront_payment", "asks the applicant to pay (training, kit, registration, background check)",
     re.compile(r"\b(pay|payment|fee|deposit|purchase)\b[^.]{0,60}\b(training|starter kit|registration|equipment|"
                r"background check|software|materials)\b", re.I)),
    ("financial_details", "asks for bank, payment or identity details",
     re.compile(r"\b(bank account|routing number|social security|ssn|wire transfer|money order|"
                r"cash(ing)? (a )?checks?|western union|paypal account)\b", re.I)),
    ("off_platform_chat", "moves contact to chat apps",
     re.compile(r"\b(whats ?app|telegram|google hangouts|yahoo messenger|skype id|text (us|me) at)\b", re.I)),
    ("personal_email", "contact via a free personal email address",
     re.compile(r"[\w.+-]+@(gmail|yahoo|hotmail|outlook|aol|live)\.[a-z.]+", re.I)),
    ("unrealistic_pay", "unrealistic or guaranteed earnings",
     re.compile(r"(earn|make|up to)\s*\$?\s?\d[\d,]{2,}\s*(\+|or more)?\s*(per|a|/)\s*(day|week)|"
                r"guaranteed (income|pay|earnings)|unlimited (income|earning)", re.I)),
    ("no_experience_wfh", "no experience needed + work from home + quick money",
     re.compile(r"no (prior )?experience (is )?(needed|required|necessary)[^.]{0,120}"
                r"(work from home|from home|home[- ]based)|"
                r"(work from home|home[- ]based)[^.]{0,120}no (prior )?experience (is )?(needed|required|necessary)", re.I)),
]


def red_flags(text: str) -> list[dict[str, str]]:
    """Return the rule-based red flags found in ``text`` with the matching snippet."""
    found = []
    for flag_id, why, pattern in RED_FLAGS:
        m = pattern.search(text)
        if m:
            found.append({"flag": flag_id, "why": why, "evidence": m.group(0)[:120]})
    return found


@dataclass
class ScamScreen:
    """Trained classifier plus the decision thresholds chosen on validation data."""

    pipeline: Pipeline
    threshold_medium: float
    threshold_high: float
    metrics: dict[str, float] = field(default_factory=dict)

    def probability(self, text: str) -> float:
        """Estimated probability that the posting is fraudulent."""
        return float(self.pipeline.predict_proba([text])[0, 1])

    def top_terms(self, text: str, k: int = 5) -> list[str]:
        """Terms in ``text`` that pushed the scam score up the most (tf-idf weight x coefficient)."""
        vec, clf = self.pipeline.named_steps["tfidf"], self.pipeline.named_steps["clf"]
        x = vec.transform([text]).tocoo()
        contrib = x.data * clf.coef_[0][x.col]
        order = np.argsort(contrib)[::-1]
        names = vec.get_feature_names_out()
        return [names[x.col[i]] for i in order[:k] if contrib[i] > 0]

    def assess(self, text: str) -> dict[str, object]:
        """Classifier probability and risk band, its top terms, and the unvalidated rule checklist."""
        p = self.probability(text)
        band = "high" if p >= self.threshold_high else "medium" if p >= self.threshold_medium else "low"
        return {"probability": round(p, 3), "risk": band, "top_terms": self.top_terms(text),
                "red_flags": red_flags(text), "model": True}

    def save(self, path: str | Path) -> Path:
        """Persist with joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "ScamScreen":
        """Load a model saved with :meth:`save` (only load files you trained yourself)."""
        return joblib.load(path)


def rules_only(text: str) -> dict[str, object]:
    """Assessment when no trained model is available: no risk band, only the checklist."""
    return {"probability": None, "risk": "unknown", "top_terms": [], "red_flags": red_flags(text), "model": False}


def make_pipeline() -> Pipeline:
    """TF-IDF (words + bigrams) and class-balanced logistic regression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=3, max_features=60_000,
                                  strip_accents="unicode")),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=3000, C=4.0)),
    ])


def _threshold_for_precision(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Smallest probability threshold whose precision on (y, p) reaches ``target``."""
    for t in np.unique(p):
        pred = p >= t
        if pred.sum() and y[pred].mean() >= target:
            return float(t)
    return 0.99


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float, prefix: str) -> dict[str, float]:
    prec, rec, f1, _ = precision_recall_fscore_support(y, p >= threshold, average="binary", zero_division=0)
    return {f"{prefix}_pr_auc": float(average_precision_score(y, p)), f"{prefix}_roc_auc": float(roc_auc_score(y, p)),
            f"{prefix}_precision": float(prec), f"{prefix}_recall": float(rec), f"{prefix}_f1": float(f1)}


def train(df: pd.DataFrame, seed: int = 42, target_precision_high: float = 0.8,
          target_precision_medium: float = 0.4) -> tuple[ScamScreen, pd.DataFrame]:
    """Train on 60%, choose thresholds on 20% validation, report on a 20% test split.

    Returns the screen and the test-set predictions (for error analysis).
    The split is stratified and random: EMSCAD has no posting dates, so a
    time-based split is impossible (a limitation stated in the README).
    """
    text_col = "model_text" if "model_text" in df.columns else "jd_text"
    X, y = df[text_col].to_numpy(), df["fraudulent"].to_numpy()
    X_tmp, X_test, y_tmp, y_test, idx_tmp, idx_test = train_test_split(
        X, y, np.arange(len(df)), test_size=0.2, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=seed)

    pipe = make_pipeline().fit(X_train, y_train)
    p_val = pipe.predict_proba(X_val)[:, 1]
    t_high = _threshold_for_precision(y_val, p_val, target_precision_high)
    t_med = min(_threshold_for_precision(y_val, p_val, target_precision_medium), t_high)

    p_test = pipe.predict_proba(X_test)[:, 1]
    rule_hit = np.array([bool(red_flags(t)) for t in X_test])
    rp, rr, rf, _ = precision_recall_fscore_support(y_test, rule_hit, average="binary", zero_division=0)
    metrics = {
        "n_train": int(len(y_train)), "n_val": int(len(y_val)), "n_test": int(len(y_test)),
        "fraud_rate_test": float(y_test.mean()),
        "threshold_medium": t_med, "threshold_high": t_high,
        **_metrics(y_test, p_test, t_high, "test_high"),
        **{k: v for k, v in _metrics(y_test, p_test, t_med, "test_medium").items() if "auc" not in k},
        "rules_precision": float(rp), "rules_recall": float(rr), "rules_f1": float(rf),
        "rules_flag_rate": float(rule_hit.mean()),
    }
    test_frame = df.iloc[idx_test][["title", "fraudulent"]].assign(probability=p_test, rule_flagged=rule_hit)
    return ScamScreen(pipe, t_med, t_high, metrics), test_frame
