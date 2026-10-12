"""Real job postings from the EMSCAD dataset (Kaggle: shivamb/real-or-fake-fake-jobposting-prediction).

Source: Employment Scam Aegean Dataset (EMSCAD), Laboratory of Information &
Communication Systems Security, University of the Aegean. About 18k real job
postings (2012-2014) with a ``fraudulent`` label. Licence on Kaggle: CC0 Public Domain.
Paper: Vidros, S., Kolias, C., Kambourakis, G. & Akoglu, L. (2017). Automatic Detection
of Online Recruitment Frauds: Characteristics, Methods, and a Public Dataset.
Future Internet, 9(1), 6.

The dataset is downloaded by the user (Kaggle requires a login); it is never committed.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd

KAGGLE_DATASET = "shivamb/real-or-fake-fake-jobposting-prediction"
CSV_NAME = "fake_job_postings.csv"
REQUIRED = ["title", "company_profile", "description", "requirements", "benefits", "fraudulent"]
TEXT_SECTIONS = [("location", "Location"), ("employment_type", "Employment type"),
                 ("company_profile", "About the company"), ("description", "Description"),
                 ("requirements", "Requirements"), ("benefits", "Benefits")]
SOURCE_NOTE = ("Real job posting from EMSCAD (University of the Aegean), via Kaggle "
               f"'{KAGGLE_DATASET}', CC0 Public Domain.")


class DatasetError(ValueError):
    """Raised when the dataset file is missing or has the wrong shape."""


def build_jd_text(row: pd.Series) -> str:
    """Rebuild a posting as the plain text a job seeker would copy from a job board."""
    parts = [str(row.get("title", "")).strip()]
    for col, heading in TEXT_SECTIONS:
        value = row.get(col)
        if isinstance(value, str) and value.strip():
            parts.append(f"{heading}: {value.strip()}" if col in {"location", "employment_type"}
                         else f"\n{heading}\n{value.strip()}")
    return "\n".join(parts).strip()


def load_emscad(path: str | Path) -> pd.DataFrame:
    """Load ``fake_job_postings.csv`` (or the Kaggle zip containing it).

    Adds ``jd_text`` and drops exact duplicate postings, which would otherwise
    leak between train and test splits.
    """
    path = Path(path)
    if path.is_dir():
        path = path / CSV_NAME
    if not path.exists():
        raise DatasetError(f"dataset not found at {path}. Download it with:\n"
                           f"  kaggle datasets download {KAGGLE_DATASET} -p data/raw --unzip")
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            with zf.open(CSV_NAME) as fh:
                df = pd.read_csv(fh)
    else:
        df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DatasetError(f"not the EMSCAD file: missing columns {missing}")
    df["fraudulent"] = df["fraudulent"].astype(int)
    df["jd_text"] = df.apply(build_jd_text, axis=1)
    # Training text = the posting's own words only. Section headings added by build_jd_text are
    # left out so the classifier cannot learn this tool's layout (e.g. "no company profile" showing
    # up as the bigram "time description") instead of the posting's language.
    content_cols = ["title"] + [c for c, _ in TEXT_SECTIONS]
    df["model_text"] = df.reindex(columns=content_cols).fillna("").astype(str).agg("\n".join, axis=1)
    before = len(df)
    df = df.drop_duplicates(subset="jd_text").reset_index(drop=True)
    df.attrs["duplicates_removed"] = before - len(df)
    return df


def select_postings(df: pd.DataFrame, keywords: list[str] | None = None, n: int = 5, seed: int = 42,
                    fraudulent: bool | None = False, min_chars: int = 600) -> pd.DataFrame:
    """Sample postings whose title/description match any keyword (case-insensitive)."""
    sub = df[df["jd_text"].str.len() >= min_chars]
    if fraudulent is not None:
        sub = sub[sub["fraudulent"] == int(fraudulent)]
    if keywords:
        pattern = "|".join(re.escape(k.strip()) for k in keywords if k.strip())
        hay = sub["title"].fillna("") + " " + sub.get("function", pd.Series("", index=sub.index)).fillna("")
        sub = sub[hay.str.contains(pattern, case=False, regex=True)]
    if sub.empty:
        raise DatasetError("no postings match the filters")
    return sub.sample(n=min(n, len(sub)), random_state=seed)


def export_postings(sample: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Write each posting as a Markdown file usable with ``autoapply run --jd``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for _, row in sample.iterrows():
        slug = re.sub(r"[^a-z0-9]+", "-", str(row["title"]).lower()).strip("-")[:50]
        p = out_dir / f"emscad_{row.get('job_id', _)}_{slug}.md"
        label = "FRAUDULENT (labelled scam)" if row["fraudulent"] == 1 else "legitimate"
        p.write_text(f"<!-- {SOURCE_NOTE} EMSCAD job_id={row.get('job_id', '?')}; label: {label}. -->\n\n"
                     + row["jd_text"] + "\n", encoding="utf-8")
        paths.append(p)
    return paths
