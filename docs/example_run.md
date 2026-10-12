# Example run: real data, offline baseline

> **What this is.** A real run of autoapply on a **real job posting** from the EMSCAD dataset (Kaggle, CC0),
> scored against **my real CV**, using the **offline baseline** (`AUTOAPPLY_MODEL=baseline`). No LLM and
> no API were involved: the "agents" here are simple heuristics. The output was checked for contact
> details before publishing (none appeared).
> The Claude agents produce a different, adaptive draft; that run needs an API key.

## 1. Single posting: "Digital Procurement Manager" (EMSCAD job_id 9728)

### Human checkpoint (`autoapply run ... --no-interactive`)

```
HUMAN REVIEW CHECKPOINT  (run <id>)
==============================================================================
Role: Digital Procurement Manager at Hayes-Corp (HK, , Hong Kong)
Fit:  84/100, recommendation: apply
      Baseline keyword overlap: 4 of 6 requirements fully matched.
Scam screen: risk LOW (model probability 4%)
   terms raising the score: excel, contract, send your, see, type

--- Cover letter draft ---
Dear Hiring Manager,

I would like to be considered for the Digital Procurement Manager role at Hayes-Corp. Relevant to your requirements: My CV notes: "M.A. International Management candidate (graduating September 2026) with three data-focused internships across strategy consulting, logistics analytics, and data science. I turn complex datasets into business decisions using Python, SQL, and Power BI, and communicate findings clearly to non-technical stakeholders. Seeking a Data Analyst position in Berlin starting July 2026.". My CV notes: "Presented results directly to internal stakeholders, sharpening data storytelling and executive communication.". My CV notes: "Top Performer (client-evaluated) on a live consulting project applying Business Model Canvas and Blue Ocean Strategy frameworks.".

I would welcome the chance to discuss the role.

[Baseline draft generated without an LLM: rewrite before sending.]

--- CV bullet suggestions ---
  [L07] M.A. International Management candidate (graduating September 2026) with three data-focused internships across strategy consulting, logistics analytics, and data science. I turn complex datasets into business decisions using Python, SQL, and Power BI, and communicate findings clearly to non-technical stakeholders. Seeking a Data Analyst position in Berlin starting July 2026.
        (was: M.A. International Management candidate (graduating September 2026) with three data-focused internships across strategy consulting, logistics analytics, and data science. I turn complex datasets into business decisions using Python, SQL, and Power BI, and communicate findings clearly to non-technical stakeholders. Seeking a Data Analyst position in Berlin starting July 2026.)
  [L13] Presented results directly to internal stakeholders, sharpening data storytelling and executive communication.
        (was: Presented results directly to internal stakeholders, sharpening data storytelling and executive communication.)
  [L27] Top Performer (client-evaluated) on a live consulting project applying Business Model Canvas and Blue Ocean Strategy frameworks.
        (was: Top Performer (client-evaluated) on a live consulting project applying Business Model Canvas and Blue Ocean Strategy frameworks.)

--- Fabrication review: PASS ---
  No issues flagged.
==============================================================================
```

### After `autoapply resume <run_id> --approve`

`STATUS.md`:

#### Digital Procurement Manager at Hayes-Corp

Approved by a human reviewer. **NOT submitted.**
autoapply never submits applications: review the files and apply yourself.


`match_report.md`:

#### Fit: 84/100 (apply)

Baseline keyword overlap: 4 of 6 requirements fully matched.

| Requirement | Status | CV lines | Reasoning |
|---|---|---|---|
| 5+ years relevant experience. | met | L08 | CV mentions experience. (1/1 key terms) |
| Negotiation, vendor management and contract management skills. | partial | L07, L24, L31 | CV mentions management, skills. (2/5 key terms) |
| Clear communication skills and a willingness to have complete responsibility for projects from start to finish. | met | L07, L13, L27 | CV mentions communication, projects (2/4 key terms) |
| Strong Excel skills, ability to work well with remote stakeholders and to travel when required. | met | L07, L11, L13 | CV mentions excel, remote, stakeholders (3/5 key terms) |
| Fluent in English and Mandarin | met | L38 | CV mentions english, fluent (2/3 key terms) |
| IT procurement background an advantage. | missing | - | No CV line mentions advantage., background, it, procurement |

**Strengths:** 5+ years relevant experience.; Clear communication skills and a willingness to have complete responsibility for projects from start to finish.; Strong Excel skills, ability to work well with remote stakeholders and to travel when required.; Fluent in English and Mandarin

**Gaps:** 


**Where the baseline fails here:** it marks *"Fluent in English and Mandarin"* as **met** because the CV
contains "English" and "fluent", and it treats "5+ years relevant experience" as partly met. Keyword
overlap cannot read meaning, so it inflates this procurement role (which needs 5+ years and Mandarin) to
84/100 "apply". This is the main argument for the LLM Matcher, and the human checkpoint is where a
person catches it.

## 2. Batch evaluation: 13 real postings (`autoapply eval --n 10 --n-fraud 3`)

### Evaluation on real job postings

Postings run: 13 (succeeded: 13, failed: 0)

| Metric | Value |
|---|---:|
| Mean match score | 56.5 |
| Recommendation apply / stretch / skip | 2 / 10 / 1 |
| Drafts passing review at checkpoint | 100% |
| Runs needing an auto-revision | 0% |
| Mean high-severity flags at checkpoint | 0.00 |
| Mean rule / LLM flags | 0.31 / 0.00 |
| Mean model calls per posting | 4.0 |
| Mean input / cache-read / output tokens | 0 / 0 / 0 |
| Labelled scams: screen risk medium/high | 100% (n=3) |
| Labelled scams: Parser reported scam signals | 0% |
| Legit postings: screen risk medium/high (false alarms) | 0% (n=10) |
| Legit postings: Parser reported scam signals | 0% |


| title | labelled scam | scam_risk | scam p | reqs found | match_score | recommendation | rule flags |
|---|---|---|---|---|---|---|---|
| Media Analyst | no | low | 0.019 | 6 | 58 | stretch | 1 |
| ETL Data Analyst | no | low | 0.057 | 2 | 88 | apply | 0 |
| Digital Procurement Manager | no | low | 0.039 | 6 | 84 | apply | 0 |
| Supply Chain Consultant | no | low | 0.075 | 9 | 56 | stretch | 0 |
| ECOMMERCE LOGISTICS AND SUPPLY CHAIN MANAGERS | no | low | 0.112 | 3 | 50 | stretch | 0 |
| Data Analyst (Marketing) | no | low | 0.024 | 4 | 62 | stretch | 1 |
| Business Analyst - Decision Sciences | no | low | 0.032 | 11 | 56 | stretch | 2 |
| Business Analyst | no | low | 0.114 | 4 | 12 | skip | 0 |
| Game Analyst | no | low | 0.007 | 10 | 55 | stretch | 0 |
| Business Analyst - CRM | no | low | 0.021 | 7 | 64 | stretch | 0 |
| Accounting Clerk - $25 | **yes** | high | 0.974 | 6 | 50 | stretch | 0 |
| Home Based Payroll Data Entry Clerk Position - Earn $100-$200 Daily  | **yes** | high | 0.994 | 1 | 50 | stretch | 0 |
| Payroll Clerk  | **yes** | high | 0.682 | 1 | 50 | stretch | 0 |

**Reading the results:**
- **Scam screen:** all 3 labelled scams scored high risk and none of the 10 legitimate postings did. The
  sample is small, and these postings may be part of the classifier's training split; the held-out
  metrics in the README are the real evidence.
- **Baseline matcher:** scores bunch at 50-64 for almost everything, *including the scams*. A payroll
  data-entry scam gets the same "stretch" as a genuine analyst role. Keyword overlap cannot tell
  relevance from coincidence, so the scam screen is what protects the user here.
- **Grounding:** the template writer only quotes CV lines verbatim, so there are no high-severity flags
  by construction; it is safe but rigid ("rewrite before sending").
