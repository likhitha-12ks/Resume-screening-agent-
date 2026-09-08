# Resume Screening Agent

An AI agent that takes a **job description** and a **folder of resumes**,
and produces a **ranked, scored shortlist with reasoning** for each
candidate.

> My agent takes a job description + a folder of resumes and produces a
> ranked CSV/JSON shortlist, each candidate scored 0–100 with a
> one-sentence explanation of why they matched (or didn't).

---

## How it works (pipeline)

```
JD + resumes  →  parse text  →  extract skills/experience/education
              →  score (TF-IDF similarity + skill overlap)
              →  reason (Groq/Llama, or offline template)
              →  rank  →  CSV + JSON + console output
```

This is the standard **Input → Think → Act → Output** loop:

- **Input**: job description + resume files (PDF / DOCX / TXT)
- **Think**: an LLM (Groq-hosted Llama) reads the extracted signals and writes a
  plain-English justification for each candidate's score
- **Act**: parses files, computes NLP similarity, extracts structured
  fields — none of this needs the LLM, which keeps the agent fast,
  cheap, and auditable
- **Output**: a ranked shortlist printed to the console and saved as
  `output/results.csv` and `output/results.json`

---

## 1. Install

Requires Python 3.10+.

```bash
git clone <this-repo-url>
cd resume-screening-agent
pip install -r requirements.txt
```

## 2. Configure your API key (optional but recommended)

The agent uses Groq (fast, free-tier-friendly Llama inference) to
generate the human-readable "reasoning" sentence for each candidate.
It runs perfectly well **without** an API key too (see
[Offline mode](#offline-mode-no-api-key)), which is useful for
reviewers who just want to see it work immediately.

```bash
export GROQ_API_KEY="gsk_..."
```

Get a free key at https://console.groq.com/keys.

Optionally override the model (default: `llama-3.3-70b-versatile`):

```bash
export GROQ_MODEL="llama-3.1-8b-instant"   # e.g. for faster/cheaper calls
```

## 3. Run it

```bash
python main.py --jd sample_data/job_description.txt --resumes sample_data/resumes
```

This screens the 10 sample resumes included in this repo (mixed
`.txt`, `.docx`, and `.pdf` — to prove multi-format parsing works)
against a sample "Backend Software Engineer" job description, and
prints a ranked shortlist to the console, then saves it to
`output/results.csv` and `output/results.json`.

### Run it on your own data

```bash
python main.py --jd path/to/your_job_description.pdf --resumes path/to/your_resumes_folder
```

- `--jd` accepts `.pdf`, `.docx`, or `.txt`
- `--resumes` must be a folder containing `.pdf`, `.docx`, and/or `.txt` files (handles 10+ resumes in one run — tested with 10 in `sample_data/resumes`)

### All options

```
--jd PATH         Path to the job description file (required)
--resumes PATH    Path to a folder of resumes (required)
--out PATH        Output file prefix (default: output/results)
--no-llm          Force offline reasoning even if GROQ_API_KEY is set
--top N           Only print the top N candidates to the console (CSV/JSON always contain all candidates)
```

### Offline mode (no API key)

```bash
python main.py --jd sample_data/job_description.txt --resumes sample_data/resumes --no-llm
```

Without a `GROQ_API_KEY` (or with `--no-llm`), the agent still runs the full
pipeline and produces the same ranking — only the *wording* of the
reasoning sentence changes, from an LLM-written sentence to a
deterministic template built from the same extracted data. **The score
and rank never depend on the LLM being available.**

---

## Scoring method

Each resume gets a score from 0–100, a weighted blend of two signals:

| Signal | Weight | What it captures |
|---|---|---|
| **TF-IDF cosine similarity** | 50% | Overall topical/contextual overlap between the full JD text and the full resume text. Catches relevant experience described in different words than the JD. |
| **Skill overlap ratio** | 50% | `(skills in both JD and resume) / (skills in JD)`. Makes the score explainable and resistant to resumes that read similarly but are missing hard requirements. |

```
score = round((0.5 * tfidf_similarity + 0.5 * skill_overlap) * 100, 2)
```

The skill vocabulary starts from a ~60-term default tech/soft-skill
list and is **extended automatically** from any `Skills:` / `Requirements:`
/ `Tech stack:` / `Preferred:` labeled line in the JD, so the agent
adapts to whatever role you paste in rather than being hard-coded to
one domain.

Alongside the score, the agent also extracts and reports (for
transparency, not part of the numeric score):
- **Experience years** — regex-matched from phrases like "5 years", "3+ yrs"
- **Education level** — highest of PhD / Master's / Bachelor's / Associate detected
- **Matched / missing skills** — explicit lists, so reviewers can sanity-check the score at a glance

---

## Design tradeoffs & limitations

- **Heuristic extraction, not an ML/NER model.** Skills, experience
  years, and education are pulled with regex/keyword matching rather
  than a trained named-entity model. This is transparent, free, fast,
  and works offline — but it will miss skills phrased in unusual ways
  and can't infer seniority beyond an explicit "N years" mention.
- **Skill vocabulary is keyword-based.** A resume that says "built
  services with FastAPI" matches; one that only implies API design
  experience through project descriptions without naming the tool
  may under-score on skill overlap (though TF-IDF similarity partially
  compensates for this).
- **TF-IDF, not embeddings.** TF-IDF was chosen over a sentence-embedding
  model (e.g. OpenAI/Cohere embeddings) to keep the agent runnable
  fully offline with `scikit-learn` and no extra API cost. It's less
  semantically aware than embeddings (e.g. won't connect "postgres"
  and "PostgreSQL" unless both appear literally) — swapping in an
  embeddings-based similarity function in `scorer.py` would be a
  natural upgrade.
- **LLM is used for explanation, not extraction or scoring.** This
  keeps the ranking deterministic and reproducible across runs (same
  inputs → same score every time), and keeps the agent usable without
  an API key. The tradeoff is that the LLM never gets a chance to
  catch nuance the heuristics missed (e.g. an unusual title implying
  seniority).
- **Scanned/image-only PDFs aren't supported.** Text extraction relies
  on the PDF having a real text layer (`pdfplumber`); scanned resumes
  without OCR will be skipped with a warning rather than silently
  scored as 0.
- **Name extraction is a heuristic** (first plausible-looking line of
  the resume) and can be wrong for unconventional resume layouts. The
  filename is used as a fallback.

---

## Project structure

```
resume-screening-agent/
├── main.py                 # CLI entry point
├── src/
│   ├── parsers.py          # PDF/DOCX/TXT → plain text
│   ├── extractor.py        # skills / experience / education extraction
│   ├── scorer.py            # TF-IDF similarity + skill-overlap scoring
│   ├── llm_client.py        # Groq reasoning (with offline fallback)
│   └── ranker.py            # orchestrates the full pipeline
├── sample_data/
│   ├── job_description.txt
│   └── resumes/              # 10 sample resumes (.txt / .docx / .pdf mix)
├── output/                   # results.csv / results.json land here
├── requirements.txt
└── README.md
```

## Example output (offline mode, sample data)

```
#1  Michael Osei  —  Score: 55.2/100
    File: sample_data/resumes/resume_09.docx
    Similarity: 33.12%  |  Skill match: 77.27%
    Experience: 9 yrs  |  Education: Master's
    Matched skills: agile, aws, ci/cd, communication, docker, fastapi, ...
    Missing skills: computer vision, nlp, nosql, pandas, rest api
    Reasoning: Overall match score 55.2/100. Matches on: agile, aws,
    ci/cd, communication, docker, fastapi (+11 more). Missing: computer
    vision, nlp, nosql, pandas (+1 more). ~9 years of relevant
    experience noted. Highest education detected: Master's.
```
