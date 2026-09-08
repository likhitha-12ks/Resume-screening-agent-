"""
extractor.py
------------
Pulls structured signals out of raw resume/JD text:
  - skills (matched against a skill vocabulary, expandable at runtime)
  - years of experience
  - education level

This is deliberately heuristic/regex-based (fast, free, offline, and
transparent) rather than a black box, so reviewers can see exactly why
a candidate got a given score. The LLM (see llm_client.py) is layered
on top to add human-readable reasoning, not to replace this signal
extraction.
"""

import re

# A reasonably broad default vocabulary. Any skill mentioned in the JD
# that ISN'T in this list is still auto-added at runtime (see
# `build_skill_vocabulary`), so the agent adapts to whatever role you
# throw at it instead of being hard-coded to one domain.
DEFAULT_SKILL_VOCAB = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "sql", "nosql", "mongodb", "postgresql", "mysql", "redis",
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd",
    "machine learning", "deep learning", "nlp", "computer vision",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "data analysis", "data engineering", "etl", "spark", "hadoop",
    "excel", "power bi", "tableau", "git", "linux", "agile", "scrum",
    "rest api", "graphql", "microservices", "html", "css",
    "project management", "communication", "leadership",
}

EDUCATION_LEVELS = [
    (r"\bph\.?d\.?\b|\bdoctorate\b", "PhD"),
    (r"\bm\.?s\.?\b|\bmaster'?s?\b|\bm\.?tech\b|\bmba\b", "Master's"),
    (r"\bb\.?s\.?\b|\bbachelor'?s?\b|\bb\.?tech\b|\bb\.?e\.?\b", "Bachelor's"),
    (r"\bassociate'?s?\b|\bdiploma\b", "Associate/Diploma"),
]

EXPERIENCE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\b",
    re.IGNORECASE,
)


SKILL_LABEL_PATTERN = re.compile(
    r"^\s*(skills?|requirements?|tech stack|preferred)\s*:\s*(.+)$",
    re.IGNORECASE,
)


def build_skill_vocabulary(jd_text: str) -> set[str]:
    """Extend the default skill vocabulary with comma-separated items found
    on explicitly labeled lines (e.g. "Skills: Python, SQL, AWS"), so the
    agent isn't blind to niche tools specific to this JD. Only lines that
    START with a clear label are used, to avoid pulling in unrelated
    sentence fragments from prose bullet points.
    """
    vocab = set(DEFAULT_SKILL_VOCAB)

    for line in jd_text.splitlines():
        match = SKILL_LABEL_PATTERN.match(line)
        if not match:
            continue
        items = match.group(2).split(",")
        for item in items:
            cleaned = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", item).strip().lower()
            # Keep short, skill-like tokens only (reject full sentences).
            if 1 < len(cleaned) < 30 and len(cleaned.split()) <= 3:
                vocab.add(cleaned)

    return vocab


def extract_skills(text: str, vocab: set[str]) -> set[str]:
    text_lower = text.lower()
    found = set()
    for skill in vocab:
        # word-boundary-ish match; skills can contain +, #, ., / so we
        # escape then relax boundaries around those characters.
        pattern = re.escape(skill).replace(r"\ ", r"\s+")
        if re.search(rf"(?<![a-zA-Z0-9]){pattern}(?![a-zA-Z0-9])", text_lower):
            found.add(skill)
    return found


def extract_experience_years(text: str) -> float:
    matches = EXPERIENCE_PATTERN.findall(text)
    if not matches:
        return 0.0
    return max(float(m) for m in matches)


def extract_education(text: str) -> str:
    text_lower = text.lower()
    for pattern, label in EDUCATION_LEVELS:
        if re.search(pattern, text_lower):
            return label
    return "Not specified"


def extract_candidate_name(text: str, fallback: str) -> str:
    """Best-effort guess at the candidate's name: usually the first
    non-empty line of a resume, if it looks like a name (short, no digits,
    no @ symbol). Falls back to the filename if nothing plausible is found.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "@" in line or any(ch.isdigit() for ch in line):
            continue
        words = line.split()
        if 1 <= len(words) <= 4 and all(w[0].isupper() for w in words if w[0].isalpha()):
            return line
        break  # only ever consider the first non-empty line
    return fallback


def profile_candidate(text: str, vocab: set[str], filename: str) -> dict:
    return {
        "name": extract_candidate_name(text, fallback=filename),
        "skills": extract_skills(text, vocab),
        "experience_years": extract_experience_years(text),
        "education": extract_education(text),
    }
