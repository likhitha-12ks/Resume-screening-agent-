"""
scorer.py
---------
Computes the relevance score for each resume against the job
description.

Two signals are blended:
  1. TF-IDF cosine similarity between the full JD text and the full
     resume text (the "NLP similarity" requirement). This captures
     overall topical/contextual overlap, not just keyword hits.
  2. Skill-overlap ratio between skills extracted from the JD and
     skills extracted from the resume. This makes the score explainable
     and resistant to resumes that are topically similar but missing
     hard requirements (or vice versa).

The blend is a simple weighted average (`SIMILARITY_WEIGHT` /
`SKILL_WEIGHT`), which keeps the method transparent and easy to justify
in the README, per the assignment's "note explaining the scoring
method" deliverable.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SIMILARITY_WEIGHT = 0.5
SKILL_WEIGHT = 0.5


def tfidf_similarity(jd_text: str, resume_texts: list[str]) -> list[float]:
    """Return cosine similarity of each resume to the JD, scaled 0-1."""
    corpus = [jd_text] + resume_texts
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        # Happens if the corpus is empty/all-stopwords; fail safe to 0s.
        return [0.0] * len(resume_texts)

    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]
    sims = cosine_similarity(jd_vector, resume_vectors)[0]
    return [float(s) for s in sims]


def skill_overlap_score(jd_skills: set[str], resume_skills: set[str]) -> float:
    if not jd_skills:
        return 0.0
    matched = jd_skills & resume_skills
    return len(matched) / len(jd_skills)


def combined_score(similarity: float, skill_score: float) -> float:
    return round((SIMILARITY_WEIGHT * similarity + SKILL_WEIGHT * skill_score) * 100, 2)
