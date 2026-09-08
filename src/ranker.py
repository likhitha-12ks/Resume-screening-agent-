"""
ranker.py
---------
Orchestrates the full pipeline for one run:

    JD + resumes  -->  parse  -->  extract  -->  score  -->  reason  -->  rank

This is the "Input -> Think -> Act -> Output" loop described in the
build guide, applied to a batch of resumes instead of a single chat
turn.
"""

from pathlib import Path

from . import extractor, parsers, scorer
from .llm_client import generate_reasoning


def screen_resumes(jd_path: str, resumes_folder: str, use_llm: bool = True) -> list[dict]:
    jd_text = parsers.extract_text(jd_path)
    resume_files = parsers.collect_resume_files(resumes_folder)

    if not resume_files:
        raise FileNotFoundError(f"No supported resume files found in {resumes_folder}")

    vocab = extractor.build_skill_vocabulary(jd_text)
    jd_skills = extractor.extract_skills(jd_text, vocab)

    profiles = []
    resume_texts = []
    skipped = []

    for path in resume_files:
        try:
            text = parsers.extract_text(path)
        except Exception as exc:  # noqa: BLE001 - report and continue the batch
            skipped.append({"file": path, "error": str(exc)})
            continue

        if not text.strip():
            skipped.append({"file": path, "error": "No extractable text (possibly a scanned/image PDF)"})
            continue

        filename = Path(path).stem
        profile = extractor.profile_candidate(text, vocab, filename)
        profile["file"] = path
        profiles.append(profile)
        resume_texts.append(text)

    if not profiles:
        raise ValueError("No resumes could be parsed successfully.")

    similarities = scorer.tfidf_similarity(jd_text, resume_texts)

    results = []
    for profile, similarity in zip(profiles, similarities):
        skill_score = scorer.skill_overlap_score(jd_skills, profile["skills"])
        score = scorer.combined_score(similarity, skill_score)

        reasoning = None
        if use_llm:
            reasoning = generate_reasoning(profile, jd_text, jd_skills, score)
        else:
            from .llm_client import _offline_reasoning
            reasoning = _offline_reasoning(profile, jd_skills, score)

        results.append({
            "name": profile["name"],
            "file": profile["file"],
            "score": score,
            "similarity_score": round(similarity * 100, 2),
            "skill_match_score": round(skill_score * 100, 2),
            "matched_skills": sorted(profile["skills"] & jd_skills),
            "missing_skills": sorted(jd_skills - profile["skills"]),
            "experience_years": profile["experience_years"],
            "education": profile["education"],
            "reasoning": reasoning,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(results, start=1):
        r["rank"] = i

    if skipped:
        for s in skipped:
            print(f"[warning] Skipped {s['file']}: {s['error']}")

    return results
