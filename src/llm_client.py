"""
llm_client.py
-------------
This is the "AI language model" brain of the agent (per the assignment's
definition: "Uses an AI language model to understand and think about
it"). It turns the structured signals computed by scorer.py/extractor.py
into a short, human-readable justification for each candidate's rank.

Works in two modes:
  - LLM mode: if GROQ_API_KEY is set, ask a Groq-hosted model (Llama 3.3
    70B by default) to write the reasoning sentence. Uses plain `requests`
    against Groq's OpenAI-compatible REST endpoint, so no extra SDK is
    required beyond what's already in requirements.txt.
  - Offline mode: otherwise, a deterministic template generates the same
    kind of sentence from the extracted data. This keeps the agent
    runnable out-of-the-box for reviewers who don't want to plug in an
    API key, while still producing genuinely useful reasoning text.

Nothing about the ranking itself depends on the LLM being available -
only the wording of the explanation changes. This mirrors STEP 3/4 of
the build guide: the model is the "understand and think" layer, the
Python code is the "glue."
"""

import os

import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are an HR screening assistant. Given a job description and one "
    "candidate's extracted profile (matched skills, missing skills, years "
    "of experience, education, similarity score), write ONE concise "
    "sentence (max 35 words) explaining why this candidate is or isn't a "
    "strong fit. Be specific and factual, do not invent information that "
    "wasn't provided."
)


def _offline_reasoning(profile: dict, jd_skills: set[str], score: float) -> str:
    matched = sorted(profile["skills"] & jd_skills)
    missing = sorted(jd_skills - profile["skills"])

    parts = [f"Overall match score {score}/100."]

    if matched:
        shown = ", ".join(matched[:6])
        extra = f" (+{len(matched) - 6} more)" if len(matched) > 6 else ""
        parts.append(f"Matches on: {shown}{extra}.")
    else:
        parts.append("No direct skill overlap with the job description was detected.")

    if missing:
        shown = ", ".join(missing[:4])
        extra = f" (+{len(missing) - 4} more)" if len(missing) > 4 else ""
        parts.append(f"Missing: {shown}{extra}.")

    if profile["experience_years"]:
        parts.append(f"~{profile['experience_years']:g} years of relevant experience noted.")

    if profile["education"] != "Not specified":
        parts.append(f"Highest education detected: {profile['education']}.")

    return " ".join(parts)


def _llm_reasoning(profile: dict, jd_text: str, jd_skills: set[str], score: float) -> str | None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None

    model = os.environ.get("GROQ_MODEL", GROQ_DEFAULT_MODEL)

    matched = sorted(profile["skills"] & jd_skills)
    missing = sorted(jd_skills - profile["skills"])

    user_prompt = (
        f"Job description (truncated):\n{jd_text[:1500]}\n\n"
        f"Candidate: {profile['name']}\n"
        f"Matched skills: {', '.join(matched) if matched else 'none'}\n"
        f"Missing skills: {', '.join(missing) if missing else 'none'}\n"
        f"Years of experience detected: {profile['experience_years']}\n"
        f"Education detected: {profile['education']}\n"
        f"Computed match score: {score}/100\n\n"
        "Write the one-sentence assessment now."
    )

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 120,
                "temperature": 0.3,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()
        return text or None
    except Exception:
        # Any API/network/parsing error -> fall back to offline reasoning
        # rather than crashing the whole batch run.
        return None


def generate_reasoning(profile: dict, jd_text: str, jd_skills: set[str], score: float) -> str:
    llm_result = _llm_reasoning(profile, jd_text, jd_skills, score)
    if llm_result:
        return llm_result
    return _offline_reasoning(profile, jd_skills, score)
