#!/usr/bin/env python3
"""
Resume Screening Agent
=======================
Ranks a folder of resumes against a job description and outputs a
scored, ordered shortlist with reasoning.

Usage:
    python main.py --jd path/to/job_description.txt --resumes path/to/resumes_folder
    python main.py --jd sample_data/job_description.txt --resumes sample_data/resumes

Options:
    --jd PATH          Path to the job description (.txt, .pdf, or .docx)
    --resumes PATH     Path to a folder of resumes (.txt, .pdf, or .docx)
    --out PATH         Output file prefix (default: output/results) ->
                        writes PATH.csv and PATH.json
    --no-llm           Skip the Groq API call and use offline
                        templated reasoning even if GROQ_API_KEY is set
    --top N            Only print the top N candidates to the console
                        (default: show all)

Requires GROQ_API_KEY as an environment variable to use a Groq-hosted
Llama model for the reasoning text. Without it, the agent still runs
end-to-end using a deterministic offline explanation - only the wording
changes, not the ranking.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.ranker import screen_resumes  # noqa: E402



def parse_args():
    parser = argparse.ArgumentParser(description="Resume Screening Agent")
    parser.add_argument("--jd", required=True, help="Path to the job description file")
    parser.add_argument("--resumes", required=True, help="Path to folder of resumes")
    parser.add_argument("--out", default="output/results", help="Output file prefix")
    parser.add_argument("--no-llm", action="store_true", help="Disable Groq reasoning, use offline mode")
    parser.add_argument("--top", type=int, default=None, help="Only show top N in console output")
    return parser.parse_args()


def print_shortlist(results, top=None):
    shown = results[: top] if top else results
    print("\n" + "=" * 70)
    print("RESUME SCREENING RESULTS")
    print("=" * 70)
    for r in shown:
        print(f"\n#{r['rank']}  {r['name']}  —  Score: {r['score']}/100")
        print(f"    File: {r['file']}")
        print(f"    Similarity: {r['similarity_score']}%  |  Skill match: {r['skill_match_score']}%")
        print(f"    Experience: {r['experience_years']:g} yrs  |  Education: {r['education']}")
        if r["matched_skills"]:
            print(f"    Matched skills: {', '.join(r['matched_skills'])}")
        if r["missing_skills"]:
            print(f"    Missing skills: {', '.join(r['missing_skills'])}")
        print(f"    Reasoning: {r['reasoning']}")
    print("\n" + "=" * 70)
    print(f"Total candidates ranked: {len(results)}")
    print("=" * 70 + "\n")


def save_outputs(results, out_prefix):
    out_path = Path(out_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON: full fidelity, nested lists preserved.
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # CSV: flatten list fields for spreadsheet-friendliness.
    csv_rows = []
    for r in results:
        row = dict(r)
        row["matched_skills"] = "; ".join(r["matched_skills"])
        row["missing_skills"] = "; ".join(r["missing_skills"])
        csv_rows.append(row)
    csv_path = out_path.with_suffix(".csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    return csv_path, json_path


def main():
    args = parse_args()

    try:
        results = screen_resumes(args.jd, args.resumes, use_llm=not args.no_llm)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print_shortlist(results, top=args.top)
    csv_path, json_path = save_outputs(results, args.out)
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
