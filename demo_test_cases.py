
# Install streamlit if not already installed
# !pip install streamlit # Removed this line

import os
import sys
import json
from dotenv import load_dotenv # Import load_dotenv

# Force reload 'app' module if already loaded in sys.modules
if 'app' in sys.modules:
    del sys.modules['app']

# Add the directory containing app.py to the Python path
sys.path.append('/content')

# Import from app.py, which contains the core logic
from app import ResumeStore, build_llm, evaluate_resume, answer_question, compare_candidates

"""
demo_test_cases.py
-------------------
Demonstrates the assignment's 5 example test cases end-to-end from the
command line (no Streamlit needed), using the sample resumes in
./sample_resumes/ (generate them first with generate_sample_resumes.py).

Requires a GROQ_API_KEY (or OPENAI_API_KEY if using OpenAI):
    export GROQ_API_KEY="gsk-..."
    # Optional: python generate_sample_resumes.py (if you need sample PDFs)
    python demo_test_cases.py
"""

RESUME_DIR = "/content/sample_resumes"
#RESUME_DIR = os.path.join(os.path.dirname(__file__), "sample_resumes")

JOB_DESCRIPTION = """
Data Scientist (Mid-Level)

We are looking for a Data Scientist with:
- 3+ years of experience in Python, SQL, and statistical analysis
- Hands-on experience building and deploying machine learning models (e.g. scikit-learn, XGBoost, TensorFlow, or PyTorch)
- Experience designing and analyzing A/B tests
- Experience with cloud ML platforms (AWS SageMaker, GCP Vertex AI, or similar)
- Strong communication skills and experience presenting insights to stakeholders (Tableau/Power BI a plus)
- Bachelor's degree in a quantitative field (Master's preferred)
"""


def line(title=""):
    print("\n" + "=" * 90)
    if title:
        print(title)
        print("=" * 90)


def print_evaluation(name, ev):
    print(f"\nCandidate file: {name}")
    print(f"Candidate name (from resume): {ev.candidate_name}")
    print(f"Match Score: {ev.match_score}/100")
    print(f"Recommendation: {ev.recommendation}")
    print(f"Summary: {ev.summary}")
    print(f"Matching Skills: {', '.join(ev.matching_skills)}")
    print(f"Missing Skills: {', '.join(ev.missing_skills)}")
    print("Strengths:")
    for s in ev.strengths:
        print(f"  - {s}")
    print("Weaknesses:")
    for w in ev.weaknesses:
        print(f"  - {w}")
    print(f"Justification: {ev.justification}")


def main():
    # Load environment variables from .env file
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY not found. Please ensure it's set in a .env file or as an environment variable.")

    resumes = {
        "Resume_A_Priya_Sharma.pdf": "Resume A (Priya Sharma)",
        "Resume_B_James_Whitfield.pdf": "Resume B (James Whitfield)",
        "Resume_C_Diego_Alvarez.pdf": "Resume C (Diego Alvarez)",
    }
    for fname in resumes:
        path = os.path.join(RESUME_DIR, fname)
        if not os.path.exists(path):
            sys.exit(f"Missing {path}. Run `python generate_sample_resumes.py` first.")

    # The ResumeStore and build_llm functions now expect the groq_api_key
    store = ResumeStore() # ResumeStore no longer needs an API key in its constructor
    llm = build_llm(groq_api_key=api_key)

    for fname in resumes:
        store.add_resume_from_path(os.path.join(RESUME_DIR, fname), fname)

    # ----------------------------------------------------------------- #
    # Test Case 1: Evaluate Resume A for a Data Scientist role
    # ----------------------------------------------------------------- #
    line("TEST CASE 1: Evaluate Resume A for the Data Scientist role")
    eval_a = evaluate_resume(store, llm, "Resume_A_Priya_Sharma.pdf", JOB_DESCRIPTION)
    print_evaluation("Resume_A_Priya_Sharma.pdf", eval_a)

    # ----------------------------------------------------------------- #
    # Test Case 2: Compare Resume A and Resume B for the same JD
    # ----------------------------------------------------------------- #
    line("TEST CASE 2: Compare Resume A and Resume B for the same JD")
    eval_b = evaluate_resume(store, llm, "Resume_B_James_Whitfield.pdf", JOB_DESCRIPTION)
    print_evaluation("Resume_B_James_Whitfield.pdf", eval_b)

    ab_comparison = compare_candidates(
        llm,
        {
            "Resume_A_Priya_Sharma.pdf": eval_a,
            "Resume_B_James_Whitfield.pdf": eval_b,
        },
    )
    print("\n--- A vs. B Comparison ---")
    print(f"Ranking: {ab_comparison.ranking}")
    print(f"Best candidate: {ab_comparison.best_candidate}")
    print(f"Justification: {ab_comparison.justification}")

    # ----------------------------------------------------------------- #
    # Test Case 3: Identify missing skills in Resume C
    # ----------------------------------------------------------------- #
    line("TEST CASE 3: Identify missing skills in Resume C")
    eval_c = evaluate_resume(store, llm, "Resume_C_Diego_Alvarez.pdf", JOB_DESCRIPTION)
    print_evaluation("Resume_C_Diego_Alvarez.pdf", eval_c)
    print("\n--- Direct RAG Q&A version ---")
    answer = answer_question(
        store, llm, "Resume_C_Diego_Alvarez.pdf",
        "What skills required for a Data Scientist role (Python ML, A/B testing, cloud ML) are missing from this resume?"
    )
    print(answer)

    # ----------------------------------------------------------------- #
    # Test Case 4: Recommend the best candidate among multiple resumes
    # ----------------------------------------------------------------- #
    line("TEST CASE 4: Recommend the best candidate among all resumes")
    all_evals = {
        "Resume_A_Priya_Sharma.pdf": eval_a,
        "Resume_B_James_Whitfield.pdf": eval_b,
        "Resume_C_Diego_Alvarez.pdf": eval_c,
    }
    full_comparison = compare_candidates(llm, all_evals)
    print(f"Ranking (best to worst): {full_comparison.ranking}")
    print(f"Best candidate: {full_comparison.best_candidate}")
    print(f"Justification: {full_comparison.justification}")

    # ----------------------------------------------------------------- #
    # Test Case 5: Generate a hiring recommendation with justification
    # ----------------------------------------------------------------- #
    line("TEST CASE 5: Hiring recommendation with justification (Resume A)")
    print(f"Recommendation: {eval_a.recommendation}")
    print(f"Justification: {eval_a.justification}")

    # Save all structured results to JSON for the report/submission
    out_path = "/content/demo_results.json" # Directly specify path since __file__ is not defined
    with open(out_path, "w") as f:
        json.dump(
            {
                "job_description": JOB_DESCRIPTION,
                "evaluations": {k: v.model_dump() for k, v in all_evals.items()},
                "comparison": full_comparison.model_dump(),
            },
            f,
            indent=2,
        )
    line(f"All structured results saved to {out_path}")


if __name__ == "__main__":
    main()
