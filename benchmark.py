"""
benchmark.py
Evaluation script for the Resume-to-Job Matching pipeline.
Tests semantic similarity scores and skill extraction accuracy on known test cases.

Run with: python benchmark.py
"""

import json
from utils.embeddings import compute_semantic_similarity, compute_skill_overlap_score

# ─── Test Cases ───────────────────────────────────────────────────────────────
# Format: (resume_skills, jd_skills, expected_overlap_range)
SKILL_OVERLAP_TESTS = [
    {
        "name": "Perfect Match — Python Backend Dev",
        "resume_skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS", "Git", "REST APIs"],
        "jd_skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS", "REST APIs"],
        "expected_min": 90,
    },
    {
        "name": "Partial Match — Frontend to Backend",
        "resume_skills": ["JavaScript", "React", "CSS", "HTML", "Node.js", "MongoDB"],
        "jd_skills": ["Python", "Django", "PostgreSQL", "Docker", "AWS", "Redis"],
        "expected_max": 40,
    },
    {
        "name": "Moderate Match — Data to ML Engineer",
        "resume_skills": ["Python", "Pandas", "SQL", "Tableau", "Excel", "Statistics"],
        "jd_skills": ["Python", "PyTorch", "TensorFlow", "MLOps", "Kubernetes", "scikit-learn"],
        "expected_min": 20,
        "expected_max": 60,
    },
    {
        "name": "Semantic Match — Similar but Different Names",
        "resume_skills": ["GCP", "Google Cloud Platform", "BigQuery", "Pub/Sub"],
        "jd_skills": ["AWS", "Google Cloud", "Cloud Storage", "Message Queue"],
        "expected_min": 30,
    },
]

# Semantic similarity test cases
SEMANTIC_SIMILARITY_TESTS = [
    {
        "name": "High Similarity — Same Domain",
        "text1": "Experienced Python developer with 5 years building REST APIs using FastAPI and Django. "
                 "Strong background in PostgreSQL, Redis, and Docker. AWS certified. Agile team environment.",
        "text2": "We are looking for a Python backend engineer to build and maintain REST APIs. "
                 "Must have Django or FastAPI experience, PostgreSQL, Docker, and AWS familiarity. "
                 "Agile development process.",
        "expected_min": 0.70,
    },
    {
        "name": "Low Similarity — Different Domains",
        "text1": "Graphic designer with 4 years experience in Figma, Adobe Creative Suite, "
                 "brand identity, print media and visual communication.",
        "text2": "Senior data engineer needed. Must know Spark, Hadoop, Kafka, dbt, "
                 "Airflow, Python, SQL, and cloud data warehouses (Snowflake, BigQuery).",
        "expected_max": 0.40,
    },
    {
        "name": "Moderate Similarity — Adjacent Fields",
        "text1": "Machine learning engineer skilled in Python, scikit-learn, TensorFlow, "
                 "model training, A/B testing and data pipelines.",
        "text2": "Data scientist role: statistical modeling, Python, R, Jupyter, "
                 "ML algorithms, Tableau, and business analytics.",
        "expected_min": 0.45,
        "expected_max": 0.85,
    },
]


def run_skill_overlap_tests():
    """Run skill overlap scoring tests."""
    print("\n" + "="*60)
    print("SKILL OVERLAP SCORING — Benchmark Results")
    print("="*60)

    passed = 0
    for test in SKILL_OVERLAP_TESTS:
        result = compute_skill_overlap_score(test["resume_skills"], test["jd_skills"])
        score = result["score"]
        min_ok = score >= test.get("expected_min", 0)
        max_ok = score <= test.get("expected_max", 100)
        status = "✅ PASS" if (min_ok and max_ok) else "❌ FAIL"
        if min_ok and max_ok:
            passed += 1

        print(f"\n{status} | {test['name']}")
        print(f"       Score: {score:.1f}% | "
              f"Expected: [{test.get('expected_min', 0)}–{test.get('expected_max', 100)}]")
        print(f"       Exact matches: {result['exact_matches']}")
        print(f"       Semantic matches: {len(result['semantic_matches'])}")

    total = len(SKILL_OVERLAP_TESTS)
    print(f"\n{'─'*60}")
    print(f"Skill Overlap Tests: {passed}/{total} passed "
          f"({passed/total*100:.0f}% accuracy)")
    return passed, total


def run_semantic_similarity_tests():
    """Run semantic similarity scoring tests."""
    print("\n" + "="*60)
    print("SEMANTIC SIMILARITY — Benchmark Results")
    print("="*60)

    passed = 0
    for test in SEMANTIC_SIMILARITY_TESTS:
        score = compute_semantic_similarity(test["text1"], test["text2"])
        min_ok = score >= test.get("expected_min", 0.0)
        max_ok = score <= test.get("expected_max", 1.0)
        status = "✅ PASS" if (min_ok and max_ok) else "❌ FAIL"
        if min_ok and max_ok:
            passed += 1

        print(f"\n{status} | {test['name']}")
        print(f"       Score: {score:.3f} | "
              f"Expected: [{test.get('expected_min', 0):.2f}–{test.get('expected_max', 1.0):.2f}]")

    total = len(SEMANTIC_SIMILARITY_TESTS)
    print(f"\n{'─'*60}")
    print(f"Semantic Similarity Tests: {passed}/{total} passed "
          f"({passed/total*100:.0f}% accuracy)")
    return passed, total


if __name__ == "__main__":
    print("🚀 Running ResumeMatch AI Benchmark Evaluation...")
    print("   (Uses sentence-transformers only — no API cost)")

    s_passed, s_total = run_skill_overlap_tests()
    sim_passed, sim_total = run_semantic_similarity_tests()

    total_passed = s_passed + sim_passed
    total_tests = s_total + sim_total

    print("\n" + "="*60)
    print("OVERALL BENCHMARK SUMMARY")
    print("="*60)
    print(f"  Total Tests  : {total_tests}")
    print(f"  Passed       : {total_passed}")
    print(f"  Failed       : {total_tests - total_passed}")
    print(f"  Accuracy     : {total_passed/total_tests*100:.1f}%")
    print("="*60)

    # Save results to JSON
    results = {
        "skill_overlap": {"passed": s_passed, "total": s_total},
        "semantic_similarity": {"passed": sim_passed, "total": sim_total},
        "overall": {"passed": total_passed, "total": total_tests,
                    "accuracy_percent": round(total_passed/total_tests*100, 1)},
    }
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n📊 Results saved to benchmark_results.json")
