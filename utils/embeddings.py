"""
embeddings.py
Computes semantic similarity between resume and job description
using sentence-transformers (local, no API cost).
"""

import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load model once and cache it for the entire session
@st.cache_resource
def _get_model() -> SentenceTransformer:
    """Load the sentence transformer model once and keep it cached."""
    return SentenceTransformer("all-MiniLM-L6-v2")



def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two texts using sentence embeddings.
    Returns a float between 0.0 and 1.0.
    """
    model = _get_model()
    embeddings = model.encode([text1[:5000], text2[:5000]])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(score)


def compute_skill_overlap_score(resume_skills: list[str], jd_skills: list[str]) -> dict:
    """
    Compute exact + semantic skill overlap between resume and JD skill lists.
    Returns a dict with exact matches, fuzzy matches, and a composite score.
    """
    if not resume_skills or not jd_skills:
        return {"exact_matches": [], "score": 0.0}

    model = _get_model()

    # Normalize to lowercase
    resume_lower = [s.lower().strip() for s in resume_skills]
    jd_lower = [s.lower().strip() for s in jd_skills]

    # Exact matches
    exact_matches = [s for s in jd_lower if s in resume_lower]

    # Semantic matches for non-exact skills
    unmatched_jd = [s for s in jd_skills if s.lower().strip() not in resume_lower]
    unmatched_resume = [s for s in resume_skills if s.lower().strip() not in jd_lower]

    semantic_matches = []
    if unmatched_jd and unmatched_resume:
        jd_embeddings = model.encode(unmatched_jd)
        resume_embeddings = model.encode(unmatched_resume)
        sim_matrix = cosine_similarity(jd_embeddings, resume_embeddings)

        for i, jd_skill in enumerate(unmatched_jd):
            best_score = np.max(sim_matrix[i])
            best_match_idx = np.argmax(sim_matrix[i])
            if best_score > 0.75:  # threshold for semantic similarity
                semantic_matches.append({
                    "jd_skill": jd_skill,
                    "resume_skill": unmatched_resume[best_match_idx],
                    "similarity": float(best_score),
                })

    total_jd = len(jd_skills)
    matched = len(exact_matches) + len(semantic_matches)
    composite_score = (matched / total_jd * 100) if total_jd > 0 else 0

    return {
        "exact_matches": exact_matches,
        "semantic_matches": semantic_matches,
        "score": round(composite_score, 1),
        "total_jd_skills": total_jd,
        "total_matched": matched,
    }
