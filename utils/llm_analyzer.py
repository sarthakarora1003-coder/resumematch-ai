"""
llm_analyzer.py
Handles all Gemini API calls with structured Pydantic output schemas.
Uses a SINGLE combined API call to stay within free tier rate limits (5 req/min).
"""

import os
import json
import re
import google.generativeai as genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


def configure_gemini(api_key: str = None):
    """Configure the Gemini client."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY not found. Set it in .env or pass it directly.")
    genai.configure(api_key=key)
    return genai.GenerativeModel("gemini-3.8-flash")


# ─── Pydantic Schemas ────────────────────────────────────────────────────────

class SkillSet(BaseModel):
    technical_skills: list[str] = Field(description="Programming languages, frameworks, tools, platforms")
    soft_skills: list[str] = Field(description="Communication, leadership, teamwork, etc.")
    domain_knowledge: list[str] = Field(description="Industry domains, methodologies, business knowledge")
    certifications: list[str] = Field(description="Professional certifications and credentials")
    education_level: str = Field(description="Highest education level (e.g., Bachelor's in CS)")


class JobRequirements(BaseModel):
    required_skills: list[str] = Field(description="Must-have technical and soft skills")
    preferred_skills: list[str] = Field(description="Nice-to-have skills")
    required_experience_years: int = Field(description="Minimum years of experience required")
    job_title: str = Field(description="The job title/role")
    industry: str = Field(description="Industry or domain")
    key_responsibilities: list[str] = Field(description="Main job responsibilities")


class MatchAnalysis(BaseModel):
    overall_score: int = Field(description="ATS compatibility score 0-100")
    technical_match_score: int = Field(description="Technical skills match 0-100")
    experience_match_score: int = Field(description="Experience relevance score 0-100")
    matched_skills: list[str] = Field(description="Skills present in both resume and JD")
    missing_required_skills: list[str] = Field(description="Required skills absent from resume")
    missing_preferred_skills: list[str] = Field(description="Preferred skills absent from resume")
    strengths: list[str] = Field(description="Candidate's strongest alignment points")
    weaknesses: list[str] = Field(description="Candidate's biggest gaps")
    hire_recommendation: str = Field(description="Strong Match / Moderate Match / Weak Match")
    summary: str = Field(description="2-3 sentence executive summary of the match")


class ImprovedBullets(BaseModel):
    original: str = Field(description="Original bullet point")
    improved: str = Field(description="Rewritten bullet with action verbs and quantifiable impact")
    reasoning: str = Field(description="Why this change improves ATS ranking")


class ResumeImprovements(BaseModel):
    ats_keyword_gaps: list[str] = Field(description="High-value JD keywords missing from resume")
    improved_bullets: list[ImprovedBullets] = Field(description="Rewritten resume bullet points")
    suggested_skills_to_add: list[str] = Field(description="Skills to learn or highlight")
    cover_letter_talking_points: list[str] = Field(description="Key points for cover letter")
    action_plan: list[str] = Field(description="Prioritized steps to improve candidacy")


# ─── Helper ───────────────────────────────────────────────────────────────────

def _extract_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code fences."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        json_str = match.group(1).strip()
    else:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            json_str = match.group(0)
        else:
            raise ValueError("No JSON found in LLM response")
    return json.loads(json_str)


# ─── Single Combined API Call ─────────────────────────────────────────────────

def run_full_analysis(
    model,
    resume_text: str,
    jd_text: str,
    semantic_score: float,
) -> tuple[SkillSet, JobRequirements, MatchAnalysis, ResumeImprovements]:
    """
    Run the COMPLETE analysis in a SINGLE Gemini API call.
    Extracts skills, parses JD, computes match, and generates improvements
    all in one request — avoids hitting the free tier 5 req/min rate limit.
    """
    prompt = f"""
You are an expert NLP-based resume analyzer, ATS specialist, and career coach.
Analyze the resume and job description below, then return ONE comprehensive JSON object.

SEMANTIC SIMILARITY SCORE (embedding cosine similarity, 0-1): {semantic_score:.3f}

RESUME:
{resume_text[:4000]}

JOB DESCRIPTION:
{jd_text[:3000]}

Return ONLY a single valid JSON object with this exact structure — no extra text outside the JSON:

{{
  "resume_skills": {{
    "technical_skills": ["Python", "FastAPI", "..."],
    "soft_skills": ["Communication", "..."],
    "domain_knowledge": ["Backend Development", "..."],
    "certifications": ["AWS Certified", "..."],
    "education_level": "Bachelor's in Computer Science"
  }},
  "job_requirements": {{
    "required_skills": ["Python", "Docker", "..."],
    "preferred_skills": ["Kubernetes", "..."],
    "required_experience_years": 3,
    "job_title": "Software Engineer",
    "industry": "Technology",
    "key_responsibilities": ["Build APIs", "Maintain databases", "..."]
  }},
  "match_analysis": {{
    "overall_score": 75,
    "technical_match_score": 80,
    "experience_match_score": 70,
    "matched_skills": ["Python", "FastAPI", "..."],
    "missing_required_skills": ["Docker", "..."],
    "missing_preferred_skills": ["Kubernetes", "..."],
    "strengths": ["Strong Python background", "..."],
    "weaknesses": ["No Docker experience", "..."],
    "hire_recommendation": "Strong Match",
    "summary": "The candidate is a strong match with solid Python skills. However, Docker experience is missing. Overall a competitive applicant."
  }},
  "improvements": {{
    "ats_keyword_gaps": ["Docker", "CI/CD", "..."],
    "improved_bullets": [
      {{
        "original": "Worked on backend systems",
        "improved": "Architected and deployed 3 RESTful microservices using FastAPI, reducing API response time by 40%",
        "reasoning": "Adds action verb, quantifiable metric, and relevant ATS keywords"
      }},
      {{
        "original": "Did database work",
        "improved": "Optimized 15+ PostgreSQL queries reducing average query time from 800ms to 120ms",
        "reasoning": "Quantifies impact and uses JD keyword PostgreSQL"
      }},
      {{
        "original": "Helped with deployments",
        "improved": "Automated deployment pipeline using Docker and GitHub Actions, cutting release time by 60%",
        "reasoning": "Introduces missing keyword Docker which is required in the JD"
      }}
    ],
    "suggested_skills_to_add": ["Docker", "Kubernetes", "..."],
    "cover_letter_talking_points": ["Highlight FastAPI expertise", "Address Docker gap proactively", "..."],
    "action_plan": ["Complete a Docker crash course this week", "Add Docker to resume projects", "..."]
  }}
}}

JSON:
"""
    response = model.generate_content(prompt)
    data = _extract_json_from_response(response.text)

    resume_skills = SkillSet(**data["resume_skills"])
    job_req = JobRequirements(**data["job_requirements"])
    match = MatchAnalysis(**data["match_analysis"])
    improvements = ResumeImprovements(**data["improvements"])

    return resume_skills, job_req, match, improvements
