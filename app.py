"""
app.py
Main Streamlit application — Resume-to-Job Matching & Skill Gap Explainer with ATS Parsing.

Run with: streamlit run app.py
"""

import streamlit as st
import os
from dotenv import load_dotenv

from utils.resume_parser import extract_text_from_upload, parse_resume
from utils.llm_analyzer import (
    configure_gemini,
    run_full_analysis,
)
from utils.embeddings import compute_semantic_similarity, compute_skill_overlap_score
from utils.charts import gauge_chart, skills_bar_chart, radar_chart

load_dotenv()

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeMatch AI — ATS Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide sidebar and its toggle arrow completely */
    [data-testid="collapsedControl"] { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }

    .main { background-color: #f8fafc; }
    /* Fix invisible metric text */
    [data-testid="stMetric"] {
        background: white; padding: 16px; border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    [data-testid="stMetric"] label,
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] span {
        color: #64748b !important; font-size: 0.85rem !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        color: #0f172a !important; font-weight: 700 !important;
    }
    .skill-chip-green {
        display: inline-block; background: #dcfce7; color: #166534;
        padding: 4px 10px; border-radius: 20px; margin: 3px;
        font-size: 0.8rem; font-weight: 500;
    }
    .skill-chip-red {
        display: inline-block; background: #fee2e2; color: #991b1b;
        padding: 4px 10px; border-radius: 20px; margin: 3px;
        font-size: 0.8rem; font-weight: 500;
    }
    .skill-chip-amber {
        display: inline-block; background: #fef3c7; color: #92400e;
        padding: 4px 10px; border-radius: 20px; margin: 3px;
        font-size: 0.8rem; font-weight: 500;
    }
    .score-badge-green { color: #16a34a; font-weight: 700; font-size: 1.1rem; }
    .score-badge-amber { color: #d97706; font-weight: 700; font-size: 1.1rem; }
    .score-badge-red   { color: #dc2626; font-weight: 700; font-size: 1.1rem; }
    .card {
        background: white; padding: 20px; border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 16px;
    }
    h1, h2, h3, h4 { color: #ffffff !important; }
    p, label, .stMarkdown p { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ─── API Key (loaded from .env, editable via top expander) ────────────────────
api_key = os.getenv("GEMINI_API_KEY", "")
with st.expander("⚙️ API Key Configuration", expanded=not bool(api_key)):
    api_key = st.text_input(
        "Gemini API Key",
        value=api_key,
        type="password",
        help="Get your free key at aistudio.google.com — save it in .env to auto-load",
    )

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
    <h1 style='font-size:2.4rem; color:#1e293b;'>🎯 ResumeMatch AI</h1>
    <p style='color:#64748b; font-size:1.1rem;'>
        Upload your resume + paste a Job Description → get your ATS score,
        skill gap analysis, and personalized improvement plan powered by LLM.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─── Input Section ────────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📄 Your Resume")
    uploaded_file = st.file_uploader(
        "Upload resume (PDF or TXT)",
        type=["pdf", "txt"],
        help="Supports PDF and plain text resumes",
    )
    resume_text_manual = st.text_area(
        "Or paste resume text here",
        height=200,
        placeholder="Paste your resume content here if not uploading a file...",
    )

with col2:
    st.subheader("💼 Job Description")
    jd_text = st.text_area(
        "Paste the full job description",
        height=300,
        placeholder="""Example:
Software Engineer — Backend
Company: TechCorp

Requirements:
• 3+ years Python experience
• Proficiency in FastAPI, Django
• Experience with AWS or GCP
• Strong SQL skills (PostgreSQL)
• Familiarity with Docker & Kubernetes
• Experience with REST API design
• Good communication skills
...
        """,
    )

st.divider()

# ─── Analyze Button ───────────────────────────────────────────────────────────
analyze_btn = st.button(
    "🚀 Analyze Resume & Generate Report",
    type="primary",
    use_container_width=True,
)

# ─── Analysis Pipeline ────────────────────────────────────────────────────────
if analyze_btn:
    # Validate inputs
    resume_text = ""
    if uploaded_file:
        resume_text = extract_text_from_upload(uploaded_file)
    elif resume_text_manual.strip():
        resume_text = resume_text_manual.strip()

    if not resume_text:
        st.error("⚠️ Please upload a resume or paste resume text.")
        st.stop()
    if not jd_text.strip():
        st.error("⚠️ Please paste a Job Description.")
        st.stop()
    if not api_key:
        st.error("⚠️ Please enter your Gemini API key in the configuration section above.")
        st.stop()

    # ── Configure Gemini ──
    try:
        model = configure_gemini(api_key)
    except Exception as e:
        st.error(f"❌ Failed to connect to Gemini API: {e}")
        st.stop()

    # ── Pipeline Progress ──
    progress = st.progress(0, text="Starting NLP pipeline...")

    with st.spinner("Step 1/3 — Traditional NLP parsing (spaCy + regex)..."):
        parsed = parse_resume(resume_text)
        progress.progress(15, text="Step 1/3 — Traditional NLP parsing complete ✓")

    with st.spinner("Step 2/3 — Computing semantic similarity (sentence-transformers)..."):
        semantic_score = compute_semantic_similarity(resume_text, jd_text)
        progress.progress(35, text="Step 2/3 — Semantic similarity computed ✓")

    with st.spinner("Step 3/3 — Running full LLM analysis with Gemini..."):
        try:
            resume_skills, job_req, match, improvements = run_full_analysis(
                model, resume_text, jd_text, semantic_score
            )
            skill_overlap = compute_skill_overlap_score(
                resume_skills.technical_skills + resume_skills.soft_skills,
                job_req.required_skills + job_req.preferred_skills,
            )
        except Exception as e:
            st.error(f"❌ LLM extraction failed: {e}")
            st.stop()
        progress.progress(100, text="✅ Analysis complete!")

    st.success("✅ Analysis complete! Scroll down to view your full report.")
    st.divider()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1 — EXECUTIVE SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    st.header("📊 Executive Summary")

    rec = match.hire_recommendation
    if "Strong" in rec:
        badge_class = "score-badge-green"
        rec_icon = "🟢"
    elif "Moderate" in rec:
        badge_class = "score-badge-amber"
        rec_icon = "🟡"
    else:
        badge_class = "score-badge-red"
        rec_icon = "🔴"

    st.markdown(f"""
    <div class='card'>
        <h3>Recommendation: <span class='{badge_class}'>{rec_icon} {rec}</span></h3>
        <p style='color:#475569; font-size:1rem; margin-top:8px;'>{match.summary}</p>
    </div>
    """, unsafe_allow_html=True)

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.plotly_chart(gauge_chart(match.overall_score, "ATS Score"), use_container_width=True)
    with g2:
        st.plotly_chart(gauge_chart(match.technical_match_score, "Technical Match"), use_container_width=True)
    with g3:
        st.plotly_chart(gauge_chart(match.experience_match_score, "Experience Match"), use_container_width=True)
    with g4:
        semantic_pct = int(semantic_score * 100)
        st.plotly_chart(gauge_chart(semantic_pct, "Semantic Similarity"), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2 — SKILLS ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    st.header("🛠️ Skills Analysis")

    tab1, tab2, tab3 = st.tabs(["Match Overview", "Skill Details", "Radar Chart"])

    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(
                skills_bar_chart(match.matched_skills, match.missing_required_skills, match.missing_preferred_skills),
                use_container_width=True,
            )
        with c2:
            st.subheader("✅ Matched Skills")
            chips = " ".join([f"<span class='skill-chip-green'>{s}</span>" for s in match.matched_skills])
            st.markdown(chips or "<i>No matched skills found.</i>", unsafe_allow_html=True)

            st.subheader("❌ Missing Required Skills")
            chips_red = " ".join([f"<span class='skill-chip-red'>{s}</span>" for s in match.missing_required_skills])
            st.markdown(chips_red or "<i>None — great match!</i>", unsafe_allow_html=True)

            st.subheader("⚠️ Missing Preferred Skills")
            chips_amber = " ".join([f"<span class='skill-chip-amber'>{s}</span>" for s in match.missing_preferred_skills])
            st.markdown(chips_amber or "<i>None</i>", unsafe_allow_html=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Your Resume Skills")
            st.markdown("**Technical Skills:**")
            st.markdown(", ".join(resume_skills.technical_skills) or "—")
            st.markdown("**Soft Skills:**")
            st.markdown(", ".join(resume_skills.soft_skills) or "—")
            st.markdown("**Domain Knowledge:**")
            st.markdown(", ".join(resume_skills.domain_knowledge) or "—")
            st.markdown("**Certifications:**")
            st.markdown(", ".join(resume_skills.certifications) or "—")
            st.markdown(f"**Education:** {resume_skills.education_level}")

        with c2:
            st.subheader("Job Requirements")
            st.markdown(f"**Role:** {job_req.job_title}")
            st.markdown(f"**Industry:** {job_req.industry}")
            st.markdown(f"**Experience Required:** {job_req.required_experience_years}+ years")
            st.markdown("**Required Skills:**")
            st.markdown(", ".join(job_req.required_skills) or "—")
            st.markdown("**Preferred Skills:**")
            st.markdown(", ".join(job_req.preferred_skills) or "—")
            st.markdown("**Key Responsibilities:**")
            for r in job_req.key_responsibilities[:5]:
                st.markdown(f"• {r}")

    with tab3:
        radar_scores = {
            "ATS Score": match.overall_score,
            "Technical": match.technical_match_score,
            "Experience": match.experience_match_score,
            "Semantic": int(semantic_score * 100),
            "Skill Overlap": int(skill_overlap["score"]),
        }
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(radar_chart(radar_scores), use_container_width=True)
        with c2:
            st.subheader("Score Breakdown")
            for name, score in radar_scores.items():
                color = "#22c55e" if score >= 70 else "#f59e0b" if score >= 40 else "#ef4444"
                st.markdown(f"""
                <div style='display:flex; justify-content:space-between; align-items:center;
                            padding:8px 0; border-bottom:1px solid #f1f5f9;'>
                    <span style='color:#374151; font-weight:500;'>{name}</span>
                    <span style='color:{color}; font-weight:700; font-size:1.1rem;'>{score}%</span>
                </div>
                """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3 — STRENGTHS & WEAKNESSES
    # ═══════════════════════════════════════════════════════════════════
    st.header("💪 Strengths & Gaps")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("✅ Strengths")
        for strength in match.strengths:
            st.markdown(f"""
            <div style='background:#f0fdf4; border-left:4px solid #22c55e;
                        padding:10px 14px; border-radius:0 8px 8px 0; margin-bottom:8px;
                        color:#166534;'>
                {strength}
            </div>""", unsafe_allow_html=True)

    with c2:
        st.subheader("⚠️ Gaps to Address")
        for weakness in match.weaknesses:
            st.markdown(f"""
            <div style='background:#fff7ed; border-left:4px solid #f97316;
                        padding:10px 14px; border-radius:0 8px 8px 0; margin-bottom:8px;
                        color:#9a3412;'>
                {weakness}
            </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 4 — TRADITIONAL NLP INSIGHTS
    # ═══════════════════════════════════════════════════════════════════
    st.header("🔬 NLP Parsing Insights")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Est. Experience", f"{parsed['estimated_experience_years']} yrs")
    with c2:
        sections = parsed["sections_detected"]
        present = sum(sections.values())
        st.metric("Sections Detected", f"{present}/6")
    with c3:
        st.metric("Semantic Score", f"{int(semantic_score * 100)}%")

    st.subheader("Resume Section Checklist")
    sec_cols = st.columns(3)
    section_labels = {
        "education": "🎓 Education",
        "experience": "💼 Experience",
        "skills": "🛠️ Skills",
        "projects": "📁 Projects",
        "certifications": "🏆 Certifications",
        "achievements": "⭐ Achievements",
    }
    for i, (key, label) in enumerate(section_labels.items()):
        with sec_cols[i % 3]:
            found = parsed["sections_detected"].get(key, False)
            st.markdown(f"{'✅' if found else '❌'} {label}")

    with st.expander("📌 Named Entities Extracted (spaCy NER)"):
        ent_cols = st.columns(4)
        ent_labels = {"PERSON": "People", "ORG": "Organizations", "GPE": "Locations", "DATE": "Dates"}
        for i, (key, label) in enumerate(ent_labels.items()):
            with ent_cols[i]:
                st.markdown(f"**{label}**")
                entities = parsed["entities"].get(key, [])
                for e in entities[:8]:
                    st.markdown(f"• {e}")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 5 — IMPROVEMENT PLAN
    # ═══════════════════════════════════════════════════════════════════
    st.header("🚀 Your Improvement Plan")

    st.subheader("🔑 High-Value ATS Keywords to Add")
    kw_html = " ".join([
        f"<span class='skill-chip-red'>{kw}</span>"
        for kw in improvements.ats_keyword_gaps
    ])
    st.markdown(kw_html, unsafe_allow_html=True)

    st.subheader("✏️ Resume Bullet Point Rewrites")
    for i, bullet in enumerate(improvements.improved_bullets, 1):
        with st.expander(f"Bullet {i}: {bullet.original[:60]}..."):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**❌ Original:**")
                st.markdown(f"> {bullet.original}")
            with c2:
                st.markdown("**✅ Improved:**")
                st.markdown(f"> {bullet.improved}")
            st.info(f"💡 **Why this helps:** {bullet.reasoning}")

    st.subheader("📚 Skills to Learn or Highlight")
    for skill in improvements.suggested_skills_to_add:
        st.markdown(f"• {skill}")

    st.subheader("📝 Cover Letter Talking Points")
    for point in improvements.cover_letter_talking_points:
        st.markdown(f"""
        <div style='background:#eff6ff; border-left:4px solid #3b82f6;
                    padding:10px 14px; border-radius:0 8px 8px 0; margin-bottom:8px;
                    color:#1e40af;'>
            {point}
        </div>""", unsafe_allow_html=True)

    st.subheader("🗺️ Prioritized Action Plan")
    for i, step in enumerate(improvements.action_plan, 1):
        st.markdown(f"""
        <div style='background:#faf5ff; border-left:4px solid #a855f7;
                    padding:10px 14px; border-radius:0 8px 8px 0; margin-bottom:8px;
                    color:#6b21a8;'>
            <strong>Step {i}:</strong> {step}
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.caption(
        "ResumeMatch AI · NLP Pipeline: spaCy NER + Regex → Sentence-Transformers (semantic similarity) "
        "→ Google Gemini API (structured extraction & analysis) → Streamlit"
    )

else:
    # ─── Landing State ─────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding:40px; background:#1e293b; border-radius:16px;
                box-shadow:0 1px 3px rgba(0,0,0,0.3);'>
        <p style='font-size:3rem; margin:0;'>📄 ➡️ 💼 ➡️ 🎯</p>
        <h3 style='color:#ffffff; margin:16px 0 8px;'>Upload your resume + paste a job description</h3>
        <p style='color:#94a3b8;'>Then click <strong style='color:#e2e8f0;'>Analyze Resume & Generate Report</strong> to get your ATS score,
        skill gap breakdown, and personalized improvement plan.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔍 What this tool analyzes:")
    feat_cols = st.columns(3)
    features = [
        ("🧠", "NLP Parsing", "spaCy NER extracts organizations, dates, and experience from your resume automatically."),
        ("🤖", "LLM Skill Extraction", "Gemini AI identifies technical skills, soft skills, certifications, and domain knowledge."),
        ("📐", "Semantic Similarity", "Sentence-transformers compute embedding-based cosine similarity between resume and JD."),
        ("🎯", "ATS Score", "Get an overall ATS compatibility score (0–100) used by real Applicant Tracking Systems."),
        ("🔎", "Skill Gap Analysis", "See exactly which required and preferred skills are missing from your resume."),
        ("✏️", "Resume Improvements", "AI rewrites your bullet points with action verbs, metrics, and ATS-friendly keywords."),
    ]
    for i, (icon, title, desc) in enumerate(features):
        with feat_cols[i % 3]:
            st.markdown(f"""
            <div style='background:#1e293b; padding:16px; border-radius:12px; margin-bottom:12px;
                        box-shadow:0 1px 3px rgba(0,0,0,0.3); min-height:120px;
                        border: 1px solid #334155;'>
                <p style='font-size:1.8rem; margin:0 0 8px;'>{icon}</p>
                <h4 style='margin:0 0 6px; color:#ffffff;'>{title}</h4>
                <p style='margin:0; color:#94a3b8; font-size:0.85rem;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

