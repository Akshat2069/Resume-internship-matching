import base64
import os
import re
import tempfile

import streamlit as st

from src.parser import extract_text
from src.jd_extractor import JDExtractor
from src.resume_processor import ResumeProcessor
from src.matching.pipeline import MatchingPipeline
from evaluation.evaluate import evaluate

from fairness.fairness_audit import (
    mask_resume,
    run_fairness_audit
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Resume ↔ JD Semantic Matcher",
    page_icon="🎯",
    layout="wide"
)

@st.cache_resource(show_spinner="Loading ML models (first time only)...")
def load_pipeline():
    return MatchingPipeline()


# ============================================================
# LOCAL JD FALLBACK
# ============================================================

def local_jd_extraction(jd_text: str) -> dict:
    """
    Lightweight local fallback used only when Gemini extraction
    fails because of quota/service availability.

    The normal Gemini JDExtractor remains the first choice.
    """

    text = jd_text.strip()
    lower_text = text.lower()

    # --------------------------------------------------------
    # EXPERIENCE
    # --------------------------------------------------------

    min_years_experience = None

    range_matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)"
        r"\s*(?:\+?\s*)?years?\s+(?:of\s+)?(?:relevant\s+|professional\s+|"
        r"work\s+|software\s+)?experience",
        lower_text
    )

    if range_matches:

        values = []

        for minimum, maximum in range_matches:

            try:
                values.append(float(minimum))
            except ValueError:
                pass

        if values:
            min_years_experience = min(values)

    if min_years_experience is None:

        experience_matches = re.findall(
            r"(?:minimum|at least)\s+(\d+(?:\.\d+)?)"
            r"\s*\+?\s*years?",
            lower_text
        )

        if experience_matches:

            try:
                min_years_experience = float(
                    experience_matches[0]
                )
            except ValueError:
                pass

    if min_years_experience is None:

        experience_matches = re.findall(
            r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+"
            r"(?:of\s+)?(?:relevant\s+|professional\s+|work\s+|software\s+)"
            r"experience",
            lower_text
        )

        if experience_matches:

            try:
                min_years_experience = float(
                    experience_matches[0]
                )
            except ValueError:
                pass

    # --------------------------------------------------------
    # SKILLS
    # --------------------------------------------------------

    known_skills = [
        "python",
        "java",
        "c++",
        "c",
        "javascript",
        "typescript",
        "react",
        "node.js",
        "nodejs",
        "express",
        "sql",
        "mysql",
        "postgresql",
        "mongodb",
        "html",
        "css",
        "tailwind",
        "git",
        "github",
        "linux",
        "unix",
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "gcp",
        "cloud",
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "tensorflow",
        "pytorch",
        "scikit-learn",
        "pandas",
        "numpy",
        "data structures",
        "algorithms",
        "data structures and algorithms",
        "operating systems",
        "computer networks",
        "networking",
        "distributed systems",
        "system design",
        "rest api",
        "rest apis",
        "fastapi",
        "flask",
        "spring boot",
        "kafka",
        "websocket",
        "socket.io",
        "webrtc",
        "faiss",
        "sentence transformers",
        "nlp",
        "natural language processing",
        "statistics",
        "probability",
        "figma",
        "ui/ux",
        "ui design",
        "ux design",
    ]

    required_skills = []

    for skill in known_skills:

        if skill.lower() in lower_text:

            if skill not in required_skills:
                required_skills.append(skill)

    # --------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------

    education_requirement = None

    education_patterns = [
        r"b\.?\s*tech",
        r"b\.?\s*e\.?",
        r"bachelor(?:'s)?\s+degree",
        r"master(?:'s)?\s+degree",
        r"m\.?\s*tech",
        r"m\.?\s*e\.?",
        r"computer science",
        r"engineering degree",
    ]

    education_matches = []

    for pattern in education_patterns:

        if re.search(pattern, lower_text):

            education_matches.append(pattern)

    if education_matches:

        education_requirement = (
            "Degree/Education requirement specified in JD"
        )

    # --------------------------------------------------------
    # CERTIFICATIONS
    # --------------------------------------------------------

    certifications_required = []

    certification_keywords = [
        "aws certified",
        "azure certification",
        "google cloud certification",
        "certified kubernetes",
        "professional certification",
        "certification required",
    ]

    for certification in certification_keywords:

        if certification in lower_text:

            certifications_required.append(
                certification
            )

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    location_policy = "Not Specified"

    location_keywords = [
        "remote",
        "hybrid",
        "work from home",
        "onsite",
        "on-site",
        "in office",
        "office based",
    ]

    for keyword in location_keywords:

        if keyword in lower_text:

            location_policy = keyword.title()

            break

    # --------------------------------------------------------
    # SENIORITY
    # --------------------------------------------------------

    seniority_level = None

    if "intern" in lower_text:

        seniority_level = "Intern"

    elif "entry level" in lower_text:

        seniority_level = "Entry Level"

    elif "junior" in lower_text:

        seniority_level = "Junior"

    elif "senior" in lower_text:

        seniority_level = "Senior"

    elif "lead" in lower_text:

        seniority_level = "Lead"

    # --------------------------------------------------------
    # REQUIREMENTS
    # --------------------------------------------------------

    requirements = []

    requirement_lines = []

    for line in text.splitlines():

        cleaned = line.strip()

        if not cleaned:
            continue

        cleaned_lower = cleaned.lower()

        if (
            cleaned.startswith("-")
            or cleaned.startswith("•")
            or cleaned.startswith("*")
            or "required skills" in cleaned_lower
            or "requirements" in cleaned_lower
            or "qualifications" in cleaned_lower
        ):

            cleaned = re.sub(
                r"^[\-\•\*]\s*",
                "",
                cleaned
            ).strip()

            if (
                len(cleaned) >= 10
                and len(cleaned) <= 300
            ):

                requirement_lines.append(
                    cleaned
                )

    # Remove section headings accidentally captured

    section_words = [
        "requirements",
        "required skills",
        "qualifications",
        "responsibilities",
        "job description",
        "about the role",
        "about us",
        "preferred skills",
    ]

    for line in requirement_lines:

        if line.lower() in section_words:
            continue

        if line not in requirements:

            requirements.append(line)

    # If bullet extraction produced nothing, create
    # requirement entries from detected skills.

    if not requirements:

        for skill in required_skills:

            requirements.append(
                f"Experience with {skill}"
            )

    # Limit requirement count to avoid noisy fallback data.

    requirements = requirements[:30]

    # --------------------------------------------------------
    # FINAL STRUCTURED OBJECT
    # --------------------------------------------------------

    return {
        "job_id": "local_fallback_job",
        "raw_text": text,

        "required_skills": required_skills,

        "preferred_skills": [],

        "min_years_experience": (
            min_years_experience
        ),

        "education_requirement": (
            education_requirement
        ),

        "certifications_required": (
            certifications_required
        ),

        "seniority_level": (
            seniority_level
        ),

        "location_policy": (
            location_policy
        ),

        "requirements": requirements,
    }


def extract_jd_with_fallback(jd_text: str):
    """
    First try the existing Gemini-based JDExtractor.

    If Gemini is unavailable or quota is exhausted,
    use the local fallback.
    """

    try:

        jd_extractor = JDExtractor()

        jd_data = jd_extractor.extract(
            jd_text
        )

        return jd_data, False

    except Exception as error:
        print(f"Gemini extraction failed: {error}")
        jd_data = local_jd_extraction(
            jd_text
        )
        return jd_data, True




# ============================================================
# POSITIVUS HEADER & DESIGN SYSTEM
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #F8FAFC;
}

/* Beautiful Modern Dark Background */
.stApp {
    background-color: #0F172A;
    background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
        radial-gradient(at 0% 100%, rgba(139, 92, 246, 0.15) 0px, transparent 50%),
        radial-gradient(#334155 1px, transparent 1px);
    background-size: 100% 100%, 100% 100%, 100% 100%, 100% 100%, 24px 24px;
    background-attachment: fixed;
}

h1, h2, h3, h4, h5, h6, .st-emotion-cache-10trblm {
    font-family: 'Outfit', sans-serif !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* Hide default streamlit header/footer */
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Modern Beautiful Dark Button Styling */
.stButton > button {
    border-radius: 999px !important;
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%) !important;
    color: #F8FAFC !important;
    border: 1px solid #334155 !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-family: 'Outfit', sans-serif !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stButton > button:hover {
    background: #1E293B !important;
    color: #A78BFA !important;
    border: 1px solid #6366F1 !important;
    box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.15), 0 4px 6px -2px rgba(99, 102, 241, 0.1) !important;
    transform: translateY(-2px) !important;
}

/* Primary Button Override */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.39) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
    transform: translateY(-3px) !important;
}

/* Dark Glassmorphism Cards */
div[data-testid="stMetric"], .stExpander, div[data-testid="stFileUploader"] > div {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    padding: 1.5rem;
    transition: all 0.3s ease;
}
div[data-testid="stMetric"]:hover, .stExpander:hover, div[data-testid="stFileUploader"] > div:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Upload zone override */
div[data-testid="stFileUploader"] > div > section {
    background-color: transparent !important;
}
div[data-testid="stFileUploader"] > div {
    border: 2px dashed #475569 !important;
    background: rgba(30, 41, 59, 0.4) !important;
}
div[data-testid="stFileUploader"] > div:hover {
    border: 2px dashed #8B5CF6 !important;
    background: rgba(139, 92, 246, 0.1) !important;
}

/* Section Headers & Badges */
.positivus-section-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.positivus-badge {
    background: linear-gradient(135deg, #1E1B4B 0%, #2E1065 100%);
    color: #A78BFA;
    padding: 8px 20px;
    border-radius: 999px;
    border: 1px solid #4C1D95;
    font-family: 'Outfit', sans-serif;
    font-size: 28px;
    font-weight: 700;
    display: inline-block;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
}

/* Input Fields */
.stTextInput > div > div > input, .stSelectbox > div > div {
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
    background-color: #1E293B !important;
    color: #F8FAFC !important;
    transition: all 0.2s ease !important;
}
.stTextInput > div > div > input:focus, .stSelectbox > div > div:focus {
    border-color: #8B5CF6 !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.3) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="positivus-section-header" style="margin-top: 2rem;">
    <span class="positivus-badge" style="font-size: 40px; padding: 4px 16px;">🎯 Resume ↔ JD Semantic Matcher</span>
</div>
<p style="font-size: 20px; font-weight: 500; margin-bottom: 3rem;">
    Retrieve, rerank, filter, and explain candidate matches against multiple Job Descriptions.
</p>
""", unsafe_allow_html=True)

# ============================================================
# UPLOAD SECTION
# ============================================================

col1, col2 = st.columns(2)

with col1:
    jd_files = st.file_uploader(
        "📄 Upload 1–3 Job Descriptions",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="jd_upload"
    )

with col2:
    resume_files = st.file_uploader(
        "👥 Upload candidate resumes",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="resume_upload"
    )


# ============================================================
# UPLOAD STATUS
# ============================================================

if jd_files:

    if len(jd_files) > 3:

        st.error(
            "Please upload a maximum of 3 Job Descriptions."
        )

        st.stop()

    st.success(
        f"{len(jd_files)} Job Description(s) uploaded."
    )


if resume_files:

    st.success(
        f"{len(resume_files)} candidate resume(s) uploaded."
    )


# ============================================================
# START MATCHING
# ============================================================

if jd_files and resume_files:

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Job Descriptions",
            len(jd_files)
        )

    with col2:

        st.metric(
            "Candidates",
            len(resume_files)
        )

    with col3:

        st.metric(
            "Pipeline",
            "Retrieve → Rerank → Filter"
        )

    start_matching = st.button(
        "🚀 Start Matching",
        type="primary",
        use_container_width=True
    )

    if start_matching:

        # ====================================================
        # SAVE AND PROCESS RESUMES
        # ====================================================

        resume_paths = []

        original_filenames = []

        original_file_bytes = {}

        for resume_file in resume_files:

            suffix = os.path.splitext(
                resume_file.name
            )[1]

            file_bytes = (
                resume_file.getvalue()
            )

            temp_resume = (
                tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                )
            )

            temp_resume.write(
                file_bytes
            )

            temp_resume.close()

            resume_paths.append(
                temp_resume.name
            )

            original_filenames.append(
                resume_file.name
            )

            original_file_bytes[
                resume_file.name
            ] = file_bytes

        # ====================================================
        # PROCESS RESUMES
        # ====================================================

        with st.spinner(
            f"Processing {len(resume_paths)} resumes..."
        ):

            processor = ResumeProcessor()

            resumes = processor.process_files(
                resume_paths
            )

        # Restore original filenames

        for resume, original_filename in zip(
            resumes,
            original_filenames
        ):

            resume["filename"] = (
                original_filename
            )

        successful_resumes = [
            resume
            for resume in resumes
            if "error" not in resume
        ]

        failed_resumes = [
            resume
            for resume in resumes
            if "error" in resume
        ]

        if not successful_resumes:

            st.error(
                "No resumes could be processed."
            )

            st.stop()

        # ====================================================
        # MATCH EACH JD
        # ====================================================

        all_jd_results = []

        # Cache the pipeline globally so ML models are not reloaded on every run
        pipeline = load_pipeline()

        for jd_number, jd_file in enumerate(
            jd_files,
            start=1
        ):

            st.write(
                f"### Processing JD {jd_number}: "
                f"{jd_file.name}"
            )

            # ------------------------------------------------
            # SAVE JD TEMPORARILY
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(
                    jd_file.name
                )[1]
            ) as temp_jd:

                temp_jd.write(
                    jd_file.getbuffer()
                )

                jd_path = temp_jd.name

            # ------------------------------------------------
            # EXTRACT JD TEXT
            # ------------------------------------------------

            with st.spinner(
                f"Reading Job Description {jd_number}..."
            ):

                jd_text = extract_text(
                    jd_path
                )

            # ------------------------------------------------
            # GEMINI + FALLBACK EXTRACTION
            # ------------------------------------------------

            with st.spinner(
                f"Extracting requirements from JD {jd_number}..."
            ):

                jd_data, used_fallback = (
                    extract_jd_with_fallback(
                        jd_text
                    )
                )

            if used_fallback:

                st.warning(
                    f"Gemini quota/service was unavailable for "
                    f"JD {jd_number}. "
                    f"Using local JD extraction so matching can continue."
                )

            # ------------------------------------------------
            # EXISTING MATCHING PIPELINE
            # ------------------------------------------------

            with st.spinner(
                f"Matching JD {jd_number} against "
                f"{len(successful_resumes)} resumes..."
            ):

                results = pipeline.match(
                    jd_text=jd_text,
                    jd_data=jd_data,
                    resumes=successful_resumes,
                    retrieval_top_k=len(
                        successful_resumes
                    )
                )

            # ------------------------------------------------
            # STORE RESULTS
            # ------------------------------------------------

            all_jd_results.append(
                {
                    "jd_number": jd_number,
                    "filename": jd_file.name,
                    "jd_text": jd_text,
                    "jd_data": jd_data,
                    "results": results
                }
            )

        # ====================================================
        # STORE RESULTS
        # ====================================================

        st.session_state[
            "all_jd_results"
        ] = all_jd_results

        st.session_state[
            "successful_resumes"
        ] = successful_resumes

        st.session_state[
            "original_file_bytes"
        ] = original_file_bytes

        st.session_state[
            "failed_resumes"
        ] = failed_resumes

        st.success(
            f"Matching completed for "
            f"{len(jd_files)} Job Description(s) "
            f"and {len(successful_resumes)} candidate(s)."
        )


# ============================================================
# RESULTS
# ============================================================

if "all_jd_results" in st.session_state:

    all_jd_results = st.session_state[
        "all_jd_results"
    ]

    original_file_bytes = (
        st.session_state.get(
            "original_file_bytes",
            {}
        )
    )

    failed_resumes = (
        st.session_state.get(
            "failed_resumes",
            []
        )
    )

    st.divider()

    st.markdown("""
    <div class="positivus-section-header" style="margin-top: 4rem;">
        <span class="positivus-badge">Candidate Ranking</span>
        <p style="margin: 0; font-size: 20px;">Ranked Candidate Shortlists</p>
    </div>
    """, unsafe_allow_html=True)

    if failed_resumes:

        st.warning(
            f"{len(failed_resumes)} resume(s) "
            "could not be processed."
        )

    # ========================================================
    # ONE SECTION PER JD
    # ========================================================

    for jd_result in all_jd_results:

        jd_number = jd_result[
            "jd_number"
        ]

        jd_filename = jd_result[
            "filename"
        ]

        results = jd_result[
            "results"
        ]

        st.divider()

        st.header(
            f"📌 JD {jd_number}: {jd_filename}"
        )

        st.caption(
            f"{len(results)} candidate(s) ranked for this Job Description."
        )

        # ====================================================
        # TOP-N DISPLAY
        # ====================================================

        top_n = st.selectbox(
            "Number of top candidates to show",
            options=[5, 6, 10, 15, 20],
            index=0,
            key=f"top_n_{jd_number}"
        )

        displayed_results = results[:top_n]

        st.caption(
            f"Showing top {min(top_n, len(results))} candidate(s) based on final score."
        )

        # ====================================================
        # AUTOMATIC PROXY EVALUATION
        # ====================================================
        # No manual relevance labels are required.
        #
        # A proxy relevance score is generated from signals
        # already produced by the matching pipeline:
        #
        # - requirement coverage
        # - skill coverage
        # - structured filter score
        #
        # These metrics are proxy evaluation metrics and are
        # not human-labeled ground-truth evaluation.

        ranked_resume_ids = [
            result["resume_id"]
            for result in results
        ]

        proxy_relevance_map = {}

        for result in results:

            skill_coverage = (
                float(
                    result.get(
                        "skill_coverage",
                        0.0
                    )
                ) / 100.0
            )

            matched_requirements = result.get(
                "matched_requirements",
                []
            )

            unmatched_requirements = result.get(
                "unmatched_requirements",
                []
            )

            total_requirements = (
                len(matched_requirements)
                + len(unmatched_requirements)
            )

            if total_requirements > 0:

                requirement_coverage = (
                    len(matched_requirements)
                    / total_requirements
                )

            else:

                requirement_coverage = (
                    skill_coverage
                )

            filter_score = float(
                result.get(
                    "filter_score",
                    0.0
                )
            )

            # IMPORTANT:
            # This proxy does not directly use final_score.
            # It uses underlying matching signals instead.

            proxy_score = (
                0.45 * requirement_coverage
                + 0.35 * skill_coverage
                + 0.20 * filter_score
            )

            if proxy_score >= 0.75:

                relevance = 3

            elif proxy_score >= 0.50:

                relevance = 2

            elif proxy_score >= 0.25:

                relevance = 1

            else:

                relevance = 0

            proxy_relevance_map[
                result["resume_id"]
            ] = relevance

        try:

            evaluation_results = evaluate(
                ranked_ids=ranked_resume_ids,
                relevance_map=proxy_relevance_map,
                k=top_n
            )

            st.subheader(
                f"📈 Automatic Evaluation Metrics @ {top_n}"
            )

            metric_col1, metric_col2, metric_col3 = (
                st.columns(3)
            )

            with metric_col1:

                st.metric(
                    f"Precision@{top_n}",
                    f"{evaluation_results[f'Precision@{top_n}'] * 100:.1f}%"
                )

            with metric_col2:

                st.metric(
                    f"Recall@{top_n}",
                    f"{evaluation_results[f'Recall@{top_n}'] * 100:.1f}%"
                )

            with metric_col3:

                st.metric(
                    f"NDCG@{top_n}",
                    f"{evaluation_results[f'NDCG@{top_n}'] * 100:.1f}%"
                )

            st.caption(
                "Automatic proxy evaluation based on requirement coverage, "
                "skill coverage, and structured filter signals. "
                "These are proxy metrics, not human-labeled ground-truth metrics."
            )

        except Exception as evaluation_error:

            st.warning(
                f"Evaluation metrics could not be calculated: "
                f"{evaluation_error}"
            )

        # ====================================================
        # FAIRNESS AUDIT
        # ====================================================

        st.markdown("""
        <div class="positivus-section-header" style="margin-top: 4rem;">
            <span class="positivus-badge">Fairness Audit</span>
            <p style="margin: 0; font-size: 20px;">Measure ranking sensitivity by masking demographics</p>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "Masks demographic-correlated information from the actual "
            "uploaded resumes and reruns the same matching pipeline. "
            "The audit compares ranking sensitivity and does not modify "
            "the original candidate scores or ranking."
        )

        successful_resumes = st.session_state.get(
            "successful_resumes",
            []
        )

        if successful_resumes:

            run_fairness = st.button(
                "🔍 Run Fairness Audit",
                key=f"fairness_{jd_number}"
            )

            if run_fairness:

                with st.spinner(
                    "Running masked fairness audit..."
                ):

                    masked_resumes = []

                    for resume in successful_resumes:

                        masked_resume = resume.copy()

                        masked_resume["raw_text"] = mask_resume(
                            resume.get("raw_text", "")
                        )

                        if "structured_data" in resume:

                            masked_resume["structured_data"] = (
                                resume["structured_data"].copy()
                            )

                        masked_resumes.append(
                            masked_resume
                        )

                    fairness_pipeline = MatchingPipeline()

                    masked_results = fairness_pipeline.match(
                        jd_text=jd_result["jd_text"],
                        jd_data=jd_result["jd_data"],
                        resumes=masked_resumes,
                        retrieval_top_k=len(masked_resumes)
                    )

                    original_ranking = [
                        result["resume_id"]
                        for result in results
                    ]

                    masked_ranking = [
                        result["resume_id"]
                        for result in masked_results
                    ]

                    fairness_report = run_fairness_audit(
                        original_ranking=original_ranking,
                        masked_ranking=masked_ranking,
                        k=min(
                            top_n,
                            len(original_ranking)
                        )
                    )

                st.success(
                    "Fairness audit completed using the actual uploaded resumes."
                )

                # Map internal temporary resume IDs back to the
                # original uploaded resume filenames for display.
                resume_name_map = {
                    resume.get("resume_id"): resume.get(
                        "filename",
                        resume.get(
                            "resume_id",
                            "Unknown Resume"
                        )
                    )
                    for resume in successful_resumes
                }

                fairness_col1, fairness_col2, fairness_col3 = (
                    st.columns(3)
                )

                with fairness_col1:

                    st.metric(
                        "Kendall's Tau",
                        f"{fairness_report['kendall_tau']:.3f}"
                    )

                with fairness_col2:

                    st.metric(
                        "Spearman's Rho",
                        f"{fairness_report['spearman_rho']:.3f}"
                    )

                with fairness_col3:

                    selection_difference = (
                        fairness_report[
                            "top_k_selection"
                        ][
                            "selection_difference"
                        ]
                    )

                    st.metric(
                        "Top-K Selection Difference",
                        f"{selection_difference * 100:.1f}%"
                    )

                top_k_data = fairness_report[
                    "top_k_selection"
                ]

                original_top_k = top_k_data[
                    "original_top_k"
                ]

                masked_top_k = top_k_data[
                    "masked_top_k"
                ]

                overlap_count = top_k_data[
                    "overlap_count"
                ]

                st.write(
                    f"**Top-{min(top_n, len(original_ranking))} "
                    f"overlap:** {overlap_count} candidate(s)"
                )

                with st.expander(
                    "View Original vs Masked Ranking"
                ):

                    ranking_col1, ranking_col2 = (
                        st.columns(2)
                    )

                    with ranking_col1:

                        st.write(
                            "**Original Ranking**"
                        )

                        for position, resume_id in enumerate(
                            original_top_k,
                            start=1
                        ):

                            display_name = resume_name_map.get(
                                resume_id,
                                resume_id
                            )

                            st.write(
                                f"{position}. {display_name}"
                            )

                    with ranking_col2:

                        st.write(
                            "**Masked Ranking**"
                        )

                        for position, resume_id in enumerate(
                            masked_top_k,
                            start=1
                        ):

                            display_name = resume_name_map.get(
                                resume_id,
                                resume_id
                            )

                            st.write(
                                f"{position}. {display_name}"
                            )

                st.info(
                    "Fairness audit only — masking is used to measure "
                    "ranking sensitivity. The original ranking and "
                    "candidate scores are not modified."
                )

        else:

            st.warning(
                "Fairness audit could not run because the processed "
                "resume data is unavailable."
            )

        # ====================================================
        # RANKING TABLE
        # ====================================================

        st.subheader(
            "Candidate Ranking"
        )

        header = st.columns(
            [0.5, 2.2, 1.3, 1.5, 1.2, 1.3]
        )

        with header[0]:

            st.write("**Rank**")

        with header[1]:

            st.write("**Candidate**")

        with header[2]:

            st.write("**Retrieval Score**")

        with header[3]:

            st.write("**Re-ranking Score**")

        with header[4]:

            st.write("**Filter Score**")

        with header[5]:

            st.write("**Final Score**")

        for result in displayed_results:

            row = st.columns(
                [0.5, 2.2, 1.3, 1.5, 1.2, 1.3]
            )

            with row[0]:

                st.write(
                    f"**{result['rank']}**"
                )

            with row[1]:

                st.write(
                    f"**{result['filename']}**"
                )

            with row[2]:

                st.write(
                    f"{result['retrieval_score']:.3f}"
                )

            with row[3]:

                st.write(
                    f"{result['cross_encoder_score'] * 100:.1f}%"
                )

            with row[4]:

                st.write(
                    f"{result['filter_score'] * 100:.1f}%"
                )

            with row[5]:

                st.write(
                    f"**{result['final_score']:.1f}%**"
                )

        # ====================================================
        # CANDIDATE ANALYSIS
        # ====================================================

        st.subheader(
            "👤 Candidate Analysis"
        )

        for result in displayed_results:

            candidate_title = (
                f"#{result['rank']}  |  "
                f"{result['filename']}  |  "
                f"{result['final_score']:.1f}%"
            )

            with st.expander(
                candidate_title
            ):

                st.subheader(
                    "Match Signals"
                )

                c1, c2, c3, c4 = st.columns(4)

                with c1:

                    st.metric(
                        "Final Score",
                        f"{result['final_score']:.1f}%"
                    )

                with c2:

                    st.metric(
                        "Retrieval Score",
                        f"{result['retrieval_score']:.3f}"
                    )

                with c3:

                    st.metric(
                        "Re-ranking Score",
                        f"{result['cross_encoder_score'] * 100:.1f}%"
                    )

                with c4:

                    st.metric(
                        "Filter Score",
                        f"{result['filter_score'] * 100:.1f}%"
                    )

                # ============================================
                # SKILL / EXPERIENCE
                # ============================================

                st.subheader(
                    "Candidate Fit"
                )

                c1, c2 = st.columns(2)

                with c1:

                    st.metric(
                        "Skill Coverage",
                        f"{result['skill_coverage']:.1f}%"
                    )

                with c2:

                    experience = result[
                        "experience_match"
                    ]

                    st.metric(
                        "Experience",
                        experience["status"]
                    )

                # ============================================
                # STRUCTURED FILTERS
                # ============================================

                filter_details = result[
                    "filter_details"
                ]

                experience_filter = (
                    filter_details.get(
                        "experience",
                        {}
                    )
                )

                certification_filter = (
                    filter_details.get(
                        "certifications",
                        {}
                    )
                )

                required_years = (
                    experience_filter.get(
                        "required_years"
                    )
                )

                candidate_years = (
                    experience_filter.get(
                        "candidate_years"
                    )
                )

                if required_years is not None:

                    st.subheader(
                        "Experience"
                    )

                    st.write(
                        f"Required: {required_years} years"
                    )

                    if candidate_years is not None:

                        st.write(
                            f"Candidate: {candidate_years} years"
                        )

                    st.write(
                        f"Status: "
                        f"{experience_filter.get('status')}"
                    )

                required_certs = (
                    certification_filter.get(
                        "required"
                    )
                )

                candidate_certs = (
                    certification_filter.get(
                        "candidate"
                    )
                )

                if required_certs:

                    st.subheader(
                        "Certifications"
                    )

                    st.write(
                        f"Required: {required_certs}"
                    )

                    if candidate_certs:

                        st.write(
                            f"Candidate: {candidate_certs}"
                        )

                # ============================================
                # REQUIREMENT EVIDENCE
                # ============================================

                matched_requirements = (
                    result[
                        "matched_requirements"
                    ]
                )

                unmatched_requirements = (
                    result[
                        "unmatched_requirements"
                    ]
                )

                if matched_requirements:

                    st.subheader(
                        "✅ Requirement Evidence"
                    )

                    for item in matched_requirements:

                        st.write(
                            f"**{item['requirement']}**"
                        )

                        st.caption(
                            f"Similarity: "
                            f"{item['similarity']:.3f}"
                        )

                        st.info(
                            f"Evidence: "
                            f"{item['evidence']}"
                        )

                if unmatched_requirements:

                    st.subheader(
                        "❌ Missing Requirements"
                    )

                    for item in unmatched_requirements:

                        st.write(
                            f"✗ {item['requirement']}"
                        )

                # ============================================
                # SKILLS
                # ============================================

                matched_skills = (
                    result[
                        "matched_skills"
                    ]
                )

                unmatched_skills = (
                    result[
                        "unmatched_skills"
                    ]
                )

                if matched_skills or unmatched_skills:

                    st.subheader(
                        "🛠 Skills"
                    )

                    skill_col1, skill_col2 = (
                        st.columns(2)
                    )

                    with skill_col1:

                        st.write(
                            "**Matching Skills**"
                        )

                        if matched_skills:

                            for skill in matched_skills:

                                st.write(
                                    f"✓ {skill['skill']}"
                                )

                        else:

                            st.write(
                                "None"
                            )

                    with skill_col2:

                        st.write(
                            "**Missing Skills**"
                        )

                        if unmatched_skills:

                            for skill in unmatched_skills:

                                st.write(
                                    f"✗ {skill['skill']}"
                                )

                        else:

                            st.write(
                                "None"
                            )

                # ============================================
                # EXPLANATION
                # ============================================

                explanation = result.get(
                    "explanation"
                )

                if explanation:

                    st.subheader(
                        "🧠 Evidence-Based Explanation"
                    )

                    st.info(
                        explanation
                    )

                # ============================================
                # ORIGINAL RESUME
                # ============================================

                st.subheader(
                    "📄 Original Resume"
                )

                filename = result[
                    "filename"
                ]

                file_bytes = (
                    original_file_bytes.get(
                        filename
                    )
                )

                if file_bytes:

                    extension = (
                        os.path.splitext(
                            filename
                        )[1].lower()
                    )

                    st.download_button(
                        label="📥 Download Original Resume",
                        data=file_bytes,
                        file_name=filename,
                        mime=(
                            "application/pdf"
                            if extension == ".pdf"
                            else None
                        ),
                        key=(
                            f"download_"
                            f"{jd_number}_"
                            f"{filename}"
                        )
                    )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.info(
        "Upload 1–3 Job Descriptions and one or more "
        "candidate resumes to begin."
    )
