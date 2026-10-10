---
marp: true
theme: default
class: lead
style: |
  section {
    background-color: #0F172A;
    background-image: radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%), radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.15) 0px, transparent 50%);
    color: #F8FAFC;
    font-family: 'Inter', sans-serif;
  }
  h1, h2, h3 {
    color: #FFFFFF;
    font-family: 'Outfit', sans-serif;
  }
  strong { color: #A78BFA; }
---

# Semantic Resume Matcher
## AI-Powered Candidate Ranking Pipeline

---

# Core Workflow Overview

1. **Document Ingestion**: Upload Job Descriptions (JD) and Candidate Resumes (PDF).
2. **Text Parsing**: PDF text is extracted via OCR/PDF parsers.
3. **Information Extraction**: LLM processes unstructured text into structured JSON.
4. **Semantic Matching**: High-dimensional embeddings evaluate similarity.
5. **Reranking & Filtering**: Pipeline retrieves top matches and filters out unqualified candidates.

---

# Information Extraction Model

**Model Used**: `gemini-3.8-flash` (Google GenAI)

- **Job Description Extraction**:
  - Automatically identifies: Required Skills, Preferred Skills, Minimum Experience, Education, and Location Policy.
- **Resume Extraction**:
  - Synthesizes candidate history into discrete skills, years of experience, and educational background.
- **Resilience**: Features an offline fallback mechanism using SpaCy/Regex if API limits are reached.

---

# Semantic Matching Pipeline

**Model Used**: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
**Vector Store**: FAISS (Facebook AI Similarity Search)

### How it works:
1. **Embedding**: Candidate skills are converted into dense vector representations.
2. **Indexing**: FAISS creates an optimized Cosine similarity index.
3. **Retrieval**: For each JD requirement, the system fetches the nearest neighbor candidate profiles in milliseconds.

---

# Fairness Audit & Explainability

- **Explainable AI**: Generates a natural language explanation detailing *why* a candidate matched (e.g., "Candidate has 5 years of Python, matching JD requirement of 3 years").
- **Fairness Audit**: Evaluates the shortlist to ensure there is no inherent bias based on non-critical demographic or stylistic factors.
- **Score Breakdown**: Provides individual scores for Requirements, Experience, and Skills.

---

# Tech Stack

- **Frontend**: Streamlit with custom Dark Glassmorphism CSS.
- **Backend Logic**: Python 3
- **Generative AI**: Google Gemini 3.8 Flash
- **Embeddings**: HuggingFace SentenceTransformers
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **Deployment**: Local / Cloud ready
