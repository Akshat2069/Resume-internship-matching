# Resume ↔ Job Description Semantic Matching Engine

### AI-powered semantic matching for smarter, fairer, and more inclusive hiring

The **Resume ↔ Job Description Semantic Matching Engine** is an NLP-based application that matches candidate resumes with job descriptions using semantic understanding rather than relying only on exact keyword matching.

Developed by **Team VisionStack**, the system retrieves relevant candidates, reranks them based on contextual relevance, evaluates skill coverage, and provides an explainable ranked shortlist to support recruiters.

## 🚀 Live Demo

- **Live Application:** https://resume-internship-matching-cpacukmycjaqkvpnkk5u4z.streamlit.app/
- **GitHub Repository:** https://github.com/Akshat2069/Resume-internship-matching

## 🎯 Problem Statement

Traditional Applicant Tracking Systems (ATS) often rely heavily on keyword matching. This can lead to:

- Relevant candidates being overlooked because they use different wording.
- Candidates with many matching keywords ranking highly despite weak overall relevance.
- Recruiters spending significant time reviewing large numbers of resumes.
- Limited explanations for why a resume matches a particular job.

## 💡 Proposed Solution

Our system uses a semantic Resume–Job Description matching pipeline to understand the meaning and relevance of candidate information.

### Key Features

- **Multiple document uploads:** Supports resumes and job descriptions in PDF, DOCX, and text-based formats as implemented.
- **Semantic embeddings:** Converts text into numerical vectors to represent meaning.
- **FAISS retrieval:** Retrieves potentially relevant candidates efficiently.
- **Cross-Encoder reranking:** Performs a more detailed comparison between job descriptions and candidate text.
- **Skill and experience matching:** Uses structured matching signals and configured filters.
- **Explainable results:** Displays ranked candidates, matching skills, missing requirements, and supporting evidence.
- **Evaluation metrics:** Includes Precision@K, Recall@K, and NDCG@K evaluation approaches.
- **Fairness audit:** Tests ranking sensitivity by masking selected resume information and comparing rankings.

## ⚙️ System Architecture

The system follows a two-stage retrieval and reranking approach.

1. **Input:** Upload resumes and one or more job descriptions.
2. **Parsing and extraction:** Extract document text and structured information.
3. **Semantic embedding:** Generate 384-dimensional embeddings using `all-MiniLM-L6-v2`.
4. **Stage 1 — Retrieval:** Use FAISS to retrieve the top candidate matches.
5. **Stage 2 — Reranking:** Use a Cross-Encoder model to score resume–job description pairs.
6. **Matching and score fusion:** Combine reranking, filter, and skill-coverage signals.
7. **Explainable output:** Present a ranked shortlist with matching evidence and missing requirements.

### Scoring Formula

The current presentation specifies the following weighted score:

`Final Score = 0.35 × Rerank + 0.25 × Filter + 0.40 × Skill Coverage`

The input scores must be normalized consistently for this formula to be meaningful.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application and matching pipeline |
| Sentence Transformers | Semantic embeddings and Cross-Encoder scoring |
| `all-MiniLM-L6-v2` | Resume and job-description embeddings |
| FAISS | Vector similarity search |
| PyMuPDF | PDF text extraction |
| python-docx | DOCX document processing |
| Gemini API | Language-model-assisted job-description extraction, with local fallback as described in the project |
| NumPy and Pandas | Numerical operations and data processing |
| Streamlit | Interactive web interface and deployment |

## 📊 Evaluation

The project includes evaluation approaches based on:

- **Precision@K:** Measures the proportion of relevant results in the top K.
- **Recall@K:** Measures the proportion of relevant candidates retrieved in the top K.
- **NDCG@K:** Evaluates ranking quality, giving more importance to relevant results near the top.

**Evaluation limitation:** The presentation identifies the current metrics as automatic proxy metrics based on requirement, skill, and filter coverage. They are not human-judged ground-truth evaluation results. A larger manually labelled benchmark is a future improvement.

## ⚖️ Fairness Audit

The fairness audit investigates whether candidate rankings change when selected demographic-correlated information is masked.

The proposed analysis includes:

- Masking selected attributes such as name, gender, or college.
- Comparing candidate rankings before and after masking.
- Measuring ranking changes using Kendall's Tau, Spearman's Rho, and Top-K overlap or difference.
- Investigating whether some groups may be disadvantaged.

Ranking changes are treated as audit signals, not proof of demographic discrimination. More comprehensive testing is needed to establish fairness.

## 💻 Installation and Setup

### Prerequisites

- Python 3.10 or another version supported by the project's dependencies.
- Git.
- A compatible environment for installing the project's Python packages.

### Step 1: Clone the repository

```bash
git clone https://github.com/Akshat2069/Resume-internship-matching.git
cd Resume-internship-matching
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### Step 3: Install dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

If the project uses a different dependency file, follow the instructions provided in the repository.

### Step 4: Configure the Gemini API key

If your local configuration requires Gemini API access, configure the key using an environment variable rather than hardcoding it in source code.

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="your_api_key"
```

macOS or Linux:

```bash
export GEMINI_API_KEY="your_api_key"
```

The project may support local fallback behavior when the Gemini service is unavailable. Check the implementation and configuration before running it.

### Step 5: Run the application

If `app.py` is the Streamlit entry point:

```bash
streamlit run app.py
```

Open the local URL printed in the terminal.

## 📁 Project Structure

The exact repository structure may differ. A typical structure for this application is:

```text
Resume-internship-matching/
├── app.py
├── requirements.txt
├── src/
│   └── matching/
│       ├── embeddings.py
│       ├── retrieval.py
│       ├── reranker.py
│       └── pipeline.py
├── README.md
└── ...
```

## 🔮 Future Improvements

- Build a larger, manually labelled resume–job description benchmark.
- Improve parsing of complex PDFs and unusual document layouts.
- Evaluate alternative embedding models on the same dataset.
- Explore learning-to-rank methods instead of fixed scoring weights.
- Improve scalability, latency, and recruiter workflow integration.
- Strengthen privacy safeguards and fairness evaluation.
