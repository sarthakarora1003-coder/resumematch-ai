# 🎯 ResumeMatch AI — NLP-Powered ATS Resume Analyzer

An end-to-end NLP application that matches resumes to job descriptions, computes ATS compatibility scores, identifies skill gaps, and generates personalized improvement plans — powered by Google Gemini API.

---

## 🚀 Features

| Feature | Technology |
|---|---|
| PDF/TXT Resume Parsing | `pdfplumber` |
| Named Entity Recognition | `spaCy` (NER) |
| Contact Info & Date Extraction | Regex (NLTK patterns) |
| Semantic Similarity Scoring | `sentence-transformers` (all-MiniLM-L6-v2) |
| Skill Extraction (Structured) | Google Gemini API (Pydantic JSON output) |
| ATS Match Analysis | Google Gemini API |
| Resume Improvement Suggestions | Google Gemini API |
| Interactive Dashboard | Streamlit + Plotly |

---

## 🧠 NLP Pipeline Architecture

```
[ Resume (PDF/TXT) ] + [ Job Description (text) ]
         │
         ▼
1. Traditional NLP Parsing (spaCy NER + Regex)
   ├── Named Entity Extraction (ORG, DATE, PERSON)
   ├── Contact Info (email, phone, URL)
   ├── Section Detection
   └── Experience Year Estimation
         │
         ▼
2. Semantic Similarity (sentence-transformers)
   └── Cosine Similarity on document embeddings (0–1)
         │
         ▼
3. LLM Skill Extraction (Gemini API, Structured JSON)
   ├── Resume: technical skills, soft skills, domain knowledge, certs
   └── JD: required skills, preferred skills, responsibilities, experience
         │
         ▼
4. ATS Match Analysis (Gemini API, Pydantic schema)
   ├── Overall ATS Score (0–100)
   ├── Technical Match Score
   ├── Experience Match Score
   ├── Matched / Missing Skills
   └── Hire Recommendation
         │
         ▼
5. Improvement Generation (Gemini API)
   ├── ATS Keyword Gap List
   ├── Rewritten Bullet Points
   ├── Cover Letter Talking Points
   └── Prioritized Action Plan
         │
         ▼
6. Streamlit Dashboard
   ├── Gauge charts, radar chart
   ├── Color-coded skill chips
   ├── Strengths & Weaknesses cards
   └── Full Improvement Report
```

---

## ⚙️ Setup

### 1. Clone / Navigate to the project
```bash
cd "nlp 2"
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Set your Gemini API key
Get a free key at [aistudio.google.com](https://aistudio.google.com), then:
```bash
# Edit the .env file:
GEMINI_API_KEY=your_key_here
```
Or enter it directly in the app's sidebar.

### 5. Run the Streamlit app
```bash
streamlit run app.py
```

### 6. Run Benchmarks (optional — no API cost)
```bash
python benchmark.py
```

---

## 📁 Project Structure

```
nlp 2/
├── app.py                    # Main Streamlit application
├── benchmark.py              # Evaluation script (7 test cases)
├── requirements.txt          # Python dependencies
├── .env                      # API key config
├── utils/
│   ├── resume_parser.py      # Traditional NLP parsing (spaCy, regex, NLTK)
│   ├── llm_analyzer.py       # Gemini API integration (Pydantic schemas)
│   ├── embeddings.py         # Semantic similarity (sentence-transformers)
│   └── charts.py             # Plotly visualizations
└── sample_data/              # (optional) sample resumes/JDs for testing
```

---

## 📊 Evaluation Metrics

The benchmark script tests:
- **Skill Overlap Scoring**: 4 test cases covering perfect/partial/semantic matches
- **Semantic Similarity**: 3 test cases (high/moderate/low domain overlap)

Run `python benchmark.py` to generate `benchmark_results.json` with accuracy scores.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Google Gemini 1.5 Flash API** — LLM backbone for structured extraction & analysis
- **sentence-transformers** — Local embedding model for semantic similarity
- **spaCy** — Named entity recognition and NLP preprocessing
- **pdfplumber** — PDF text extraction
- **Streamlit** — Interactive web UI
- **Plotly** — Interactive charts and visualizations
- **Pydantic** — Structured output validation for LLM responses

---

## 📝 License

Academic project — for educational use only.
