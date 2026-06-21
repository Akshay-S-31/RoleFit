# RoleFit: AI-Powered Candidate Ranking Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen.svg)
![Tests](https://img.shields.io/badge/Tests-42%2F42_Passed-success.svg)

RoleFit is a high-performance, precision AI recruiting pipeline built for the **Redrob Data & AI Challenge**. It evaluates and ranks candidates against a Job Description (JD) using a multi-stage approach of embeddings, cross-encoder re-ranking, and rule-based behavioral scoring.

The pipeline is capable of searching through **100,000 candidates** and outputting the top 100 ranked candidates in **under 20 seconds** on a standard CPU, well within the 5-minute requirement.

---

## Architecture 

RoleFit uses a strict 4-stage funnel to narrow down the candidate pool precisely and efficiently, avoiding the overhead and hallucinations of LLMs.

### 1. Bi-Encoder Retrieval (FAISS)
The Job Description is embedded using `sentence-transformers/all-MiniLM-L6-v2`. A pre-computed FAISS index rapidly searches the entire candidate pool and retrieves a longlist of the top-200 semantically relevant candidates in `<0.1s`.

### 2. Cross-Encoder Re-Ranking
The top 200 candidate profiles are paired with the JD and fed into a Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). This model evaluates the relevance between the two texts jointly, providing a highly accurate semantic fit score that outperforms standard embedding similarity.

### 3. Rule-Based Scoring & Honeypot Detection
A robust deterministic scoring engine evaluates candidate metadata:
- **Honeypot Detector:** Flags internally inconsistent profiles (e.g., 5+ expert skills with 0 months of experience) and completely removes them from the running.
- **Skill Matcher:** Evaluates proficiency and depth in core NLP/IR skills (e.g., Pinecone, FAISS, LLMs, Embeddings) specifically requested in the JD.
- **Career Fit:** Rewards ideal experience ranges (5-9 years) at relevant product companies, strictly penalizing title-chasers and consulting-only careers.
- **Behavioral Scorer:** Heavily weights recruiter response rates, notice periods, and active GitHub contributions to ensure the top candidates are actually available, reachable, and active.

### 4. Composite Scoring & Reasoning
A composite score combines the signals into a final ranking:
*   **Cross-Encoder Semantic Fit:** `30%`
*   **Career Fit:** `25%`
*   **Behavioral Fit:** `25%`
*   **Skill Match:** `20%`

A deterministic engine then generates a 1-2 sentence explanation of the score for each of the top 100 candidates, satisfying the reasoning requirement without LLM inference overhead.

---

##  Setup & Execution

### Prerequisites
- Python 3.9+
- macOS or Linux environment recommended

### 1. Install Dependencies
```bash
python3 -m pip install -r requirements.txt
```

### 2. Pre-Compute the FAISS Index (One-time setup)
Generate the FAISS vector index from the full candidate pool:
```bash
python3 precompute.py --candidates India_runs_data_and_ai_challenge/candidates.jsonl
```

### 3. Run the Ranking Pipeline
Rank the candidates and output the final submission CSV (Executes in ~18 seconds):
```bash
python3 rank.py --candidates India_runs_data_and_ai_challenge/candidates.jsonl --out output/submission.csv
```

### 4. Validate the Output
Ensure the generated submission perfectly matches the contest specifications:
```bash
python3 India_runs_data_and_ai_challenge/validate_submission.py output/submission.csv
```

---

## Streamlit UI Sandbox

RoleFit includes an interactive web sandbox where you can upload your own `candidates.jsonl` and view the ranking and reasoning via a clean UI.

To launch the sandbox:
```bash
python3 -m streamlit run app.py
```
> **Note:** The sandbox requires the FAISS index to be pre-computed. Ensure the candidates you upload are from the dataset that was indexed.

---

##  Testing

The codebase includes an extensive suite of 42 Pytest unit and integration tests covering text processing, scorers, vector databases, rerankers, and the main pipeline.

To run the full test suite:
```bash
python3 -m pytest tests/
```
