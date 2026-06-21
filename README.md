# RoleFit: AI-Powered Candidate Ranking Pipeline

RoleFit is a high-performance, precision AI recruiting pipeline built for the Redrob Hackathon. It evaluates and ranks candidates against a Job Description (JD) using a multi-stage approach of embeddings, cross-encoder re-ranking, and rule-based behavioral scoring.

The pipeline is capable of searching through 100,000 candidates and outputting the top 100 ranked candidates in under **20 seconds** on a standard CPU, well within the 5-minute requirement.

---

## 🎯 Architecture & Methodology

RoleFit uses a 4-stage funnel to narrow down the candidate pool precisely and efficiently:

1. **Bi-Encoder Retrieval (FAISS):** The JD is embedded using `all-MiniLM-L6-v2`. A pre-computed FAISS index rapidly retrieves a longlist of the top-200 semantically relevant candidates.
2. **Cross-Encoder Re-Ranking:** The top 200 candidate profiles are paired with the JD and fed into a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) which evaluates the relevance between the two texts jointly, providing a highly accurate semantic fit score.
3. **Rule-Based Scoring & Honeypot Detection:**
   - **Honeypot Detector:** Flags internally inconsistent profiles (e.g. 5+ expert skills with 0 months of experience) and removes them from the running.
   - **Skill Matcher:** Evaluates proficiency and depth in core NLP/IR skills (e.g., Pinecone, FAISS, LLMs, Embeddings).
   - **Career Fit:** Rewards ideal experience ranges (5-9 years) at relevant companies, penalizing title-chasers and consulting-only careers.
   - **Behavioral Scorer:** Heavily weights recruiter response rates, notice periods, and active GitHub contributions to ensure the top candidates are actually available and reachable.
4. **Composite Scoring & Reasoning:** A composite score combines Cross-Encoder (30%), Career (25%), Behavior (25%), and Skills (20%). A deterministic engine generates a 1-2 sentence explanation of the score for each of the top 100 candidates.

---

## 🚀 Setup & Execution

### 1. Install Dependencies
```bash
python3 -m pip install -r requirements.txt
```

### 2. Pre-Compute the Index (One-time)
Generate the FAISS index from the full candidate pool:
```bash
python3 precompute.py --candidates India_runs_data_and_ai_challenge/candidates.jsonl
```

### 3. Run the Ranking Pipeline
Rank the candidates and output the final submission CSV:
```bash
python3 rank.py --candidates India_runs_data_and_ai_challenge/candidates.jsonl --out output/submission.csv
```

### 4. Validate the Output
Ensure the submission perfectly matches the contest specifications:
```bash
python3 India_runs_data_and_ai_challenge/validate_submission.py output/submission.csv
```

---

## 🎨 Streamlit UI Sandbox

RoleFit includes a web sandbox where you can upload your own `candidates.jsonl` and view the ranking and reasoning interactively.

To launch the sandbox:
```bash
python3 -m streamlit run app.py
```

*Note: The sandbox requires the FAISS index to be pre-computed. Ensure the candidates you upload are from the dataset that was indexed.*

---

## 🧪 Testing

The codebase includes an extensive suite of 42 Pytest unit and integration tests covering the text processing, scorers, vector databases, rerankers, and pipeline.

```bash
python3 -m pytest tests/
```
