"""
RoleFit - Central Configuration

All tunable parameters in one place. No magic numbers scattered across modules.
Grouped by pipeline stage for clarity.
"""

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "India_runs_data_and_ai_challenge"
OUTPUT_DIR = PROJECT_ROOT / "output"

CANDIDATES_JSONL_PATH = DATA_DIR / "candidates.jsonl"
JD_TEXT_PATH = PROJECT_ROOT / "data" / "jd.txt"
SAMPLE_CANDIDATES_PATH = DATA_DIR / "sample_candidates.json"

# Pre-computed artifacts (created by precompute.py, used by rank.py)
FAISS_INDEX_PATH = OUTPUT_DIR / "candidate_index.faiss"
EMBEDDINGS_META_PATH = OUTPUT_DIR / "embeddings_meta.pkl"

# Final output
SUBMISSION_CSV_PATH = OUTPUT_DIR / "submission.csv"

# =============================================================================
# Stage 1: Bi-Encoder Retrieval
# =============================================================================
BIENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_TOP_K = 200  # Broad initial net — narrowed by cross-encoder

# =============================================================================
# Stage 2: Cross-Encoder Re-Ranking
# =============================================================================
CROSSENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K = 100  # Submission requires exactly 100 ranked candidates

# =============================================================================
# Stage 3: Scoring Weights
# =============================================================================

# Final composite: how much weight each signal gets
CROSS_ENCODER_WEIGHT = 0.30
SKILL_MATCH_WEIGHT = 0.20
CAREER_FIT_WEIGHT = 0.25
BEHAVIORAL_WEIGHT = 0.25

# Behavioral signal sub-weights (within the BEHAVIORAL_WEIGHT bucket)
BEHAVIORAL_WEIGHTS = {
    "recruiter_response_rate": 0.20,
    "interview_completion_rate": 0.15,
    "github_activity_score": 0.15,
    "profile_completeness_score": 0.10,
    "recency_score": 0.10,       # Derived from last_active_date
    "notice_period_score": 0.10,
    "open_to_work_flag": 0.05,
    "verified_identity": 0.05,   # Combined email + phone + linkedin
    "offer_acceptance_rate": 0.05,
    "saved_by_recruiters_30d": 0.05,
}

# =============================================================================
# JD-Specific Disqualifiers (from the actual job description)
# =============================================================================

# Candidates matching these patterns get a heavy penalty
CONSULTING_ONLY_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree", "mphasis", "l&t infotech",
    "persistent systems", "hexaware", "cyient", "ltimindtree",
}

# JD says: 5-9 years preferred range
EXPERIENCE_IDEAL_MIN = 5.0
EXPERIENCE_IDEAL_MAX = 9.0

# JD says: sub-30 day notice preferred
NOTICE_PERIOD_PREFERRED_MAX = 30
NOTICE_PERIOD_ACCEPTABLE_MAX = 60

# JD says: Pune/Noida preferred
PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "delhi ncr",
    "gurgaon", "gurugram", "bengaluru", "bangalore",
}
PREFERRED_COUNTRY = "india"

# =============================================================================
# Honeypot Detection Thresholds
# =============================================================================
# Candidates with impossible profiles (e.g., 8yrs at 3yr-old company,
# "expert" in 10 skills with 0 months usage)
HONEYPOT_MAX_EXPERT_SKILLS_ZERO_DURATION = 3  # Flag if more than this
HONEYPOT_MIN_EXPERIENCE_FOR_EXPERT_COUNT = 2  # Minimum years for many expert skills

# =============================================================================
# Output
# =============================================================================
SUBMISSION_TOP_N = 100  # Exactly 100 candidates in final output
