"""
RoleFit - Data Models

Pydantic schemas that exactly mirror candidate_schema.json from the challenge dataset.
These are the contract between every module in the pipeline — if these are wrong,
everything downstream breaks.

Design decisions:
- Fields use Optional where the schema allows null or the field may be absent.
- Enums are validated via Literal types for strictness without separate enum classes.
- RankedCandidate extends the base Candidate with scoring fields added during pipeline stages.
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Candidate Profile (nested objects)
# =============================================================================

class CandidateProfile(BaseModel):
    """Top-level profile info: identity, headline, summary, location, current role."""
    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float = Field(ge=0, le=50)
    current_title: str
    current_company: str
    current_company_size: Literal[
        "1-10", "11-50", "51-200", "201-500", "501-1000",
        "1001-5000", "5001-10000", "10001+"
    ]
    current_industry: str


class CareerEntry(BaseModel):
    """A single role in the candidate's career history."""
    company: str
    title: str
    start_date: str  # ISO date string (YYYY-MM-DD)
    end_date: Optional[str] = None  # null if current role
    duration_months: int = Field(ge=0)
    is_current: bool
    industry: str
    company_size: Literal[
        "1-10", "11-50", "51-200", "201-500", "501-1000",
        "1001-5000", "5001-10000", "10001+"
    ]
    description: str


class Education(BaseModel):
    """A single education entry."""
    institution: str
    degree: str
    field_of_study: str
    start_year: int = Field(ge=1970, le=2030)
    end_year: int = Field(ge=1970, le=2035)
    grade: Optional[str] = None
    tier: Optional[Literal["tier_1", "tier_2", "tier_3", "tier_4", "unknown"]] = None


class Skill(BaseModel):
    """A skill with proficiency level and endorsement count."""
    name: str
    proficiency: Literal["beginner", "intermediate", "advanced", "expert"]
    endorsements: int = Field(ge=0)
    duration_months: Optional[int] = Field(default=None, ge=0)


class Certification(BaseModel):
    """A professional certification."""
    name: str
    issuer: str
    year: int


class Language(BaseModel):
    """A language the candidate speaks."""
    language: str
    proficiency: Literal["basic", "conversational", "professional", "native"]


class SalaryRange(BaseModel):
    """Expected salary in INR Lakhs Per Annum."""
    min: float = Field(ge=0)
    max: float = Field(ge=0)


class RedrobSignals(BaseModel):
    """
    23 behavioral signals from the Redrob platform.
    
    These are often more predictive of hire-ability than static profile data.
    A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5%
    response rate is not actually available.
    """
    profile_completeness_score: float = Field(ge=0, le=100)
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int = Field(ge=0)
    applications_submitted_30d: int = Field(ge=0)
    recruiter_response_rate: float = Field(ge=0, le=1)
    avg_response_time_hours: float = Field(ge=0)
    skill_assessment_scores: Dict[str, float] = Field(default_factory=dict)
    connection_count: int = Field(ge=0)
    endorsements_received: int = Field(ge=0)
    notice_period_days: int = Field(ge=0, le=180)
    expected_salary_range_inr_lpa: SalaryRange
    preferred_work_mode: Literal["remote", "hybrid", "onsite", "flexible"]
    willing_to_relocate: bool
    github_activity_score: float = Field(ge=-1, le=100)
    search_appearance_30d: int = Field(ge=0)
    saved_by_recruiters_30d: int = Field(ge=0)
    interview_completion_rate: float = Field(ge=0, le=1)
    offer_acceptance_rate: float = Field(ge=-1, le=1)
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool


# =============================================================================
# Top-Level Candidate
# =============================================================================

class Candidate(BaseModel):
    """
    A single candidate record — maps 1:1 to a line in candidates.jsonl.
    
    This is the core data structure that flows through the entire pipeline.
    Every module reads from this; nothing modifies it.
    """
    candidate_id: str  # Format: CAND_XXXXXXX
    profile: CandidateProfile
    career_history: List[CareerEntry] = Field(min_length=1, max_length=10)
    education: List[Education] = Field(default_factory=list)
    skills: List[Skill] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
    languages: List[Language] = Field(default_factory=list)
    redrob_signals: RedrobSignals


# =============================================================================
# Job Description
# =============================================================================

class JobDescription(BaseModel):
    """
    Parsed job description with structured sections.
    
    The JD is not just a bag of keywords — it has explicit requirements,
    nice-to-haves, disqualifiers, and behavioral expectations. We parse
    it into sections so downstream modules can reason about each separately.
    """
    title: str
    raw_text: str  # Full JD text for embedding
    core_requirements: str  # "Things you absolutely need"
    nice_to_have: str  # "Things we'd like you to have"
    disqualifiers: str  # "Things we explicitly do NOT want"
    behavioral_expectations: str  # Culture fit, work style
    ideal_candidate: str  # "How to read between the lines" section


# =============================================================================
# Pipeline Output Models
# =============================================================================

class RankedCandidate(BaseModel):
    """
    A candidate with all scoring signals attached.
    
    Created during pipeline execution. The Candidate data is immutable;
    scoring fields are populated stage by stage.
    """
    candidate: Candidate
    
    # Stage 1: Bi-encoder retrieval
    vector_score: float = 0.0
    
    # Stage 2: Cross-encoder re-ranking
    cross_encoder_score: float = 0.0
    
    # Stage 3: Rule-based scoring
    skill_match_score: float = 0.0
    career_fit_score: float = 0.0
    behavioral_score: float = 0.0
    is_honeypot: bool = False
    
    # Final composite
    final_score: float = 0.0
    reasoning: str = ""


class SubmissionRow(BaseModel):
    """
    A single row in the output CSV.
    Maps to: candidate_id, rank, score, reasoning
    """
    candidate_id: str
    rank: int = Field(ge=1, le=100)
    score: float
    reasoning: str
