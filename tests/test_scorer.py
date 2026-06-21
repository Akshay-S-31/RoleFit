"""
Tests for the rule-based scorer.

Verifies each sub-scorer against the specific JD rules:
- SkillMatcher: required vs nice-to-have, depth scoring
- CareerFitScorer: consulting-only, experience range, title-chaser
- BehavioralScorer: response rate, recency, notice period
- HoneypotDetector: impossible profiles
- score_candidate(): integration test
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SAMPLE_CANDIDATES_PATH
from src.data_loader import load_sample_candidates
from src.models import (
    Candidate, CandidateProfile, CareerEntry, Education,
    Language, RankedCandidate, RedrobSignals, SalaryRange, Skill,
)
from src.scorer import (
    BehavioralScorer, CareerFitScorer, HoneypotDetector,
    SkillMatcher, score_candidate,
)


# =============================================================================
# Fixtures
# =============================================================================

def _make_signals(**overrides) -> RedrobSignals:
    """Build a default RedrobSignals with specific fields overridden."""
    defaults = dict(
        profile_completeness_score=80.0,
        signup_date="2024-01-01",
        last_active_date="2026-06-01",
        open_to_work_flag=True,
        profile_views_received_30d=10,
        applications_submitted_30d=2,
        recruiter_response_rate=0.70,
        avg_response_time_hours=24.0,
        skill_assessment_scores={},
        connection_count=200,
        endorsements_received=20,
        notice_period_days=30,
        expected_salary_range_inr_lpa=SalaryRange(min=20.0, max=40.0),
        preferred_work_mode="hybrid",
        willing_to_relocate=True,
        github_activity_score=60.0,
        search_appearance_30d=50,
        saved_by_recruiters_30d=5,
        interview_completion_rate=0.85,
        offer_acceptance_rate=0.8,
        verified_email=True,
        verified_phone=True,
        linkedin_connected=True,
    )
    defaults.update(overrides)
    return RedrobSignals(**defaults)


def _make_career_entry(
    company="ProductCo",
    title="ML Engineer",
    duration_months=36,
    is_current=False,
    industry="Software",
    company_size="201-500",
    start_date="2022-01-01",
    end_date="2025-01-01",
    description="Built ranking and retrieval systems.",
) -> CareerEntry:
    return CareerEntry(
        company=company, title=title, start_date=start_date, end_date=end_date,
        duration_months=duration_months, is_current=is_current,
        industry=industry, company_size=company_size, description=description,
    )


def _make_candidate(
    candidate_id="CAND_0000001",
    title="ML Engineer",
    company="ProductCo",
    years_exp=6.0,
    skills=None,
    career=None,
    signals=None,
) -> Candidate:
    """Build a minimal valid Candidate for testing."""
    if skills is None:
        skills = [
            Skill(name="FAISS", proficiency="advanced", endorsements=10, duration_months=24),
            Skill(name="Embeddings", proficiency="expert", endorsements=15, duration_months=36),
        ]
    if career is None:
        career = [_make_career_entry(company=company, title=title, is_current=True,
                                      end_date=None, duration_months=int(years_exp * 12))]
    if signals is None:
        signals = _make_signals()

    return Candidate(
        candidate_id=candidate_id,
        profile=CandidateProfile(
            anonymized_name="Test Candidate",
            headline=f"{title} | AI Systems",
            summary="Building ranking and retrieval systems.",
            location="Pune",
            country="India",
            years_of_experience=years_exp,
            current_title=title,
            current_company=company,
            current_company_size="201-500",
            current_industry="Software",
        ),
        career_history=career,
        education=[],
        skills=skills,
        certifications=[],
        languages=[Language(language="English", proficiency="professional")],
        redrob_signals=signals,
    )


# =============================================================================
# SkillMatcher Tests
# =============================================================================

class TestSkillMatcher:
    matcher = SkillMatcher()

    def test_required_skills_score_high(self):
        """Candidates with required skills (FAISS, embeddings, NLP) should score well."""
        c = _make_candidate(skills=[
            Skill(name="FAISS", proficiency="expert", endorsements=10, duration_months=30),
            Skill(name="Embeddings", proficiency="advanced", endorsements=8, duration_months=24),
            Skill(name="NLP", proficiency="advanced", endorsements=12, duration_months=36),
            Skill(name="Machine Learning", proficiency="expert", endorsements=20, duration_months=48),
            Skill(name="Qdrant", proficiency="advanced", endorsements=5, duration_months=18),
        ])
        score = self.matcher.score(c)
        assert score >= 0.60, f"Expected high skill score, got {score:.2f}"

    def test_unrelated_skills_score_low(self):
        """Marketing/accounting skills should score near zero."""
        c = _make_candidate(skills=[
            Skill(name="Marketing", proficiency="expert", endorsements=20, duration_months=60),
            Skill(name="Excel", proficiency="advanced", endorsements=15, duration_months=48),
            Skill(name="Accounting", proficiency="expert", endorsements=10, duration_months=36),
        ])
        score = self.matcher.score(c)
        assert score < 0.20, f"Expected low skill score for non-ML, got {score:.2f}"

    def test_cv_speech_penalty(self):
        """Primarily CV/speech skills without NLP should get penalized."""
        c = _make_candidate(skills=[
            Skill(name="Image Classification", proficiency="expert", endorsements=20, duration_months=48),
            Skill(name="Object Detection", proficiency="expert", endorsements=15, duration_months=36),
            Skill(name="Speech Recognition", proficiency="expert", endorsements=12, duration_months=30),
            Skill(name="GANs", proficiency="expert", endorsements=10, duration_months=24),
        ])
        score = self.matcher.score(c)
        # Score should be penalized vs. a candidate with equivalent NLP skills
        nlp_c = _make_candidate(skills=[
            Skill(name="NLP", proficiency="expert", endorsements=20, duration_months=48),
            Skill(name="Embeddings", proficiency="expert", endorsements=15, duration_months=36),
            Skill(name="FAISS", proficiency="expert", endorsements=12, duration_months=30),
            Skill(name="Machine Learning", proficiency="expert", endorsements=10, duration_months=24),
        ])
        nlp_score = self.matcher.score(nlp_c)
        assert score < nlp_score, "CV/speech primary expertise should score lower than NLP/IR"

    def test_proficiency_depth_matters(self):
        """Expert + long duration should score higher than beginner + short."""
        expert_c = _make_candidate(skills=[
            Skill(name="FAISS", proficiency="expert", endorsements=10, duration_months=36),
        ])
        beginner_c = _make_candidate(skills=[
            Skill(name="FAISS", proficiency="beginner", endorsements=1, duration_months=3),
        ])
        assert self.matcher.score(expert_c) > self.matcher.score(beginner_c)

    def test_score_in_range(self):
        """Score must always be in [0, 1]."""
        for _ in range(5):
            c = load_sample_candidates(SAMPLE_CANDIDATES_PATH)[0]
            assert 0.0 <= self.matcher.score(c) <= 1.0


# =============================================================================
# CareerFitScorer Tests
# =============================================================================

class TestCareerFitScorer:
    scorer = CareerFitScorer()

    def test_consulting_only_penalized(self):
        """Consulting-only career should get a very low score."""
        career = [
            _make_career_entry(company="TCS", title="ML Engineer", duration_months=36),
            _make_career_entry(company="Infosys", title="Senior Engineer", duration_months=24),
            _make_career_entry(company="Wipro", title="Lead Engineer", duration_months=12),
        ]
        c = _make_candidate(career=career)
        score, reason = self.scorer.score(c)
        assert score <= 0.15, f"Consulting-only should score low, got {score:.2f}"
        assert "consulting" in reason

    def test_product_company_scores_well(self):
        """Product company experience should score well."""
        career = [
            _make_career_entry(company="CRED", title="ML Engineer", duration_months=36, is_current=True, end_date=None),
        ]
        c = _make_candidate(years_exp=6.0, career=career)
        score, _ = self.scorer.score(c)
        assert score >= 0.60, f"Product company ML engineer should score well, got {score:.2f}"

    def test_ideal_experience_range(self):
        """5-9 years should be the ideal range."""
        ideal = _make_candidate(years_exp=7.0)
        too_junior = _make_candidate(years_exp=1.5)
        ideal_score, _ = self.scorer.score(ideal)
        junior_score, _ = self.scorer.score(too_junior)
        assert ideal_score > junior_score

    def test_title_chaser_penalty(self):
        """Rapid job hopping for title bumps should incur a penalty."""
        # 4 jobs each < 18 months = title-chaser pattern
        career = [
            _make_career_entry(company="Co1", title="Engineer", duration_months=10),
            _make_career_entry(company="Co2", title="Senior Engineer", duration_months=12),
            _make_career_entry(company="Co3", title="Staff Engineer", duration_months=14),
            _make_career_entry(company="Co4", title="ML Engineer", duration_months=16, is_current=True, end_date=None),
        ]
        chaser = _make_candidate(years_exp=5.5, career=career)
        stable_career = [
            _make_career_entry(company="Co1", title="ML Engineer", duration_months=48),
            _make_career_entry(company="Co2", title="Senior ML Engineer", duration_months=36, is_current=True, end_date=None),
        ]
        stable = _make_candidate(years_exp=7.0, career=stable_career)
        chaser_score, _ = self.scorer.score(chaser)
        stable_score, _ = self.scorer.score(stable)
        assert stable_score > chaser_score

    def test_wrong_domain_title_penalized(self):
        """Marketing Manager / Accountant titles should score low."""
        c = _make_candidate(title="Marketing Manager", company="Acme")
        score, _ = self.scorer.score(c)
        assert score <= 0.50, f"Wrong-domain title should score low, got {score:.2f}"


# =============================================================================
# BehavioralScorer Tests
# =============================================================================

class TestBehavioralScorer:
    scorer = BehavioralScorer()

    def test_highly_engaged_scores_high(self):
        """Active, responsive, fast-notice candidate should score high."""
        c = _make_candidate(signals=_make_signals(
            recruiter_response_rate=0.90,
            interview_completion_rate=0.95,
            github_activity_score=80.0,
            last_active_date="2026-06-18",  # Very recent
            notice_period_days=15,
            open_to_work_flag=True,
        ))
        score = self.scorer.score(c)
        assert score >= 0.75, f"Highly engaged candidate should score high, got {score:.2f}"

    def test_inactive_candidate_scores_low(self):
        """6-month inactive, unresponsive candidate should score low."""
        c = _make_candidate(signals=_make_signals(
            recruiter_response_rate=0.05,
            interview_completion_rate=0.10,
            github_activity_score=-1,
            last_active_date="2025-10-01",  # ~8 months ago
            notice_period_days=90,
            open_to_work_flag=False,
        ))
        score = self.scorer.score(c)
        assert score < 0.40, f"Inactive/unresponsive candidate should score low, got {score:.2f}"

    def test_long_notice_period_penalized(self):
        """90+ day notice should score lower than 30-day notice."""
        short = _make_candidate(signals=_make_signals(notice_period_days=30))
        long = _make_candidate(signals=_make_signals(notice_period_days=120))
        assert self.scorer.score(short) > self.scorer.score(long)

    def test_no_github_is_neutral_not_zero(self):
        """Missing GitHub (-1) should be treated as neutral, not disqualifying."""
        no_github = _make_candidate(signals=_make_signals(github_activity_score=-1))
        score = self.scorer.score(no_github)
        assert score > 0.30, "No GitHub should not produce near-zero behavioral score"


# =============================================================================
# HoneypotDetector Tests
# =============================================================================

class TestHoneypotDetector:
    detector = HoneypotDetector()

    def test_clean_profile_not_flagged(self):
        """Normal candidate should not be flagged as honeypot."""
        c = _make_candidate()
        is_hp, _ = self.detector.is_honeypot(c)
        assert not is_hp

    def test_many_expert_zero_duration_flagged(self):
        """Many expert skills with 0 months duration should be flagged."""
        c = _make_candidate(skills=[
            Skill(name=f"Skill{i}", proficiency="expert", endorsements=10, duration_months=0)
            for i in range(6)
        ])
        is_hp, reason = self.detector.is_honeypot(c)
        assert is_hp
        assert "expert skills with 0 months" in reason

    def test_date_inconsistency_flagged(self):
        """Role that ends before it starts should be flagged."""
        career = [
            _make_career_entry(
                start_date="2024-06-01",
                end_date="2023-01-01",  # ends BEFORE it starts
                duration_months=18,
            )
        ]
        c = _make_candidate(career=career)
        is_hp, reason = self.detector.is_honeypot(c)
        assert is_hp
        assert "ends before it starts" in reason


# =============================================================================
# Integration Test: score_candidate()
# =============================================================================

class TestScoreCandidate:
    def test_scores_populated(self):
        """score_candidate() should populate all three score fields."""
        c = _make_candidate()
        rc = RankedCandidate(candidate=c)
        result = score_candidate(rc)
        assert 0.0 <= result.skill_match_score <= 1.0
        assert 0.0 <= result.career_fit_score <= 1.0
        assert 0.0 <= result.behavioral_score <= 1.0
        assert not result.is_honeypot

    def test_honeypot_zeroes_scores(self):
        """Flagged honeypot should have zeroed rule-based scores."""
        c = _make_candidate(skills=[
            Skill(name=f"S{i}", proficiency="expert", endorsements=5, duration_months=0)
            for i in range(6)
        ])
        rc = RankedCandidate(candidate=c)
        result = score_candidate(rc)
        assert result.is_honeypot
        assert result.skill_match_score == 0.0
        assert result.career_fit_score == 0.0
        assert result.behavioral_score == 0.0

    def test_all_sample_candidates_score(self):
        """All 50 sample candidates should score without errors."""
        candidates = load_sample_candidates(SAMPLE_CANDIDATES_PATH)
        for c in candidates:
            rc = RankedCandidate(candidate=c)
            result = score_candidate(rc)
            assert 0.0 <= result.skill_match_score <= 1.0
            assert 0.0 <= result.career_fit_score <= 1.0
            assert 0.0 <= result.behavioral_score <= 1.0
