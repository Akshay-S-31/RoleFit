"""
Tests for data loading, model validation, and text building.

These verify:
1. Pydantic models correctly parse the real dataset format
2. JSONL streaming works without crashing
3. JD parser extracts expected sections
4. Text builder produces meaningful output
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, JD_TEXT_PATH, SAMPLE_CANDIDATES_PATH, CANDIDATES_JSONL_PATH
from src.models import Candidate, JobDescription
from src.data_loader import (
    load_candidates_streaming,
    load_sample_candidates,
    load_job_description,
)
from src.text_builder import build_candidate_text


# =============================================================================
# Model Validation Tests
# =============================================================================

class TestModels:
    """Verify Pydantic models parse real data correctly."""
    
    def test_single_candidate_from_sample(self):
        """Parse the first candidate from sample_candidates.json."""
        with open(SAMPLE_CANDIDATES_PATH) as f:
            raw = json.load(f)
        
        candidate = Candidate.model_validate(raw[0])
        assert candidate.candidate_id == "CAND_0000001"
        assert candidate.profile.anonymized_name == "Ira Vora"
        assert candidate.profile.years_of_experience == 6.9
        assert len(candidate.career_history) >= 1
        assert len(candidate.skills) >= 1
        assert candidate.redrob_signals.profile_completeness_score > 0
    
    def test_all_sample_candidates_parse(self):
        """All 50 sample candidates should parse without error."""
        with open(SAMPLE_CANDIDATES_PATH) as f:
            raw_list = json.load(f)
        
        for i, raw in enumerate(raw_list):
            try:
                Candidate.model_validate(raw)
            except Exception as e:
                pytest.fail(f"Sample candidate {i} (id={raw.get('candidate_id')}) failed: {e}")
    
    def test_candidate_has_redrob_signals(self):
        """Every candidate must have behavioral signals."""
        with open(SAMPLE_CANDIDATES_PATH) as f:
            raw = json.load(f)
        
        candidate = Candidate.model_validate(raw[0])
        signals = candidate.redrob_signals
        
        # Check key signal fields exist and are valid
        assert 0 <= signals.profile_completeness_score <= 100
        assert 0 <= signals.recruiter_response_rate <= 1
        assert -1 <= signals.github_activity_score <= 100
        assert signals.notice_period_days >= 0


# =============================================================================
# Data Loader Tests
# =============================================================================

class TestDataLoader:
    """Verify data loading functions work with real files."""
    
    def test_load_sample_candidates(self):
        """Load the sample JSON file."""
        candidates = load_sample_candidates(SAMPLE_CANDIDATES_PATH)
        assert len(candidates) == 50  # sample has 50 candidates
        assert all(isinstance(c, Candidate) for c in candidates)
    
    def test_streaming_first_10(self):
        """Stream the first 10 candidates from the full JSONL."""
        if not CANDIDATES_JSONL_PATH.exists():
            pytest.skip("Full candidates.jsonl not available")
        
        count = 0
        for candidate in load_candidates_streaming(CANDIDATES_JSONL_PATH):
            assert isinstance(candidate, Candidate)
            assert candidate.candidate_id.startswith("CAND_")
            count += 1
            if count >= 10:
                break
        
        assert count == 10
    
    def test_load_job_description(self):
        """JD loads and has all sections populated."""
        if not JD_TEXT_PATH.exists():
            pytest.skip("JD text file not yet extracted")
        
        jd = load_job_description(JD_TEXT_PATH)
        assert isinstance(jd, JobDescription)
        assert len(jd.raw_text) > 100
        assert len(jd.core_requirements) > 0, "Core requirements section not parsed"
        assert len(jd.disqualifiers) > 0, "Disqualifiers section not parsed"
        assert len(jd.ideal_candidate) > 0, "Ideal candidate section not parsed"
    
    def test_file_not_found_raises(self):
        """Missing files should raise FileNotFoundError, not silently fail."""
        with pytest.raises(FileNotFoundError):
            load_sample_candidates("/nonexistent/path.json")
        
        with pytest.raises(FileNotFoundError):
            list(load_candidates_streaming("/nonexistent/path.jsonl"))


# =============================================================================
# Text Builder Tests
# =============================================================================

class TestTextBuilder:
    """Verify text representations are meaningful."""
    
    def test_text_not_empty(self):
        """Text builder should produce non-empty output for any candidate."""
        with open(SAMPLE_CANDIDATES_PATH) as f:
            raw = json.load(f)
        
        candidate = Candidate.model_validate(raw[0])
        text = build_candidate_text(candidate)
        
        assert len(text) > 100, "Text too short — missing sections?"
        assert candidate.profile.headline in text
        assert candidate.profile.current_title in text
    
    def test_text_includes_skills(self):
        """Text should include skill names and proficiency levels."""
        with open(SAMPLE_CANDIDATES_PATH) as f:
            raw = json.load(f)
        
        candidate = Candidate.model_validate(raw[0])
        text = build_candidate_text(candidate)
        
        # At least some skills should appear in the text
        skill_names = [s.name for s in candidate.skills]
        found = sum(1 for name in skill_names if name in text)
        assert found > 0, f"No skills found in text. Skills: {skill_names}"
    
    def test_text_includes_career(self):
        """Text should include career descriptions."""
        with open(SAMPLE_CANDIDATES_PATH) as f:
            raw = json.load(f)
        
        candidate = Candidate.model_validate(raw[0])
        text = build_candidate_text(candidate)
        
        # First career entry's company should be in the text
        assert candidate.career_history[0].company in text
    
    def test_all_sample_candidates_produce_text(self):
        """Every sample candidate should produce meaningful text."""
        candidates = load_sample_candidates(SAMPLE_CANDIDATES_PATH)
        
        for c in candidates:
            text = build_candidate_text(c)
            assert len(text) > 50, f"Candidate {c.candidate_id} produced too-short text"
