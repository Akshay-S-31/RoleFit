"""
Tests for the semantic retriever.

Uses the 50 sample candidates to verify:
1. Index builds correctly
2. Save/load round-trips without data loss
3. Search returns sensible results
4. JD text builder produces focused output
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BIENCODER_MODEL, SAMPLE_CANDIDATES_PATH, JD_TEXT_PATH
from src.data_loader import load_sample_candidates, load_job_description
from src.retriever import SemanticRetriever
from src.text_builder import build_jd_text_for_embedding


# Use a module-level fixture so the model is loaded only once per test session
@pytest.fixture(scope="module")
def retriever():
    """Load the bi-encoder model once for all tests."""
    return SemanticRetriever(model_name=BIENCODER_MODEL)


@pytest.fixture(scope="module")
def sample_candidates():
    """Load sample candidates once for all tests."""
    return load_sample_candidates(SAMPLE_CANDIDATES_PATH)


@pytest.fixture(scope="module")
def jd():
    """Load JD once for all tests."""
    if not JD_TEXT_PATH.exists():
        pytest.skip("JD text file not available")
    return load_job_description(JD_TEXT_PATH)


class TestRetriever:
    """Core retriever functionality tests."""
    
    def test_build_index(self, retriever, sample_candidates):
        """Index should build without errors on sample data."""
        retriever.build_index(sample_candidates, batch_size=16, show_progress=False)
        assert retriever.index is not None
        assert retriever.index.ntotal == len(sample_candidates)
        assert len(retriever.candidate_ids) == len(sample_candidates)
    
    def test_search_returns_results(self, retriever, jd):
        """Search should return ranked results."""
        jd_text = build_jd_text_for_embedding(jd.raw_text)
        results = retriever.search(jd_text, top_k=10)
        
        assert len(results) == 10
        # Results should be (candidate_id, score) tuples
        for cid, score in results:
            assert cid.startswith("CAND_")
            assert 0 <= score <= 1  # Cosine similarity of normalized vectors
        
        # Scores should be descending
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_save_load_roundtrip(self, retriever, jd):
        """Saved index should produce identical search results after loading."""
        jd_text = build_jd_text_for_embedding(jd.raw_text)
        
        # Get results before save
        results_before = retriever.search(jd_text, top_k=10)
        
        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "test.faiss"
            meta_path = Path(tmpdir) / "test.pkl"
            
            retriever.save(idx_path, meta_path)
            
            # Create a fresh retriever and load
            retriever2 = SemanticRetriever(model_name=BIENCODER_MODEL)
            retriever2.load(idx_path, meta_path)
            
            results_after = retriever2.search(jd_text, top_k=10)
        
        # Results should be identical
        assert len(results_before) == len(results_after)
        for (cid1, s1), (cid2, s2) in zip(results_before, results_after):
            assert cid1 == cid2
            assert abs(s1 - s2) < 1e-6
    
    def test_search_with_candidates(self, retriever, sample_candidates, jd):
        """search_with_candidates should return RankedCandidate objects."""
        lookup = {c.candidate_id: c for c in sample_candidates}
        jd_text = build_jd_text_for_embedding(jd.raw_text)
        
        ranked = retriever.search_with_candidates(jd_text, lookup, top_k=5)
        
        assert len(ranked) == 5
        for rc in ranked:
            assert rc.candidate.candidate_id.startswith("CAND_")
            assert rc.vector_score > 0


class TestJDTextBuilder:
    """Verify JD text builder produces focused output."""
    
    def test_jd_text_is_focused(self, jd):
        """Focused JD text should be shorter than raw text."""
        focused = build_jd_text_for_embedding(jd.raw_text)
        assert len(focused) < len(jd.raw_text)
        assert len(focused) > 100  # Not empty
    
    def test_jd_text_contains_requirements(self, jd):
        """Focused text should include technical requirements."""
        focused = build_jd_text_for_embedding(jd.raw_text).lower()
        # These are key requirement terms from the JD
        assert "embeddings" in focused or "retrieval" in focused
        assert "vector" in focused or "ranking" in focused
