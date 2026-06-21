"""
Tests for the cross-encoder reranker and pipeline.

Uses sample candidates for speed. The full end-to-end test
(rank.py on 100K) is done separately via verify_submission.py.
"""

import csv
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BIENCODER_MODEL,
    CROSSENCODER_MODEL,
    JD_TEXT_PATH,
    SAMPLE_CANDIDATES_PATH,
    FAISS_INDEX_PATH,
    EMBEDDINGS_META_PATH,
)
from src.data_loader import load_sample_candidates, load_job_description
from src.models import RankedCandidate
from src.reranker import CrossEncoderReranker
from src.retriever import SemanticRetriever
from src.text_builder import build_jd_text_for_embedding
from src.pipeline import RecruiterPipeline


@pytest.fixture(scope="module")
def jd():
    if not JD_TEXT_PATH.exists():
        pytest.skip("JD text not available")
    return load_job_description(JD_TEXT_PATH)


@pytest.fixture(scope="module")
def sample_candidates():
    return load_sample_candidates(SAMPLE_CANDIDATES_PATH)


@pytest.fixture(scope="module")
def retriever_with_index(sample_candidates):
    """Build a small in-memory index from sample candidates."""
    r = SemanticRetriever(model_name=BIENCODER_MODEL)
    r.build_index(sample_candidates, batch_size=16, show_progress=False)
    return r


@pytest.fixture(scope="module")
def reranker():
    return CrossEncoderReranker(model_name=CROSSENCODER_MODEL)


@pytest.fixture(scope="module")
def longlist(retriever_with_index, sample_candidates, jd):
    """Top-20 candidates from the sample index."""
    lookup = {c.candidate_id: c for c in sample_candidates}
    jd_text = build_jd_text_for_embedding(jd.raw_text)
    return retriever_with_index.search_with_candidates(jd_text, lookup, top_k=20)


class TestCrossEncoderReranker:
    def test_rerank_returns_correct_count(self, reranker, longlist, jd):
        """rerank() should return exactly top_k candidates."""
        result = reranker.rerank(jd.raw_text, longlist, top_k=10)
        assert len(result) == 10

    def test_cross_encoder_scores_in_range(self, reranker, longlist, jd):
        """All cross-encoder scores should be in [0, 1] after sigmoid."""
        result = reranker.rerank(jd.raw_text, longlist, top_k=len(longlist))
        for rc in result:
            assert 0.0 <= rc.cross_encoder_score <= 1.0, (
                f"Score out of range: {rc.cross_encoder_score}"
            )

    def test_rerank_sorted_descending(self, reranker, longlist, jd):
        """Results should be sorted by cross_encoder_score descending."""
        result = reranker.rerank(jd.raw_text, longlist, top_k=len(longlist))
        scores = [rc.cross_encoder_score for rc in result]
        assert scores == sorted(scores, reverse=True)


class TestPipelineIntegration:
    def test_pipeline_produces_valid_csv(self, sample_candidates, jd):
        """Pipeline on sample candidates should produce a valid CSV structure."""
        if not FAISS_INDEX_PATH.exists():
            pytest.skip("FAISS index not built yet — run precompute.py first")

        # We can't easily run the full pipeline on sample data without rebuilding
        # the index, so this test just verifies the CSV export format via a
        # direct test of _export_csv and _generate_reasoning.
        from src.models import SubmissionRow
        from src.pipeline import RecruiterPipeline

        pipeline = RecruiterPipeline.__new__(RecruiterPipeline)

        # Build mock rows
        rows = [
            SubmissionRow(
                candidate_id=f"CAND_{i:07d}",
                rank=i,
                score=round(1.0 - (i - 1) * 0.005, 6),
                reasoning=f"Test reasoning for rank {i}.",
            )
            for i in range(1, 101)
        ]

        with tempfile.NamedTemporaryFile(
            suffix=".csv", mode="w", delete=False
        ) as f:
            tmp_path = Path(f.name)

        pipeline._export_csv(rows, tmp_path)

        # Verify CSV structure
        with open(tmp_path) as f:
            reader = csv.DictReader(f)
            read_rows = list(reader)

        assert len(read_rows) == 100
        assert set(reader.fieldnames) == {"candidate_id", "rank", "score", "reasoning"}

        # Ranks should be 1-100
        ranks = [int(r["rank"]) for r in read_rows]
        assert ranks == list(range(1, 101))

        # Scores should be non-increasing
        scores = [float(r["score"]) for r in read_rows]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"Score not non-increasing at rank {i+1}"

        tmp_path.unlink()

    def test_reasoning_generation(self, sample_candidates):
        """Reasoning should be non-empty and reasonably long."""
        from src.scorer import score_candidate
        from src.pipeline import RecruiterPipeline

        pipeline = RecruiterPipeline.__new__(RecruiterPipeline)
        c = sample_candidates[0]
        rc = RankedCandidate(candidate=c, cross_encoder_score=0.65, final_score=0.62)
        score_candidate(rc)

        reasoning = pipeline._generate_reasoning(rc)
        assert len(reasoning) > 20
        assert len(reasoning) <= 500
