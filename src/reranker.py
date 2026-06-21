"""
RoleFit - Cross-Encoder Re-Ranker (Stage 2)

Pairs each longlist candidate's text with the JD text and scores them jointly.
This is fundamentally more accurate than bi-encoder scoring because the model
reads both texts together, not as separate embeddings.

Design decisions:
- Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast on CPU, ~1-2s for 200 pairs)
  This model was trained on MS MARCO passage ranking — it understands query-document
  relevance natively, which maps well to JD-candidate matching.
- Input format: [JD_text, candidate_text] pairs. The cross-encoder sees the full
  context of both simultaneously. We use the focused JD text (core requirements +
  ideal candidate sections) not the full 9500-char version.
- Output: raw logit scores, NOT probabilities. We normalize to [0,1] using
  sigmoid so they're comparable with other scorer outputs.
- Batch size: 16 is safe for CPU memory. With 200 candidates this is ~13 batches.

Future impact:
- This score gets 40% weight in the final composite (highest of any signal).
- The cross-encoder is the most expensive compute step at rank-time (~1-3s for 200).
- If we ever increase RETRIEVAL_TOP_K beyond 500, we'd need to profile this again.
"""

import logging
import time
from typing import List, Union

from sentence_transformers import CrossEncoder

from src.models import RankedCandidate
from src.text_builder import build_candidate_text, build_jd_text_for_embedding

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    """Standard sigmoid to convert logit to [0, 1]."""
    import math
    return 1.0 / (1.0 + math.exp(-x))


class CrossEncoderReranker:
    """
    Cross-encoder re-ranking using sentence-transformers CrossEncoder.

    Usage:
        reranker = CrossEncoderReranker()
        ranked = reranker.rerank(jd_text, candidates, top_k=100)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Load the cross-encoder model.

        The model is ~80MB and loads in ~2s on CPU.

        Args:
            model_name: HuggingFace cross-encoder model name.
        """
        logger.info(f"Loading cross-encoder model: {model_name}")
        start = time.time()
        self.model = CrossEncoder(model_name, max_length=512)
        logger.info(f"Cross-encoder loaded in {time.time() - start:.1f}s")

    def rerank(
        self,
        jd_raw_text: str,
        candidates: List[RankedCandidate],
        top_k: int = 100,
        batch_size: int = 16,
    ) -> List[RankedCandidate]:
        """
        Score all candidates against the JD and return top_k, highest first.

        Populates rc.cross_encoder_score (0-1 range via sigmoid) for each candidate.
        Does NOT apply any filtering — honeypot filtering happens in pipeline.py.

        Args:
            jd_raw_text: The full JD text (function extracts focused subset internally).
            candidates: Longlist of RankedCandidates from the retriever.
            top_k: How many top candidates to return.
            batch_size: Batch size for model inference.

        Returns:
            Top-k RankedCandidates with cross_encoder_score populated, sorted
            by cross_encoder_score descending.
        """
        if not candidates:
            return []

        # Build focused JD text for the cross-encoder
        # Same focused subset as the bi-encoder — core requirements + ideal candidate
        jd_text = build_jd_text_for_embedding(jd_raw_text)

        # Build (JD, candidate) input pairs
        logger.info(f"Building {len(candidates)} input pairs for cross-encoder...")
        pairs = [
            [jd_text, build_candidate_text(rc.candidate)]
            for rc in candidates
        ]

        # Score all pairs
        logger.info(f"Scoring with cross-encoder (batch_size={batch_size})...")
        start = time.time()
        raw_scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
        elapsed = time.time() - start
        logger.info(f"Cross-encoder scored {len(candidates)} candidates in {elapsed:.1f}s")

        # Normalize scores to [0, 1] via sigmoid and store on candidates
        for rc, raw_score in zip(candidates, raw_scores):
            rc.cross_encoder_score = _sigmoid(float(raw_score))

        # Sort by cross-encoder score descending and return top_k
        sorted_candidates = sorted(
            candidates,
            key=lambda rc: rc.cross_encoder_score,
            reverse=True,
        )

        return sorted_candidates[:top_k]
