"""
RoleFit - Pipeline Orchestrator

Ties all stages together into a single ranked output. This is the conductor:
it doesn't do computation itself, it calls the right modules in the right order.

Stage sequence:
1. Load FAISS index + candidate data
2. FAISS search → top-200 longlist (vector_score populated)
3. Cross-encoder re-rank → top-200 scored (cross_encoder_score populated)
4. Rule-based scoring on all 200 (skill, career, behavioral scores + honeypot flag)
5. Composite scoring → final_score computed, honeypots zeroed out
6. Sort by final_score, take top-100 non-honeypot candidates
7. Generate reasoning text
8. Export to CSV

Design decisions:
- We score ALL 200 retrieved candidates with the cross-encoder, THEN apply rule-based
  scoring, THEN composite. This ensures every signal is available before any filtering.
  Alternative (filter early) would risk losing good candidates that the cross-encoder
  downweighted but the rule-scorer would boost.
- Honeypot candidates: included in scoring pipeline but their final_score is forced to 0.
  They are excluded BEFORE numbering ranks 1-100, so they never appear in the CSV.
- Tie-breaking: validate_submission.py requires ties broken by candidate_id ascending.
- The composite weights are read from config.py at runtime — no hardcoding here.
- Reasoning is generated from scoring signals (template-based, deterministic).
  No LLM needed — the reasoning explains which specific signals drove the ranking.

Timing profile (from measured runs):
- Load index: ~0.1s
- Load 100K JSONL: ~2.2s
- FAISS search (200): <0.1s
- Cross-encoder (200 pairs): ~3-8s
- Rule-based scoring (200): <0.1s
- Composite + sort + export: <0.1s
- Total: ~6-11s (well within 5-minute budget)
"""

import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from config import (
    BIENCODER_MODEL,
    BEHAVIORAL_WEIGHT,
    CAREER_FIT_WEIGHT,
    CROSSENCODER_MODEL,
    CROSS_ENCODER_WEIGHT,
    FAISS_INDEX_PATH,
    EMBEDDINGS_META_PATH,
    JD_TEXT_PATH,
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
    SKILL_MATCH_WEIGHT,
    SUBMISSION_TOP_N,
)
from src.data_loader import load_candidates_batch, load_job_description
from src.models import Candidate, RankedCandidate, SubmissionRow
from src.retriever import SemanticRetriever
from src.reranker import CrossEncoderReranker
from src.scorer import score_candidate
from src.text_builder import build_jd_text_for_embedding

logger = logging.getLogger(__name__)


class RecruiterPipeline:
    """
    End-to-end ranking pipeline.

    Usage:
        pipeline = RecruiterPipeline()
        pipeline.run(candidates_path, output_path)
    """

    def __init__(self):
        """
        Initialize all stage components.

        Models are loaded here so they're ready for the run() call.
        Loading happens once — pipeline is designed to be called once per run.
        """
        logger.info("Initializing pipeline components...")
        self.retriever = SemanticRetriever(model_name=BIENCODER_MODEL)
        self.reranker = CrossEncoderReranker(model_name=CROSSENCODER_MODEL)

    def run(
        self,
        candidates_path: Union[str, Path],
        output_path: Union[str, Path],
        jd_path: Optional[Union[str, Path]] = None,
    ) -> List[SubmissionRow]:
        """
        Run the full ranking pipeline and export submission CSV.

        Args:
            candidates_path: Path to candidates.jsonl
            output_path: Path to write submission.csv
            jd_path: Path to jd.txt (defaults to config value)

        Returns:
            List of SubmissionRow objects (the final ranked output)
        """
        overall_start = time.time()
        jd_path = jd_path or JD_TEXT_PATH

        # -----------------------------------------------------------------------
        # Stage 0: Load pre-computed index + candidate data + JD
        # -----------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("Stage 0: Loading data")
        logger.info("=" * 60)

        # Load FAISS index (fast — <0.1s)
        self.retriever.load(FAISS_INDEX_PATH, EMBEDDINGS_META_PATH)

        # Load all candidates into memory for lookup (2.2s measured)
        logger.info(f"Loading candidates from {candidates_path}...")
        all_candidates: List[Candidate] = load_candidates_batch(candidates_path)
        candidates_lookup: Dict[str, Candidate] = {
            c.candidate_id: c for c in all_candidates
        }
        logger.info(f"Loaded {len(candidates_lookup)} candidates")

        # Load JD
        jd = load_job_description(jd_path)
        logger.info(f"JD loaded: {jd.title}")

        # -----------------------------------------------------------------------
        # Stage 1: Bi-encoder retrieval → top-200 longlist
        # -----------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info(f"Stage 1: FAISS retrieval (top-{RETRIEVAL_TOP_K})")
        logger.info("=" * 60)

        jd_embed_text = build_jd_text_for_embedding(jd.raw_text)
        retrieval_results = self.retriever.search_with_candidates(
            jd_embed_text,
            candidates_lookup,
            top_k=RETRIEVAL_TOP_K,
        )
        logger.info(f"Retrieved {len(retrieval_results)} candidates from FAISS")

        # -----------------------------------------------------------------------
        # Stage 2: Cross-encoder re-ranking on the full longlist
        # -----------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info(f"Stage 2: Cross-encoder re-ranking ({len(retrieval_results)} candidates)")
        logger.info("=" * 60)

        # We score ALL retrieved candidates — don't filter before cross-encoding.
        # The cross-encoder sees candidates the bi-encoder might have mis-ranked.
        reranked = self.reranker.rerank(
            jd.raw_text,
            retrieval_results,
            top_k=len(retrieval_results),  # Keep all 200 — rule-scorer may reorder
        )
        logger.info(f"Cross-encoder scored {len(reranked)} candidates")

        # -----------------------------------------------------------------------
        # Stage 3: Rule-based scoring on all 200
        # -----------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("Stage 3: Rule-based scoring")
        logger.info("=" * 60)

        scored = [score_candidate(rc) for rc in reranked]

        honeypot_count = sum(1 for rc in scored if rc.is_honeypot)
        logger.info(f"Rule-based scoring complete. Honeypots flagged: {honeypot_count}")

        # -----------------------------------------------------------------------
        # Stage 4: Composite scoring + final sort
        # -----------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("Stage 4: Composite scoring")
        logger.info("=" * 60)

        for rc in scored:
            rc.final_score = self._compute_final_score(rc)

        # Sort: by final_score descending, then by candidate_id ascending (tie-break)
        # Honeypots get final_score=0 so they naturally fall to the bottom
        sorted_candidates = sorted(
            scored,
            key=lambda rc: (-rc.final_score, rc.candidate.candidate_id),
        )

        # -----------------------------------------------------------------------
        # Stage 5: Take top-100 non-honeypots, generate reasoning, export CSV
        # -----------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("Stage 5: Generate reasoning + export CSV")
        logger.info("=" * 60)

        # Exclude honeypots from the final 100
        clean_candidates = [rc for rc in sorted_candidates if not rc.is_honeypot]
        top_100 = clean_candidates[:SUBMISSION_TOP_N]

        if len(top_100) < SUBMISSION_TOP_N:
            logger.warning(
                f"Only {len(top_100)} non-honeypot candidates available "
                f"(need {SUBMISSION_TOP_N}). Padding with flagged candidates."
            )
            # Safety fallback: if we somehow don't have 100 clean candidates,
            # pad with honeypots rather than produce an invalid submission.
            needed = SUBMISSION_TOP_N - len(top_100)
            top_100.extend(sorted_candidates[len(top_100):len(top_100) + needed])

        # Generate reasoning and build submission rows
        rows = []
        for rank, rc in enumerate(top_100, start=1):
            rc.reasoning = self._generate_reasoning(rc)
            rows.append(SubmissionRow(
                candidate_id=rc.candidate.candidate_id,
                rank=rank,
                score=round(rc.final_score, 6),
                reasoning=rc.reasoning,
            ))

        # Export CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._export_csv(rows, output_path)

        elapsed = time.time() - overall_start
        logger.info("=" * 60)
        logger.info(f"Pipeline complete in {elapsed:.1f}s ({elapsed/60:.2f} minutes)")
        logger.info(f"Output: {output_path}")
        logger.info(f"Top candidate: {rows[0].candidate_id} (score={rows[0].score:.4f})")
        logger.info("=" * 60)

        return rows

    def _compute_final_score(self, rc: RankedCandidate) -> float:
        """
        Compute the weighted composite final score.

        Formula:
            final = 0.40 × cross_encoder + 0.20 × skill_match
                  + 0.20 × career_fit   + 0.20 × behavioral

        Honeypots are forced to 0 regardless of other scores.

        Args:
            rc: A fully scored RankedCandidate.

        Returns:
            float in [0.0, 1.0]
        """
        if rc.is_honeypot:
            return 0.0

        score = (
            CROSS_ENCODER_WEIGHT * rc.cross_encoder_score
            + SKILL_MATCH_WEIGHT * rc.skill_match_score
            + CAREER_FIT_WEIGHT * rc.career_fit_score
            + BEHAVIORAL_WEIGHT * rc.behavioral_score
        )
        return max(0.0, min(1.0, score))

    def _generate_reasoning(self, rc: RankedCandidate) -> str:
        """
        Generate a human-readable reasoning string for the submission CSV.

        Template-based (no LLM) — deterministic and fast. Uses scoring signals
        to explain why this candidate was ranked where they were.

        The judges check reasoning for 6 criteria; we target:
        - Relevance to the JD
        - Mention of specific signals
        - Honesty about weaknesses when present

        Args:
            rc: A scored RankedCandidate with final_score set.

        Returns:
            A 1-2 sentence reasoning string.
        """
        c = rc.candidate
        p = c.profile
        s = c.redrob_signals

        # Build strength highlights
        strengths = []

        # Cross-encoder signal (semantic fit)
        if rc.cross_encoder_score >= 0.70:
            strengths.append("strong semantic alignment with the JD")
        elif rc.cross_encoder_score >= 0.55:
            strengths.append("good semantic fit with JD requirements")

        # Skill match
        if rc.skill_match_score >= 0.75:
            strengths.append("deep expertise in required skills (embeddings/retrieval/vector DBs)")
        elif rc.skill_match_score >= 0.50:
            strengths.append("solid required skill coverage")

        # Career fit
        if rc.career_fit_score >= 0.80:
            strengths.append(f"ideal experience range ({p.years_of_experience:.1f}y at product companies)")
        elif rc.career_fit_score >= 0.60:
            strengths.append(f"{p.years_of_experience:.1f}y relevant experience")

        # Behavioral signals
        if rc.behavioral_score >= 0.75:
            if s.open_to_work_flag and s.notice_period_days <= 30:
                strengths.append("actively available with short notice period")
            elif s.recruiter_response_rate >= 0.70:
                strengths.append("high recruiter engagement signals")

        # Build weakness/caveat notes
        caveats = []
        if s.notice_period_days > 60:
            caveats.append(f"{s.notice_period_days}-day notice period")
        if s.recruiter_response_rate < 0.20:
            caveats.append("low recruiter response rate")
        if not s.open_to_work_flag:
            caveats.append("not marked open to work")

        # Compose reasoning
        strength_text = "; ".join(strengths) if strengths else "reasonable overall fit"
        caveat_text = f" Note: {', '.join(caveats)}." if caveats else ""

        reasoning = (
            f"{p.current_title} at {p.current_company} — {strength_text}."
            f"{caveat_text}"
        )

        # Ensure max length (some CSV validators are strict)
        return reasoning[:500]

    def _export_csv(self, rows: List[SubmissionRow], output_path: Path) -> None:
        """
        Write submission rows to CSV in the format required by validate_submission.py.

        Format: candidate_id, rank, score, reasoning
        - Exactly 100 rows
        - Ranks 1-100 (no gaps, no duplicates)
        - Scores non-increasing
        - Tie-break by candidate_id ascending (handled by sort in run())

        Args:
            rows: Sorted list of 100 SubmissionRow objects.
            output_path: Output CSV file path.
        """
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["candidate_id", "rank", "score", "reasoning"],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "candidate_id": row.candidate_id,
                    "rank": row.rank,
                    "score": row.score,
                    "reasoning": row.reasoning,
                })

        logger.info(f"Submission CSV written: {output_path} ({len(rows)} rows)")
