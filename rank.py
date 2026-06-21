#!/usr/bin/env python3
"""
RoleFit - Ranking Entry Point

The single command that produces submission.csv from candidates.jsonl.
Must complete within 5 minutes on CPU with 16GB RAM and no network access.

Requirements (from submission_spec):
- Pre-computed FAISS index must exist (run precompute.py first)
- No network calls during execution
- Output: exactly 100 rows, ranked 1-100, scores non-increasing

Usage:
    python rank.py
    python rank.py --candidates ./path/to/candidates.jsonl --out ./output/submission.csv
"""

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rank")

from config import CANDIDATES_JSONL_PATH, SUBMISSION_CSV_PATH, FAISS_INDEX_PATH
from src.pipeline import RecruiterPipeline


def main():
    parser = argparse.ArgumentParser(
        description="RoleFit: AI-powered candidate ranking (CPU, <5min)"
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default=str(CANDIDATES_JSONL_PATH),
        help="Path to candidates JSONL file",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(SUBMISSION_CSV_PATH),
        help="Path to output submission CSV",
    )
    args = parser.parse_args()

    # Pre-flight check: make sure precompute.py has been run
    if not FAISS_INDEX_PATH.exists():
        logger.error(
            f"FAISS index not found at {FAISS_INDEX_PATH}. "
            "Run 'python precompute.py' first."
        )
        sys.exit(1)

    start = time.time()
    logger.info("Starting RoleFit ranking pipeline...")
    logger.info(f"  Candidates: {args.candidates}")
    logger.info(f"  Output:     {args.out}")

    pipeline = RecruiterPipeline()
    rows = pipeline.run(
        candidates_path=args.candidates,
        output_path=args.out,
    )

    elapsed = time.time() - start
    logger.info(f"Done in {elapsed:.1f}s — {len(rows)} candidates ranked")
    logger.info(f"Top-3:")
    for row in rows[:3]:
        logger.info(f"  #{row.rank}: {row.candidate_id} (score={row.score:.4f})")


if __name__ == "__main__":
    main()
