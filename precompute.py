#!/usr/bin/env python3
"""
RoleFit - Pre-computation Entry Point

Run this ONCE before rank.py to build the FAISS index. This step has
no time limit and can use network access (for model download).

What it does:
1. Loads all 100K candidates from JSONL (~2s)
2. Builds text representations for each candidate
3. Generates embeddings using the bi-encoder (~10-15 min on CPU)
4. Builds and saves the FAISS index to disk

Usage:
    python precompute.py

Produces:
    output/candidate_index.faiss   (~150 MB for 100K × 384-dim vectors)
    output/embeddings_meta.pkl     (~3 MB candidate ID mapping)
"""

import logging
import sys
import time

# Configure logging before imports so all modules use it
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("precompute")

from config import (
    BIENCODER_MODEL,
    CANDIDATES_JSONL_PATH,
    FAISS_INDEX_PATH,
    EMBEDDINGS_META_PATH,
)
from src.data_loader import load_candidates_batch
from src.retriever import SemanticRetriever


def main():
    overall_start = time.time()
    
    # Step 1: Load candidates
    logger.info(f"Loading candidates from {CANDIDATES_JSONL_PATH}...")
    candidates = load_candidates_batch(CANDIDATES_JSONL_PATH)
    logger.info(f"Loaded {len(candidates)} candidates")
    
    # Step 2: Initialize retriever (downloads model on first run)
    retriever = SemanticRetriever(model_name=BIENCODER_MODEL)
    
    # Step 3: Build FAISS index (the slow part)
    retriever.build_index(candidates, batch_size=256, show_progress=True)
    
    # Step 4: Save to disk
    retriever.save(FAISS_INDEX_PATH, EMBEDDINGS_META_PATH)
    
    elapsed = time.time() - overall_start
    logger.info(f"Pre-computation complete in {elapsed / 60:.1f} minutes")
    logger.info(f"Index: {FAISS_INDEX_PATH}")
    logger.info(f"Meta:  {EMBEDDINGS_META_PATH}")


if __name__ == "__main__":
    main()
