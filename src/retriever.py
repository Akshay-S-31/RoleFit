"""
RoleFit - Semantic Retriever (Stage 1)

Bi-encoder dense vector search using sentence-transformers + FAISS.
Filters 100K candidates down to a longlist of ~200.

Design decisions:
- Uses IndexFlatIP (inner product on L2-normalized vectors = cosine similarity).
  With 100K vectors at 384 dimensions, exact search is fast enough (<100ms)
  and we don't lose any candidates to approximation errors.
- Saves FAISS index + candidate_id list to disk separately. We do NOT cache
  full Candidate objects — loading the JSONL takes only 2.2s (measured),
  well within the 5-min rank-time budget.
- JD embedding uses a focused subset of the JD (core requirements + ideal candidate),
  not the full ~9500-char text. The full JD has logistics, comp details, and hackathon
  instructions that dilute the semantic signal and exceed the model's 256-token window.
- Batch encoding with progress reporting for the ~10-15 min precompute step.

Future impact:
- If a strong candidate isn't in the top-200 here, they're lost forever.
  We cast a wide net (200) to be safe; the cross-encoder narrows to 100.
- The quality of text representations (from text_builder.py) directly determines
  who makes the longlist. Tuning text_builder changes retrieval results.
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.models import Candidate, RankedCandidate
from src.text_builder import build_candidate_text

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    Bi-encoder retrieval using sentence-transformers + FAISS.
    
    Two modes of operation:
    1. Pre-compute (slow, run once): build_index() + save()
    2. Rank-time (fast, <5min budget): load() + search()
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the retriever.
        
        The model is loaded eagerly because both precompute and rank-time
        need it (precompute for candidate embeddings, rank-time for JD embedding).
        On first run it downloads ~80MB; subsequent runs use the cache.
        
        Args:
            model_name: HuggingFace model name for the bi-encoder.
        """
        logger.info(f"Loading bi-encoder model: {model_name}")
        start = time.time()
        self.model = SentenceTransformer(model_name)
        logger.info(f"Model loaded in {time.time() - start:.1f}s")
        
        self.index: Optional[faiss.IndexFlatIP] = None
        self.candidate_ids: List[str] = []  # Maps FAISS index position → candidate_id
        self.embedding_dim: int = self.model.get_sentence_embedding_dimension()
    
    def build_index(
        self,
        candidates: List[Candidate],
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> None:
        """
        Embed all candidates and build the FAISS index.
        
        This is the slow pre-computation step (~10-15 min for 100K on CPU).
        Run once via precompute.py, then save to disk.
        
        Args:
            candidates: List of all candidates to index.
            batch_size: Encoding batch size. 256 balances speed and memory.
            show_progress: Whether to print progress during encoding.
        """
        n = len(candidates)
        logger.info(f"Building index for {n} candidates (embedding dim={self.embedding_dim})")
        
        # Step 1: Build text representations
        logger.info("Building text representations...")
        start = time.time()
        texts = [build_candidate_text(c) for c in candidates]
        self.candidate_ids = [c.candidate_id for c in candidates]
        logger.info(f"Text built in {time.time() - start:.1f}s")
        
        # Step 2: Encode all texts in batches
        logger.info(f"Encoding {n} candidates (batch_size={batch_size})...")
        start = time.time()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2 normalize so inner product = cosine similarity
            convert_to_numpy=True,
        )
        logger.info(f"Encoding completed in {time.time() - start:.1f}s")
        
        # Step 3: Build FAISS index
        # IndexFlatIP = exact inner product search (cosine sim with normalized vectors)
        logger.info("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype(np.float32))
        logger.info(f"Index built: {self.index.ntotal} vectors")
    
    def save(self, index_path: Union[str, Path], meta_path: Union[str, Path]) -> None:
        """
        Save the FAISS index and candidate ID mapping to disk.
        
        Args:
            index_path: Path to save the FAISS index (.faiss)
            meta_path: Path to save the candidate ID list (.pkl)
        """
        if self.index is None:
            raise RuntimeError("No index to save. Call build_index() first.")
        
        index_path = Path(index_path)
        meta_path = Path(meta_path)
        
        # Ensure output directory exists
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(index_path))
        logger.info(f"FAISS index saved: {index_path} ({index_path.stat().st_size / 1024 / 1024:.1f} MB)")
        
        with open(meta_path, "wb") as f:
            pickle.dump({"candidate_ids": self.candidate_ids}, f)
        logger.info(f"Metadata saved: {meta_path}")
    
    def load(self, index_path: Union[str, Path], meta_path: Union[str, Path]) -> None:
        """
        Load a pre-built FAISS index and candidate ID mapping from disk.
        
        This is the fast path used at rank-time. Loading is typically <1s.
        
        Args:
            index_path: Path to the saved FAISS index (.faiss)
            meta_path: Path to the saved candidate ID list (.pkl)
        """
        index_path = Path(index_path)
        meta_path = Path(meta_path)
        
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}. Run precompute.py first.")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}. Run precompute.py first.")
        
        start = time.time()
        self.index = faiss.read_index(str(index_path))
        
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.candidate_ids = meta["candidate_ids"]
        
        logger.info(
            f"Index loaded: {self.index.ntotal} vectors in {time.time() - start:.1f}s"
        )
    
    def search(
        self,
        query_text: str,
        top_k: int = 200,
    ) -> List[Tuple[str, float]]:
        """
        Embed the query and find the top-K most similar candidates.
        
        Returns (candidate_id, similarity_score) pairs, sorted by score descending.
        The score is cosine similarity (0-1 range for normalized vectors).
        
        Args:
            query_text: The text to search for (typically JD text).
            top_k: Number of top candidates to return.
            
        Returns:
            List of (candidate_id, vector_score) tuples, highest first.
        """
        if self.index is None:
            raise RuntimeError("No index loaded. Call load() or build_index() first.")
        
        # Embed the query
        query_embedding = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        
        # Search FAISS
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Map indices to candidate_ids
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            results.append((self.candidate_ids[idx], float(score)))
        
        return results
    
    def search_with_candidates(
        self,
        query_text: str,
        candidates_lookup: Dict[str, Candidate],
        top_k: int = 200,
    ) -> List[RankedCandidate]:
        """
        Search and return full RankedCandidate objects with vector_score populated.
        
        Convenience method that combines FAISS search with candidate data lookup.
        
        Args:
            query_text: The text to search for (typically JD text).
            candidates_lookup: Dict mapping candidate_id → Candidate object.
            top_k: Number of top candidates to return.
            
        Returns:
            List of RankedCandidate objects with vector_score set, highest first.
        """
        results = self.search(query_text, top_k)
        
        ranked = []
        for candidate_id, score in results:
            if candidate_id in candidates_lookup:
                ranked.append(RankedCandidate(
                    candidate=candidates_lookup[candidate_id],
                    vector_score=score,
                ))
            else:
                logger.warning(f"Candidate {candidate_id} found in index but not in lookup")
        
        return ranked
