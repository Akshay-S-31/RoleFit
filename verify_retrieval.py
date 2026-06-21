#!/usr/bin/env python3
"""
Quick verification: load the pre-built FAISS index, search with the JD,
and print the top-20 results for manual inspection.

Usage:
    python verify_retrieval.py
"""

import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify")

from config import (
    BIENCODER_MODEL,
    CANDIDATES_JSONL_PATH,
    FAISS_INDEX_PATH,
    EMBEDDINGS_META_PATH,
    JD_TEXT_PATH,
    RETRIEVAL_TOP_K,
)
from src.data_loader import load_candidates_batch, load_job_description
from src.retriever import SemanticRetriever
from src.text_builder import build_jd_text_for_embedding


def main():
    start = time.time()
    
    # Load the pre-built index
    retriever = SemanticRetriever(model_name=BIENCODER_MODEL)
    retriever.load(FAISS_INDEX_PATH, EMBEDDINGS_META_PATH)
    
    # Load JD
    jd = load_job_description(JD_TEXT_PATH)
    jd_text = build_jd_text_for_embedding(jd.raw_text)
    
    # Search
    results = retriever.search(jd_text, top_k=20)
    
    # Load candidate data for display
    logger.info("Loading candidates for display...")
    candidates = load_candidates_batch(CANDIDATES_JSONL_PATH)
    lookup = {c.candidate_id: c for c in candidates}
    
    elapsed = time.time() - start
    
    print(f"\n{'='*80}")
    print(f"TOP 20 CANDIDATES (retrieved in {elapsed:.1f}s total)")
    print(f"{'='*80}\n")
    
    for rank, (cid, score) in enumerate(results, 1):
        c = lookup.get(cid)
        if not c:
            print(f"  {rank}. {cid} — NOT FOUND IN DATA")
            continue
        
        p = c.profile
        signals = c.redrob_signals
        
        # Count AI-relevant skills
        ai_keywords = {"ml", "machine learning", "deep learning", "nlp", "ai",
                       "pytorch", "tensorflow", "transformers", "embeddings",
                       "vector", "retrieval", "ranking", "recommendation",
                       "bert", "llm", "gpt", "fine-tuning", "lora",
                       "faiss", "pinecone", "qdrant", "weaviate",
                       "sentence-transformers", "huggingface"}
        ai_skills = [s.name for s in c.skills 
                     if any(kw in s.name.lower() for kw in ai_keywords)]
        
        # Check consulting-only
        consulting = {"tcs", "infosys", "wipro", "accenture", "cognizant", 
                      "capgemini", "hcl", "tech mahindra", "mindtree"}
        all_companies = {e.company.lower() for e in c.career_history}
        is_consulting_only = all_companies.issubset(consulting)
        
        flag = " ⚠️ CONSULTING-ONLY" if is_consulting_only else ""
        
        print(f"  {rank:2d}. [{score:.4f}] {cid} — {p.current_title} at {p.current_company}{flag}")
        print(f"      {p.headline}")
        print(f"      Exp: {p.years_of_experience}y | Location: {p.location}, {p.country} | Notice: {signals.notice_period_days}d")
        print(f"      Response rate: {signals.recruiter_response_rate:.0%} | GitHub: {signals.github_activity_score} | Open: {signals.open_to_work_flag}")
        if ai_skills:
            print(f"      AI Skills: {', '.join(ai_skills[:8])}")
        print()


if __name__ == "__main__":
    main()
