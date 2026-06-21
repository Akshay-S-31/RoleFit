"""
RoleFit - Data Loader

Handles ingestion and validation of candidate JSONL and the job description.

Design decisions:
- JSONL is streamed line-by-line to avoid loading 465MB into memory at once.
- Each line is validated through Pydantic models — malformed records are logged and skipped.
- Provides both a generator (for streaming) and a batch loader (for FAISS indexing).
- JD is pre-extracted to a text file to avoid runtime docx parsing dependency.
- JD text is parsed into structured sections by looking for known headings.

Memory considerations:
- 100K candidates × ~5KB each ≈ 500MB when all in memory as Pydantic objects.
- The generator approach lets downstream modules process in chunks if needed.
- For FAISS indexing (Phase 2), we'll need all in memory — that's fine at 500MB.
"""

import json
import logging
from pathlib import Path
from typing import Generator, List, Optional, Union

from src.models import Candidate, JobDescription

logger = logging.getLogger(__name__)


def load_candidates_streaming(filepath: Union[str, Path]) -> Generator[Candidate, None, None]:
    """
    Stream candidates from a JSONL file, yielding validated Candidate objects.
    
    Malformed records are logged and skipped — the pipeline doesn't crash
    on a single bad record in a 100K dataset.
    
    Args:
        filepath: Path to candidates.jsonl
        
    Yields:
        Validated Candidate objects, one per JSONL line.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Candidates file not found: {filepath}")
    
    loaded = 0
    skipped = 0
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            try:
                raw = json.loads(line)
                candidate = Candidate.model_validate(raw)
                loaded += 1
                yield candidate
            except json.JSONDecodeError as e:
                skipped += 1
                logger.warning(f"Line {line_num}: Invalid JSON — {e}")
            except Exception as e:
                skipped += 1
                logger.warning(f"Line {line_num}: Validation failed — {e}")
    
    logger.info(f"Loaded {loaded} candidates, skipped {skipped}")


def load_candidates_batch(filepath: Union[str, Path]) -> List[Candidate]:
    """
    Load all candidates into memory at once.
    
    Use this when you need random access to all candidates (e.g., FAISS indexing).
    For 100K candidates, expect ~500MB memory usage.
    
    Args:
        filepath: Path to candidates.jsonl
        
    Returns:
        List of all validated Candidate objects.
    """
    return list(load_candidates_streaming(filepath))


def load_sample_candidates(filepath: Union[str, Path]) -> List[Candidate]:
    """
    Load sample candidates from a JSON array file (not JSONL).
    
    The sample_candidates.json file is a JSON array of 50 candidates,
    useful for quick testing without loading the full 100K dataset.
    
    Args:
        filepath: Path to sample_candidates.json
        
    Returns:
        List of validated Candidate objects.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Sample candidates file not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        raw_list = json.load(f)
    
    candidates = []
    for i, raw in enumerate(raw_list):
        try:
            candidates.append(Candidate.model_validate(raw))
        except Exception as e:
            logger.warning(f"Sample candidate {i}: Validation failed — {e}")
    
    logger.info(f"Loaded {len(candidates)} sample candidates")
    return candidates


def load_job_description(filepath: Union[str, Path]) -> JobDescription:
    """
    Load and parse the job description from a pre-extracted text file.
    
    The JD is split into structured sections based on known headings
    from the actual Redrob JD. This lets downstream modules reason
    about requirements, disqualifiers, and behavioral expectations separately.
    
    Args:
        filepath: Path to data/jd.txt
        
    Returns:
        A JobDescription with structured sections populated.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JD file not found: {filepath}")
    
    raw_text = filepath.read_text(encoding="utf-8")
    
    # Parse into sections by known headings from the Redrob JD
    sections = _parse_jd_sections(raw_text)
    
    return JobDescription(
        title="Senior AI/ML Engineer — Ranking & Retrieval",
        raw_text=raw_text,
        core_requirements=sections.get("core_requirements", ""),
        nice_to_have=sections.get("nice_to_have", ""),
        disqualifiers=sections.get("disqualifiers", ""),
        behavioral_expectations=sections.get("behavioral_expectations", ""),
        ideal_candidate=sections.get("ideal_candidate", ""),
    )


def _parse_jd_sections(text: str) -> dict:
    """
    Parse JD text into named sections based on known headings.
    
    The Redrob JD has clear section markers we can split on.
    If a section heading isn't found, that section gets an empty string.
    """
    sections = {}
    
    # Define section markers and their corresponding keys
    # Order matters — we search from top to bottom
    markers = [
        ("Things you absolutely need", "core_requirements"),
        ("Things we'd like you to have but won't reject you for", "nice_to_have"),
        ("Things we explicitly do NOT want", "disqualifiers"),
        ("The vibe check", "behavioral_expectations"),
        ("How to read between the lines", "ideal_candidate"),
    ]
    
    text_lower = text.lower()
    
    for i, (marker, key) in enumerate(markers):
        start = text_lower.find(marker.lower())
        if start == -1:
            sections[key] = ""
            continue
        
        # Find the end: either the next marker's start, or end of text
        end = len(text)
        for j in range(i + 1, len(markers)):
            next_start = text_lower.find(markers[j][0].lower())
            if next_start != -1:
                end = next_start
                break
        
        sections[key] = text[start:end].strip()
    
    return sections
