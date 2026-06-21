"""
RoleFit - Text Builder

Converts structured Candidate objects into rich, searchable text representations
for embedding by the bi-encoder.

Why this exists:
- The retriever needs plain text to embed, but our data is structured JSON.
- The ORDER and CONTENT of the text matters for retrieval quality.
- We front-load the most semantically important information (headline, summary,
  recent career descriptions) because embedding models weight earlier tokens more.
- We deliberately include proficiency levels and duration with skills, not just
  skill names — this helps the embedder distinguish "expert in PyTorch for 5 years"
  from "beginner in PyTorch for 2 months".

Future impact:
- This is the single most tunable component for retrieval quality.
- Changing what text goes in (and in what order) changes what the embedder "sees".
- No other module needs to change when we tune this.
"""

from src.models import Candidate


def build_candidate_text(candidate: Candidate) -> str:
    """
    Build a searchable text representation of a candidate profile.
    
    Designed for semantic embedding — front-loads the most important signals
    and includes contextual detail that keyword matchers would miss.
    
    Args:
        candidate: A validated Candidate object.
        
    Returns:
        A single string suitable for embedding. Typically 500-2000 chars.
    """
    sections = []
    
    # 1. Identity & headline — most compressed signal of who they are
    p = candidate.profile
    sections.append(f"{p.current_title} at {p.current_company} ({p.current_industry})")
    sections.append(p.headline)
    
    # 2. Professional summary — their own words about what they do
    if p.summary:
        sections.append(p.summary)
    
    # 3. Career history — descriptions contain the real evidence of what they've built
    # Most recent roles first (already ordered in data)
    for entry in candidate.career_history:
        role_line = f"{entry.title} at {entry.company} ({entry.industry}, {entry.company_size} employees) for {entry.duration_months} months"
        sections.append(role_line)
        if entry.description:
            sections.append(entry.description)
    
    # 4. Skills with proficiency — not just names, but depth indicators
    if candidate.skills:
        skill_parts = []
        for s in candidate.skills:
            skill_str = f"{s.name} ({s.proficiency}"
            if s.duration_months and s.duration_months > 0:
                skill_str += f", {s.duration_months} months"
            skill_str += f", {s.endorsements} endorsements)"
            skill_parts.append(skill_str)
        sections.append("Skills: " + "; ".join(skill_parts))
    
    # 5. Education — degree, field, and tier matter for this JD
    if candidate.education:
        edu_parts = []
        for e in candidate.education:
            edu_str = f"{e.degree} in {e.field_of_study} from {e.institution}"
            if e.tier:
                edu_str += f" ({e.tier})"
            edu_parts.append(edu_str)
        sections.append("Education: " + "; ".join(edu_parts))
    
    # 6. Certifications — can signal specialized knowledge
    if candidate.certifications:
        cert_names = [f"{c.name} ({c.issuer}, {c.year})" for c in candidate.certifications]
        sections.append("Certifications: " + "; ".join(cert_names))
    
    # 7. Key behavioral signals — embedded as text so the retriever can factor them in
    signals = candidate.redrob_signals
    context_parts = []
    context_parts.append(f"{p.years_of_experience} years experience")
    context_parts.append(f"Location: {p.location}, {p.country}")
    context_parts.append(f"Work mode: {signals.preferred_work_mode}")
    if signals.open_to_work_flag:
        context_parts.append("Open to work")
    if signals.github_activity_score > 0:
        context_parts.append(f"GitHub activity: {signals.github_activity_score}/100")
    context_parts.append(f"Notice period: {signals.notice_period_days} days")
    sections.append(" | ".join(context_parts))
    
    return "\n".join(sections)


def build_jd_text_for_embedding(jd_raw_text: str) -> str:
    """
    Build a focused JD text for bi-encoder embedding.
    
    The raw JD is ~9500 chars / ~2000+ tokens — far beyond the bi-encoder's
    256-token window. If we pass the full text, the model only sees the preamble
    (company culture blurb) and misses the actual technical requirements.
    
    We extract and concatenate the sections that matter most for semantic
    matching: the core role description, technical requirements, and ideal
    candidate profile.
    
    Args:
        jd_raw_text: The full JD text.
        
    Returns:
        A focused text string (~500-800 chars) for embedding.
    """
    # Extract the most semantically relevant sections
    focused_parts = []
    
    # The role mandate (what the person actually does)
    mandate_markers = [
        ("the high-level mandate", "in practical terms"),
        ("what you'd actually be doing", "what we mean by"),
    ]
    
    text_lower = jd_raw_text.lower()
    
    for start_marker, end_marker in mandate_markers:
        start = text_lower.find(start_marker)
        if start != -1:
            end = text_lower.find(end_marker, start)
            if end == -1:
                end = start + 500
            focused_parts.append(jd_raw_text[start:end].strip())
            break
    
    # Core requirements section
    req_start = text_lower.find("things you absolutely need")
    if req_start != -1:
        req_end = text_lower.find("things we'd like you to have", req_start)
        if req_end == -1:
            req_end = req_start + 800
        focused_parts.append(jd_raw_text[req_start:req_end].strip())
    
    # Ideal candidate section (compressed signal)
    ideal_start = text_lower.find("the \"ideal candidate\" we're imagining")
    if ideal_start == -1:
        ideal_start = text_lower.find("how to read between the lines")
    if ideal_start != -1:
        ideal_end = text_lower.find("we are aware this is a narrow profile", ideal_start)
        if ideal_end == -1:
            ideal_end = ideal_start + 600
        focused_parts.append(jd_raw_text[ideal_start:ideal_end].strip())
    
    if focused_parts:
        return "\n".join(focused_parts)
    
    # Fallback: return first 800 chars if section parsing fails
    return jd_raw_text[:800]
