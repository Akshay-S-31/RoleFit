"""
RoleFit - Rule-Based Scorer (Stage 3)

Encodes the JD's explicit logic as deterministic scoring rules.
Replaces LLM-as-a-Judge to meet the 5-min/no-network compute constraint.

Architecture:
- SkillMatcher: weighted skill scoring (required vs nice-to-have, depth not just presence)
- CareerFitScorer: career trajectory analysis + JD disqualifiers
- BehavioralScorer: converts 23 redrob signals into a 0-1 availability/engagement score
- HoneypotDetector: flags impossible profiles before any scoring

Design principles:
- All scorers return a float in [0.0, 1.0] — clean for the weighted composite
- All logic is deterministic and fast — no model calls, pure Python
- Every rule is traceable to a specific sentence in the JD
- Weights live in config.py so they can be tuned without touching this file
- The scorer never mutates the Candidate object — only writes to RankedCandidate fields

Future impact:
- These scores combine with the cross-encoder score in pipeline.py (Day 4)
- Weight tuning is the primary lever to improve final ranking quality
- Honeypot detection must run first — flagged candidates are excluded entirely
"""

import logging
from datetime import date, datetime
from typing import List, Set, Tuple

from src.models import Candidate, RankedCandidate

logger = logging.getLogger(__name__)


# =============================================================================
# JD-derived skill taxonomy
# Source: JD lines 33-43 ("Things you absolutely need" / "Things we'd like you to have")
# =============================================================================

# Must-have: production experience with these domains
# JD says: embeddings/retrieval, vector DBs, Python, evaluation frameworks
REQUIRED_SKILL_KEYWORDS = {
    # Embeddings & retrieval (core requirement #1)
    "embeddings", "sentence-transformers", "sentence transformers",
    "bi-encoder", "dense retrieval", "semantic search", "bge", "e5",
    # Vector databases (core requirement #2)
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "pgvector", "vector search", "vector database",
    # Retrieval & ranking systems
    "information retrieval", "ranking", "recommendation systems",
    "recommendation", "search", "retrieval",
    # Evaluation frameworks (core requirement #4)
    "ndcg", "mrr", "map", "a/b testing", "evaluation", "offline evaluation",
    # ML fundamentals (required depth signal)
    "machine learning", "deep learning", "nlp", "natural language processing",
    "pytorch", "tensorflow", "transformers", "hugging face", "hugging face transformers",
}

# Nice-to-have: bonus points but not required
# JD says: LLM fine-tuning, learning-to-rank, HR-tech, distributed systems, open-source
NICE_TO_HAVE_SKILL_KEYWORDS = {
    # LLM fine-tuning (nice-to-have #1)
    "lora", "qlora", "peft", "fine-tuning", "fine-tuning llms", "llm fine-tuning",
    # LLMs generally (useful but not core)
    "llms", "gpt", "llm", "large language model", "rag", "langchain",
    "llamaindex", "llaindex",
    # Learning-to-rank (nice-to-have #2)
    "learning to rank", "xgboost", "lightgbm",
    # MLOps (shows production mindset)
    "mlops", "mlflow", "bento", "bentoml", "serving", "deployment",
    # Related fields
    "knowledge graph", "graph neural", "reranking", "cross-encoder",
}

# Red flags: primary expertise in unrelated domains (JD lines 49-50)
# "People whose primary expertise is computer vision, speech, or robotics
#  without significant NLP/IR exposure"
CV_SPEECH_KEYWORDS = {
    "computer vision", "image classification", "object detection",
    "speech recognition", "speech synthesis", "tts", "asr",
    "robotics", "ros", "slam", "image segmentation", "gan", "gans",
    "diffusion models", "stable diffusion",
}

# Consulting-only companies (JD line 48)
CONSULTING_COMPANIES: Set[str] = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree", "mphasis", "ltimindtree",
    "l&t infotech", "persistent systems", "hexaware", "cyient",
    "dxc technology", "unisys", "kyndryl",
}

# Product companies that are a strong positive signal for this JD
STRONG_PRODUCT_COMPANIES: Set[str] = {
    "google", "meta", "microsoft", "amazon", "apple", "netflix",
    "openai", "anthropic", "deepmind", "hugging face", "cohere",
    "flipkart", "zomato", "swiggy", "razorpay", "cred", "meesho",
    "phonepe", "paytm", "ola", "byju", "dunzo", "zepto", "blinkit",
    "sarvam", "krutrim", "linkedin", "salesforce", "stripe", "airbnb",
    "uber", "lyft", "twitter", "x", "instacart", "doordash",
    "freshworks", "chargebee", "postman", "browserstack", "zoho",
}


# =============================================================================
# SkillMatcher
# =============================================================================

class SkillMatcher:
    """
    Scores a candidate's skills against the JD's requirements.

    Scoring logic:
    - Required skills: each match contributes based on proficiency depth
    - Nice-to-have: smaller bonus per match
    - CV/speech primary expertise without NLP/IR: penalty
    - Skill depth (proficiency level + duration) multiplies the match score

    Returns a float in [0.0, 1.0].
    """

    # Proficiency multipliers — depth matters, not just name presence
    PROFICIENCY_WEIGHTS = {
        "expert": 1.0,
        "advanced": 0.85,
        "intermediate": 0.65,
        "beginner": 0.35,
    }

    def score(self, candidate: Candidate) -> float:
        """
        Compute skill match score.

        Args:
            candidate: The candidate to score.

        Returns:
            float in [0.0, 1.0]
        """
        required_score = 0.0
        nice_score = 0.0
        cv_speech_count = 0
        nlp_ir_count = 0

        for skill in candidate.skills:
            skill_lower = skill.name.lower()

            # Proficiency multiplier
            prof_weight = self.PROFICIENCY_WEIGHTS.get(skill.proficiency, 0.5)

            # Duration bonus: longer usage = more confidence in depth
            duration_bonus = 0.0
            if skill.duration_months and skill.duration_months > 0:
                # Cap at 36 months (3 years) — beyond that diminishing returns
                duration_bonus = min(skill.duration_months / 36.0, 1.0) * 0.2

            skill_weight = prof_weight + duration_bonus

            # Check against taxonomy
            matched_required = any(kw in skill_lower for kw in REQUIRED_SKILL_KEYWORDS)
            matched_nice = any(kw in skill_lower for kw in NICE_TO_HAVE_SKILL_KEYWORDS)
            matched_cv_speech = any(kw in skill_lower for kw in CV_SPEECH_KEYWORDS)

            if matched_required:
                required_score += skill_weight
                nlp_ir_count += 1
            elif matched_nice:
                nice_score += skill_weight * 0.4  # Nice-to-have worth 40% of required

            if matched_cv_speech:
                cv_speech_count += 1

        # Normalize required score: 5 strong required skills = full score
        # (most good candidates will hit 4-8 required skills)
        required_norm = min(required_score / 6.0, 1.0)

        # Nice-to-have adds up to 20% bonus on top
        nice_norm = min(nice_score / 3.0, 1.0) * 0.2

        # Penalty: primary expertise is CV/speech without NLP/IR balance
        # JD: "you'd be re-learning fundamentals here"
        cv_penalty = 0.0
        if cv_speech_count > 3 and nlp_ir_count < 2:
            cv_penalty = 0.25  # Meaningful penalty for pure CV/speech profiles

        raw = required_norm + nice_norm - cv_penalty
        return max(0.0, min(1.0, raw))


# =============================================================================
# CareerFitScorer
# =============================================================================

class CareerFitScorer:
    """
    Scores career trajectory against JD requirements.

    Evaluates:
    1. Consulting-only career (hard disqualifier from JD line 48)
    2. Experience years vs ideal range (5-9 years)
    3. Title-chaser detection (rapid company hopping for title bumps, JD line 46)
    4. Product company experience (positive signal vs services-only)
    5. Current title alignment (JD wants ML/AI/Search engineer, not Marketing Manager)

    Returns a float in [0.0, 1.0].
    """

    # Titles that signal genuine ML/AI/Search engineering role fit
    STRONG_TITLE_KEYWORDS = {
        "machine learning", "ml engineer", "ai engineer", "applied scientist",
        "data scientist", "nlp engineer", "search engineer", "ranking",
        "recommendation", "research scientist", "applied ml", "research engineer",
        "staff engineer", "principal engineer", "senior engineer",
    }

    # Experience ideal range (from JD)
    IDEAL_MIN = 5.0
    IDEAL_MAX = 9.0

    def score(self, candidate: Candidate) -> Tuple[float, str]:
        """
        Compute career fit score.

        Args:
            candidate: The candidate to score.

        Returns:
            Tuple of (score: float in [0.0, 1.0], reason: str for debugging)
        """
        career = candidate.career_history
        profile = candidate.profile
        reasons = []

        # --- 1. Consulting-only career check ---
        all_companies = {e.company.lower() for e in career}
        is_consulting_only = all(
            any(c in company for c in CONSULTING_COMPANIES)
            for company in all_companies
        )
        if is_consulting_only and len(all_companies) > 0:
            # JD explicitly says "only worked at consulting firms = not a fit"
            # But "if currently at one but has prior product experience = fine"
            return 0.10, "consulting-only career"

        # --- 2. Experience range scoring ---
        yoe = profile.years_of_experience
        if yoe < 2.0:
            exp_score = 0.2  # Too junior
        elif yoe < self.IDEAL_MIN:
            # Scale from 0.2 to 0.8 as they approach ideal min
            exp_score = 0.2 + 0.6 * ((yoe - 2.0) / (self.IDEAL_MIN - 2.0))
        elif yoe <= self.IDEAL_MAX:
            exp_score = 1.0  # In the sweet spot
        elif yoe <= 12.0:
            # Slightly above ideal — JD says "we'll consider outside the band"
            exp_score = 0.85
        else:
            # 12+ years — likely over-experienced or title-chased
            exp_score = 0.65

        # --- 3. Title-chaser detection ---
        # JD: "switching companies every 1.5 years for title bumps"
        title_chaser_penalty = 0.0
        if len(career) >= 3:
            short_tenures = sum(
                1 for e in career
                if e.duration_months < 18 and not e.is_current
            )
            if short_tenures >= 3:
                title_chaser_penalty = 0.20
                reasons.append("title-chaser pattern")

        # --- 4. Product company bonus ---
        product_bonus = 0.0
        has_product_exp = any(
            any(pc in company for pc in STRONG_PRODUCT_COMPANIES)
            for company in all_companies
        )
        if has_product_exp:
            product_bonus = 0.10

        # --- 5. Current title alignment ---
        title_lower = profile.current_title.lower()
        title_score = 0.0
        if any(kw in title_lower for kw in self.STRONG_TITLE_KEYWORDS):
            title_score = 0.15
        elif any(bad in title_lower for bad in [
            "marketing", "accountant", "hr manager", "operations",
            "customer support", "sales", "graphic designer", "civil engineer",
            "mechanical engineer", "content writer",
        ]):
            title_score = -0.10  # Wrong domain entirely
            reasons.append("non-ML current title")

        # --- Composite ---
        # Base: 60% experience + 25% title + product bonus − penalties
        raw = (exp_score * 0.60) + (title_score) + product_bonus - title_chaser_penalty
        final = max(0.0, min(1.0, raw))

        reason_str = ", ".join(reasons) if reasons else "ok"
        return final, reason_str


# =============================================================================
# BehavioralScorer
# =============================================================================

class BehavioralScorer:
    """
    Converts the 23 redrob signals into a 0-1 availability/engagement score.

    Key insight from JD hackathon note:
    "A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5%
    recruiter response rate is, for hiring purposes, not actually available."

    This scorer is a multiplier on fit — a great candidate who is unavailable
    should rank lower than a slightly weaker candidate who is actively looking.

    Returns a float in [0.0, 1.0].
    """

    def score(self, candidate: Candidate) -> float:
        """
        Compute behavioral/availability score from redrob signals.

        Args:
            candidate: The candidate to score.

        Returns:
            float in [0.0, 1.0]
        """
        s = candidate.redrob_signals
        component_scores = []

        # --- 1. Recruiter response rate (20% weight) ---
        # JD: highest weight signal for actual reachability
        # Raw range: 0.0–1.0 → use directly
        component_scores.append(("response_rate", s.recruiter_response_rate, 0.20))

        # --- 2. Interview completion rate (15% weight) ---
        # Signals reliability and commitment once engaged
        component_scores.append(("interview_rate", s.interview_completion_rate, 0.15))

        # --- 3. GitHub activity (15% weight) ---
        # JD: "open-source contributions in the AI/ML space" = nice-to-have
        # -1 means no GitHub linked (neutral, not negative)
        if s.github_activity_score < 0:
            github_norm = 0.3  # No GitHub is mildly negative but not zero
        else:
            github_norm = s.github_activity_score / 100.0
        component_scores.append(("github", github_norm, 0.15))

        # --- 4. Recency (10% weight) ---
        # Last active date — stale profiles are unreachable
        recency = self._recency_score(s.last_active_date)
        component_scores.append(("recency", recency, 0.10))

        # --- 5. Profile completeness (10% weight) ---
        completeness_norm = s.profile_completeness_score / 100.0
        component_scores.append(("completeness", completeness_norm, 0.10))

        # --- 6. Notice period (10% weight) ---
        # JD: "sub-30 days ideal, 30+ days bar gets higher"
        notice = self._notice_score(s.notice_period_days)
        component_scores.append(("notice", notice, 0.10))

        # --- 7. Open to work flag (5% weight) ---
        open_score = 1.0 if s.open_to_work_flag else 0.3
        component_scores.append(("open_to_work", open_score, 0.05))

        # --- 8. Verified identity (5% weight) ---
        # email + phone + linkedin — signals real, contactable person
        verified_count = sum([s.verified_email, s.verified_phone, s.linkedin_connected])
        verified_norm = verified_count / 3.0
        component_scores.append(("verified", verified_norm, 0.05))

        # --- 9. Offer acceptance rate (5% weight) ---
        # -1 means no history — treat as neutral
        if s.offer_acceptance_rate < 0:
            offer_norm = 0.5
        else:
            offer_norm = s.offer_acceptance_rate
        component_scores.append(("offer_acceptance", offer_norm, 0.05))

        # --- 10. Saved by recruiters 30d (5% weight) ---
        # Market validation — other recruiters find this profile interesting
        # Normalize: 0 = 0.0, 5+ = 1.0
        saved_norm = min(s.saved_by_recruiters_30d / 5.0, 1.0)
        component_scores.append(("saved_30d", saved_norm, 0.05))

        # Weighted sum
        total = sum(val * weight for _, val, weight in component_scores)
        return max(0.0, min(1.0, total))

    def _recency_score(self, last_active_date: str) -> float:
        """
        Score recency based on days since last activity.

        Thresholds derived from the JD's note about inactive candidates:
        - Active in last 30 days: 1.0
        - 30-90 days: gradual decay
        - 6 months+ (180 days): minimum score (not zero — they might still respond)
        """
        try:
            last_active = datetime.strptime(last_active_date, "%Y-%m-%d").date()
            today = date.today()
            days_inactive = (today - last_active).days

            if days_inactive <= 30:
                return 1.0
            elif days_inactive <= 90:
                return 1.0 - 0.4 * ((days_inactive - 30) / 60.0)
            elif days_inactive <= 180:
                return 0.6 - 0.3 * ((days_inactive - 90) / 90.0)
            else:
                return 0.2  # Very stale — probably not looking

        except (ValueError, TypeError):
            return 0.5  # Unknown date — neutral

    def _notice_score(self, notice_days: int) -> float:
        """
        Score notice period.

        JD: "sub-30 days = ideal, we can buy out up to 30 days,
             30+ days bar gets higher"
        """
        if notice_days <= 0:
            return 1.0  # Immediately available
        elif notice_days <= 30:
            return 1.0  # JD says "we can buy this out"
        elif notice_days <= 60:
            return 0.75  # Acceptable but higher bar
        elif notice_days <= 90:
            return 0.50
        else:
            return 0.25  # 90+ days is a real friction point


# =============================================================================
# HoneypotDetector
# =============================================================================

class HoneypotDetector:
    """
    Detects candidates with impossible or internally inconsistent profiles.

    The dataset contains ~80 honeypots. Ranking >10 in the top 100 = disqualification.
    A good ranking system should naturally avoid them; we make this explicit.

    Honeypot patterns from submission_spec:
    - Expert in many skills with 0 months duration
    - Career timeline inconsistencies (worked at company before it was founded)
    - Impossible experience claims

    Returns: bool (True = likely honeypot, should be excluded)
    """

    # Threshold: more than this many expert skills with 0 duration = suspicious
    MAX_EXPERT_ZERO_DURATION = 3

    def is_honeypot(self, candidate: Candidate) -> Tuple[bool, str]:
        """
        Check whether a candidate has an impossible profile.

        Args:
            candidate: The candidate to inspect.

        Returns:
            Tuple of (is_honeypot: bool, reason: str)
        """
        reasons = []

        # --- Check 1: Many expert skills with 0 duration ---
        # Real experts accumulate months of usage. 0 months + expert = suspicious.
        expert_zero_dur = sum(
            1 for s in candidate.skills
            if s.proficiency == "expert"
            and (s.duration_months is None or s.duration_months == 0)
        )
        if expert_zero_dur > self.MAX_EXPERT_ZERO_DURATION:
            reasons.append(
                f"{expert_zero_dur} expert skills with 0 months duration"
            )

        # --- Check 2: Total claimed skill-months vs total experience ---
        # If sum of all skill durations >> total career months, something is off
        total_career_months = sum(e.duration_months for e in candidate.career_history)
        total_skill_months = sum(
            (s.duration_months or 0) for s in candidate.skills
        )
        # A person can use multiple skills simultaneously, so skill-months > career-months
        # is normal. But 10x more skill-months than career months is suspicious.
        if total_career_months > 0 and total_skill_months > total_career_months * 10:
            reasons.append(
                f"skill duration sum ({total_skill_months}mo) >> "
                f"career duration ({total_career_months}mo)"
            )

        # --- Check 3: Career start/end date sanity ---
        # Check for future end dates (role ended before it started, etc.)
        for entry in candidate.career_history:
            if entry.end_date and entry.start_date:
                try:
                    start = datetime.strptime(entry.start_date, "%Y-%m-%d").date()
                    end = datetime.strptime(entry.end_date, "%Y-%m-%d").date()
                    if end < start:
                        reasons.append(
                            f"role at {entry.company} ends before it starts"
                        )
                except (ValueError, TypeError):
                    pass

        # --- Check 4: Extreme skill count with too-short career ---
        # 20+ expert skills with < 2 years experience is implausible
        expert_count = sum(1 for s in candidate.skills if s.proficiency == "expert")
        if expert_count >= 15 and candidate.profile.years_of_experience < 3.0:
            reasons.append(
                f"{expert_count} expert skills with only "
                f"{candidate.profile.years_of_experience}y experience"
            )

        is_hp = len(reasons) > 0
        reason_str = "; ".join(reasons) if reasons else "clean"
        return is_hp, reason_str


# =============================================================================
# Public API — score_candidate()
# =============================================================================

# Module-level singleton instances — instantiated once, reused for all candidates
_skill_matcher = SkillMatcher()
_career_scorer = CareerFitScorer()
_behavioral_scorer = BehavioralScorer()
_honeypot_detector = HoneypotDetector()


def score_candidate(rc: RankedCandidate) -> RankedCandidate:
    """
    Apply all rule-based scorers to a RankedCandidate.

    Mutates and returns the same object with scoring fields populated:
    - rc.is_honeypot
    - rc.skill_match_score
    - rc.career_fit_score
    - rc.behavioral_score

    Note: rc.final_score and rc.reasoning are NOT set here —
    that's done in pipeline.py where cross-encoder score is also available.

    Args:
        rc: A RankedCandidate (with candidate data + vector/cross-encoder scores)

    Returns:
        The same RankedCandidate with rule-based scores populated.
    """
    candidate = rc.candidate

    # 1. Honeypot check first — flag and short-circuit
    is_hp, hp_reason = _honeypot_detector.is_honeypot(candidate)
    rc.is_honeypot = is_hp
    if is_hp:
        logger.debug(f"Honeypot detected: {candidate.candidate_id} — {hp_reason}")
        rc.skill_match_score = 0.0
        rc.career_fit_score = 0.0
        rc.behavioral_score = 0.0
        return rc

    # 2. Skill matching
    rc.skill_match_score = _skill_matcher.score(candidate)

    # 3. Career fit
    career_score, career_reason = _career_scorer.score(candidate)
    rc.career_fit_score = career_score

    # 4. Behavioral signals
    rc.behavioral_score = _behavioral_scorer.score(candidate)

    return rc
