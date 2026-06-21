import json
from src.pipeline import RecruiterPipeline
from config import CANDIDATES_JSONL_PATH

# Load candidates
candidates = []
with open(CANDIDATES_JSONL_PATH, 'r') as f:
    for line in f:
        candidates.append(json.loads(line))
        
lookup = {c['candidate_id']: c for c in candidates}

pipeline = RecruiterPipeline()

def evaluate_weights(ce, skill, career, behavioral):
    import src.pipeline as pl
    pl.CROSS_ENCODER_WEIGHT = ce
    pl.SKILL_MATCH_WEIGHT = skill
    pl.CAREER_FIT_WEIGHT = career
    pl.BEHAVIORAL_WEIGHT = behavioral
    
    # Run the pipeline (we already loaded data, so we can just call run)
    rows = pipeline.run(CANDIDATES_JSONL_PATH, "output/tuning_submission.csv")
    
    print(f"Weights: CE={ce:.2f}, SK={skill:.2f}, CA={career:.2f}, BE={behavioral:.2f}")
    
    for rank, row in enumerate(rows[:5], 1):
        cid = row.candidate_id
        c = lookup[cid]
        p = c['profile']
        s = c['redrob_signals']
        print(f"  #{rank} {cid} - {p['current_title']} @ {p['current_company']} ({p['years_of_experience']}y)")
        print(f"      Response: {s['recruiter_response_rate']:.0%}, Open: {s['open_to_work_flag']}, Notice: {s['notice_period_days']}")
    print("-" * 50)

# Default
evaluate_weights(0.40, 0.20, 0.20, 0.20)

# Increase Behavioral to punish low response rates more
evaluate_weights(0.35, 0.20, 0.20, 0.25)

# Increase Career to punish over-experienced candidates more
evaluate_weights(0.35, 0.20, 0.25, 0.20)

# Both
evaluate_weights(0.30, 0.20, 0.25, 0.25)
