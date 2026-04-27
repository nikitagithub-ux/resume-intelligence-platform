# ─────────────────────────────────────────────
#  core/profiles.py
#  Loads ideal_profiles.json and compares a candidate
#  against the profile of successful candidates for a job domain.
#  Powers the "how you compare" section of your platform.
# ─────────────────────────────────────────────

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROFILES_PATH

logger = logging.getLogger(__name__)

# Load profiles once at import time — no need to reload per request
_profiles = None

def get_profiles() -> dict:
    global _profiles
    if _profiles is None:
        try:
            with open(PROFILES_PATH) as f:
                _profiles = json.load(f)
            logger.info(f"Loaded ideal profiles for {len(_profiles)} domains")
        except FileNotFoundError:
            logger.error(f"ideal_profiles.json not found at {PROFILES_PATH}")
            _profiles = {}
    return _profiles


def compare_to_ideal(
    resume_skills: list,
    resume_experience: float,
    job_domain: str,
) -> dict:
    """
    Compares a candidate against the ideal profile for their target job domain.
    Returns structured data used by both the API response and the feedback engine.
    """
    profiles = get_profiles()
    profile  = profiles.get(job_domain)

    if not profile:
        return {
            "profile_found":        False,
            "job_domain":           job_domain,
            "skills_you_have":      [],
            "skills_to_add":        [],
            "profile_similarity":   0.0,
            "your_experience":      resume_experience,
            "typical_experience":   None,
            "experience_range":     None,
            "based_on_n_candidates": 0,
        }

    resume_set  = set(resume_skills)
    top_skills  = set(profile.get("top_skills", []))

    matched     = sorted(list(resume_set & top_skills))
    missing     = sorted(list(top_skills - resume_set))

    # Profile similarity score — skill match weighted more than experience
    skill_sim   = len(matched) / len(top_skills) if top_skills else 0.0
    exp_typical = profile.get("avg_experience", 5.0)
    exp_diff    = abs(resume_experience - exp_typical)
    exp_sim     = max(0.0, 1.0 - (exp_diff / max(exp_typical, 1)))
    similarity  = round(0.6 * skill_sim + 0.4 * exp_sim, 4)

    exp_range   = profile.get("experience_range", {})

    return {
        "profile_found":            True,
        "job_domain":               job_domain,
        "skills_you_have":          matched,
        "skills_to_add":            missing,
        "profile_similarity":       similarity,
        "your_experience":          resume_experience,
        "typical_experience":       round(exp_typical, 1),
        "experience_range":         exp_range,
        "min_viable_overlap":       profile.get("min_viable_overlap", 0.0),
        "avg_hiring_score":         profile.get("avg_hiring_score", 0.0),
        "based_on_n_candidates":    profile.get("candidate_count", 0),
        "top_skills_for_domain":    profile.get("top_skills", []),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = compare_to_ideal(
        resume_skills=["python", "sql", "docker"],
        resume_experience=3.0,
        job_domain="backend",
    )
    import json
    print(json.dumps(result, indent=2))
