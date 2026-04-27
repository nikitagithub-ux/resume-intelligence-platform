# ─────────────────────────────────────────────
#  core/features.py
#  Shared feature extraction — used by BOTH train.py and app.py
#  This is the most critical file in the project.
#  Any change here affects both training and inference equally.
# ─────────────────────────────────────────────

import re
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    SKILL_TAXONOMY, DOMAIN_KEYWORDS, EXPERIENCE_PATTERNS,
    DOMAIN_PARTIAL_CREDIT, SENIORITY_MAP, DOMAIN_MAP, FEATURE_COLUMNS
)

logger = logging.getLogger(__name__)


# ── Skill extraction ───────────────────────────

def extract_skills(text: str) -> list:
    text_lower = text.lower()
    found = set()
    sorted_terms = sorted(SKILL_TAXONOMY.keys(), key=len, reverse=True)
    for term in sorted_terms:
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text_lower):
            found.add(SKILL_TAXONOMY[term])
    return sorted(list(found))


# ── Experience extraction ──────────────────────

def extract_experience(text: str) -> float:
    text_lower = text.lower()
    for pattern in EXPERIENCE_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            return min(float(match.group(1)), 20.0)
    year_spans = re.findall(r'\b(20\d{2}|19\d{2})\b.*?\b(20\d{2}|19\d{2})\b', text_lower)
    if year_spans:
        durations = [int(e) - int(s) for s, e in year_spans if 0 < int(e) - int(s) <= 20]
        if durations:
            return float(min(max(durations), 20))
    return 0.0


# ── Domain classification ──────────────────────

def classify_domain(text: str, skills: list) -> str:
    text_lower = text.lower()
    scores = {domain: 0.0 for domain in DOMAIN_KEYWORDS}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[domain] += 1
        for skill in skills:
            if skill in keywords:
                scores[domain] += 1.5
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ── Scoring helpers ────────────────────────────

def compute_skill_overlap(resume_skills: list, job_required: list) -> float:
    if not job_required:
        return 0.0
    resume_set = set(resume_skills)
    matched = sum(1 for s in job_required if s in resume_set)
    return round(matched / len(job_required), 4)


def compute_nice_to_have(resume_skills: list, job_nice: list) -> float:
    if not job_nice:
        return 0.0
    resume_set = set(resume_skills)
    matched = sum(1 for s in job_nice if s in resume_set)
    return round(matched / len(job_nice), 4)


def compute_domain_score(resume_domain: str, job_domain: str) -> tuple:
    if resume_domain == job_domain:
        return 1, 1.0
    partial = DOMAIN_PARTIAL_CREDIT.get((resume_domain, job_domain), 0.0)
    return (1 if partial > 0 else 0), round(partial, 4)


def compute_experience_score(resume_exp: float, job_min: float, job_max: float) -> tuple:
    gap = round(resume_exp - job_min, 2)
    if job_min <= resume_exp <= job_max:
        score = 1.0
    elif resume_exp < job_min:
        shortfall = job_min - resume_exp
        score = max(0.0, 1.0 - (shortfall / max(job_min, 1)))
    else:
        excess = resume_exp - job_max
        score = max(0.5, 1.0 - (excess / 10))
    return gap, round(score, 4)


def compute_seniority(resume_exp: float, job_min: float, job_max: float) -> str:
    if resume_exp < job_min:
        return "underqualified"
    elif resume_exp > job_max:
        return "overqualified"
    return "fit"


# ── Main feature builder ───────────────────────

def build_feature_vector(
    resume_skills: list,
    resume_experience: float,
    resume_domain: str,
    job_required_skills: list,
    job_nice_to_have: list,
    job_min_experience: float,
    job_max_experience: float,
    job_domain: str,
) -> dict:
    """
    Builds the exact feature vector the model expects.
    Called at inference time from app.py.
    Output keys must match FEATURE_COLUMNS in config.py exactly.
    """
    overlap         = compute_skill_overlap(resume_skills, job_required_skills)
    nice            = compute_nice_to_have(resume_skills, job_nice_to_have)
    dom_match, dom_score = compute_domain_score(resume_domain, job_domain)
    exp_gap, exp_score   = compute_experience_score(resume_experience, job_min_experience, job_max_experience)
    seniority       = compute_seniority(resume_experience, job_min_experience, job_max_experience)

    resume_set      = set(resume_skills)
    missing_skills  = [s for s in job_required_skills if s not in resume_set]
    matched_skills  = [s for s in job_required_skills if s in resume_set]

    # Encoded categoricals — must use same maps as training
    seniority_enc   = SENIORITY_MAP.get(seniority, 1)
    resume_dom_enc  = DOMAIN_MAP.get(resume_domain, 0)
    job_dom_enc     = DOMAIN_MAP.get(job_domain, 0)

    # Profile similarity — loads from ideal_profiles.json at inference time
    try:
        from core.profiles import compare_to_ideal
        profile_data = compare_to_ideal(resume_skills, resume_experience, job_domain)
        profile_sim  = profile_data.get("profile_similarity", 0.0)
    except Exception:
        profile_sim = 0.0

    feature_vector = {
        "skill_overlap_score":      overlap,
        "nice_to_have_score":       nice,
        "skill_gap_count":          len(missing_skills),
        "domain_match":             dom_match,
        "domain_score":             dom_score,
        "experience_gap":           exp_gap,
        "experience_score":         exp_score,
        "resume_skill_count":       len(resume_skills),
        "job_total_required":       len(job_required_skills),
        "seniority_fit_encoded":    seniority_enc,
        "resume_domain_encoded":    resume_dom_enc,
        "job_domain_encoded":       job_dom_enc,
        "profile_similarity_score": profile_sim,
    }

    # Verify all expected columns are present
    missing_cols = [c for c in FEATURE_COLUMNS if c not in feature_vector]
    if missing_cols:
        raise ValueError(f"Feature vector missing columns: {missing_cols}")

    # Also return interpretable fields for feedback engine
    extras = {
        "matched_skills":   matched_skills,
        "missing_skills":   missing_skills,
        "seniority_fit":    seniority,
        "resume_domain":    resume_domain,
        "job_domain":       job_domain,
        "resume_experience": resume_experience,
    }

    return feature_vector, extras


if __name__ == "__main__":
    # Quick smoke test
    vec, extras = build_feature_vector(
        resume_skills=["python", "sql", "docker", "api"],
        resume_experience=3.0,
        resume_domain="backend",
        job_required_skills=["python", "django", "api", "sql"],
        job_nice_to_have=["docker", "aws"],
        job_min_experience=0,
        job_max_experience=3,
        job_domain="backend",
    )
    print("Feature vector:")
    for k, v in vec.items():
        print(f"  {k}: {v}")
    print("\nExtras:")
    for k, v in extras.items():
        print(f"  {k}: {v}")