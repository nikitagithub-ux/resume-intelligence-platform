# ─────────────────────────────────────────────
#  core/feedback.py
#  Groq-powered feedback engine
# ─────────────────────────────────────────────

import sys
import json
import logging
from pathlib import Path

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    Groq = None
    GROQ_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

# Initialize client once at import
client = None
if GROQ_AVAILABLE and GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq client initialized successfully")
elif not GROQ_AVAILABLE:
    logger.warning("groq not installed. Run: pip install groq")
elif not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not set. Using fallback feedback.")


def _build_prompt(
    job_title, job_domain, confidence_pct, matched_skills, missing_skills,
    resume_experience, job_min_exp, job_max_exp, seniority_fit,
    profile_comparison, nice_to_have_skills,
):
    skills_to_add = profile_comparison.get("skills_to_add", [])
    similarity    = profile_comparison.get("profile_similarity", 0.0)
    n_candidates  = profile_comparison.get("based_on_n_candidates", 0)

    prompt = f"""
You are a senior hiring consultant giving honest, specific, and encouraging feedback to a job applicant.

Here is the candidate's evaluation data for the role of **{job_title}** ({job_domain} domain):

CANDIDATE SUMMARY:
- AI shortlisting confidence: {confidence_pct:.1f}%
- Experience: {resume_experience} years (job requires {job_min_exp}–{job_max_exp} years)
- Seniority fit: {seniority_fit}
- Profile similarity to successful {job_domain} candidates: {round(similarity * 100, 1)}% (based on {n_candidates} successful profiles)

SKILL ANALYSIS:
- Matched required skills: {', '.join(matched_skills) if matched_skills else 'None'}
- Missing required skills: {', '.join(missing_skills) if missing_skills else 'None — great coverage!'}
- Nice-to-have skills they could add: {', '.join(nice_to_have_skills) if nice_to_have_skills else 'None'}
- Skills top candidates in this domain have that this candidate lacks: {', '.join(skills_to_add) if skills_to_add else 'None — strong profile!'}

Based on this data, provide feedback in the following JSON format ONLY — no preamble, no markdown, just valid JSON:

{{
  "summary": "2-3 sentence honest summary of their chances and overall fit",
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "improvements": ["specific actionable improvement 1", "specific actionable improvement 2", "specific actionable improvement 3"],
  "missing_skills_advice": "1-2 sentences on which missing skills to prioritize and why",
  "experience_advice": "1 sentence on their experience relative to the role",
  "quick_wins": ["one thing they can do this week to improve their application"]
}}

Be specific, honest, and constructive. Reference actual skill names. Do not be generic.
"""
    return prompt.strip()


def generate_feedback(
    job_title, job_domain, confidence, matched_skills, missing_skills,
    resume_experience, job_min_exp, job_max_exp, seniority_fit,
    profile_comparison, nice_to_have_skills,
):
    confidence_pct = round(confidence * 100, 1)

    if not client:
        return _fallback_feedback(confidence_pct, matched_skills, missing_skills, seniority_fit)

    try:
        prompt = _build_prompt(
            job_title, job_domain, confidence_pct, matched_skills, missing_skills,
            resume_experience, job_min_exp, job_max_exp, seniority_fit,
            profile_comparison, nice_to_have_skills,
        )

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        feedback = json.loads(raw)
        feedback["source"] = "groq"
        return feedback

    except json.JSONDecodeError as e:
        logger.warning(f"Groq returned non-JSON response: {e}")
        return _fallback_feedback(confidence_pct, matched_skills, missing_skills, seniority_fit)

    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return _fallback_feedback(confidence_pct, matched_skills, missing_skills, seniority_fit)


def _fallback_feedback(confidence_pct, matched_skills, missing_skills, seniority_fit):
    if confidence_pct >= 60:
        summary = f"Strong candidate with {confidence_pct:.0f}% shortlisting likelihood. Your skill alignment is solid."
    elif confidence_pct >= 40:
        summary = f"Moderate fit with {confidence_pct:.0f}% shortlisting likelihood. Some gaps but a viable application."
    else:
        summary = f"Currently at {confidence_pct:.0f}% shortlisting likelihood. Significant gaps to address before applying."

    strengths    = [f"Matched skill: {s}" for s in matched_skills[:3]] or ["Resume submitted successfully"]
    improvements = [f"Add skill: {s}" for s in missing_skills[:3]] or ["Broaden your skill set for this role"]

    return {
        "summary":               summary,
        "strengths":             strengths,
        "improvements":          improvements,
        "missing_skills_advice": f"Focus on: {', '.join(missing_skills[:3])}" if missing_skills else "Good skill coverage.",
        "experience_advice":     f"Experience fit: {seniority_fit}.",
        "quick_wins":            ["Add missing skills to your resume and LinkedIn profile."],
        "source":                "fallback",
    }