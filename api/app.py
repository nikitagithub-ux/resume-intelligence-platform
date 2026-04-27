# ─────────────────────────────────────────────
#  api/app.py
#  FastAPI backend — all endpoints
#  Run: uvicorn api.app:app --reload
# ─────────────────────────────────────────────

import sys
import os
import logging
import importlib.util
import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODEL_PATH, JOBS_FILE, TEMP_DIR, FEATURE_COLUMNS
from core.parser import parse_resume
from core.features import build_feature_vector
from core.profiles import compare_to_ideal
from core.feedback import generate_feedback

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Resume Intelligence API", version="1.0")

# Allow frontend (HTML/JS) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(TEMP_DIR, exist_ok=True)


# ── Load model ─────────────────────────────────
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
        return model
    except FileNotFoundError:
        logger.error(f"model.pkl not found at {MODEL_PATH}. Run ml/train.py first.")
        return None

model = load_model()


# ── Load jobs ──────────────────────────────────
def load_jobs():
    try:
        spec   = importlib.util.spec_from_file_location("jobs_data", JOBS_FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.JOBS
    except Exception as e:
        logger.error(f"Failed to load jobs: {e}")
        return []

JOBS = load_jobs()
logger.info(f"Loaded {len(JOBS)} jobs")


# ── Helper ─────────────────────────────────────
def find_job(job_id: str) -> dict:
    """Finds job by id (int) or title (string, case-insensitive)."""
    for j in JOBS:
        if str(j.get("id")) == job_id:
            return j
        if j.get("title", "").lower() == job_id.lower():
            return j
    return None


# ── Routes ─────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Resume Intelligence API is running"}


@app.get("/jobs")
def get_jobs():
    """Returns all jobs for the frontend dropdown."""
    return [
        {
            "id":     j["id"],
            "title":  j["title"],
            "domain": j["domain"],
        }
        for j in JOBS
    ]


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    job_id: str = Form(...),
):
    """
    Main prediction endpoint.
    Accepts a resume file + job_id, returns confidence score + full feedback.
    """
    # ── 1. Save uploaded file temporarily ─────
    ext       = Path(file.filename).suffix.lower()
    temp_path = os.path.join(TEMP_DIR, f"upload_{file.filename}")

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # ── 2. Validate file type ──────────────
        if ext not in (".pdf", ".docx"):
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

        # ── 3. Find job ────────────────────────
        job = find_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

        # ── 4. Parse resume ────────────────────
        parsed = parse_resume(temp_path)
        if not parsed:
            raise HTTPException(status_code=422, detail="Could not extract text from resume. Ensure it is not a scanned image.")

        # ── 5. Build feature vector ────────────
        feature_vector, extras = build_feature_vector(
            resume_skills=parsed["skills"],
            resume_experience=parsed["experience"],
            resume_domain=parsed["domain"],
            job_required_skills=job.get("required_skills", []),
            job_nice_to_have=job.get("nice_to_have", []),
            job_min_experience=job.get("min_experience", 0),
            job_max_experience=job.get("max_experience", 10),
            job_domain=job["domain"],
        )

        # ── 6. Model prediction ────────────────
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Run ml/train.py first.")

        X        = pd.DataFrame([feature_vector])[FEATURE_COLUMNS]
        prob     = float(model.predict_proba(X)[0][1])
        prediction = int(prob >= 0.5)

        # ── 7. Profile comparison ──────────────
        profile_comparison = compare_to_ideal(
            resume_skills=parsed["skills"],
            resume_experience=parsed["experience"],
            job_domain=job["domain"],
        )

        # ── 8. Gemini feedback ─────────────────
        feedback = generate_feedback(
            job_title=job["title"],
            job_domain=job["domain"],
            confidence=prob,
            matched_skills=extras["matched_skills"],
            missing_skills=extras["missing_skills"],
            resume_experience=parsed["experience"],
            job_min_exp=job.get("min_experience", 0),
            job_max_exp=job.get("max_experience", 10),
            seniority_fit=extras["seniority_fit"],
            profile_comparison=profile_comparison,
            nice_to_have_skills=job.get("nice_to_have", []),
        )

        # ── 9. Return response ─────────────────
        return {
            "status":           "success",
            "candidate_name":   parsed["name"],
            "job_title":        job["title"],
            "job_domain":       job["domain"],

            # Core result
            "confidence":       round(prob, 4),
            "confidence_pct":   round(prob * 100, 1),
            "prediction":       prediction,
            "prediction_label": "Likely Shortlisted" if prediction == 1 else "Needs Improvement",

            # Breakdown
            "skill_overlap":        feature_vector["skill_overlap_score"],
            "experience_score":     feature_vector["experience_score"],
            "domain_score":         feature_vector["domain_score"],
            "seniority_fit":        extras["seniority_fit"],
            "resume_experience":    parsed["experience"],

            # Skills
            "matched_skills":       extras["matched_skills"],
            "missing_skills":       extras["missing_skills"],
            "detected_skills":      parsed["skills"],
            "resume_domain":        parsed["domain"],

            # Profile comparison
            "profile_comparison":   profile_comparison,

            # Gemini feedback
            "feedback":             feedback,
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/health")
def health():
    return {
        "model_loaded": model is not None,
        "jobs_loaded":  len(JOBS),
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
