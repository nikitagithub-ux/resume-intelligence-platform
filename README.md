# Resume Intelligence Platform

An end-to-end hiring intelligence system that predicts a candidate's likelihood of being shortlisted for a specific job and provides actionable feedback.

## What it does
- Upload a resume (PDF or DOCX) and select a job role
- Returns a shortlisting confidence score based on skill overlap, experience fit, and domain alignment
- Compares the candidate against ideal profiles of previously successful candidates
- Generates personalised feedback using Groq (LLaMA 3.3) — missing skills, strengths, and improvement suggestions

## Tech Stack
- **Backend:** FastAPI, Python
- **ML Model:** XGBoost (91% accuracy, 0.93 AUC) trained on 37,000+ resume-job pairs
- **LLM Feedback:** Groq API (LLaMA 3.3 70B)
- **Data Pipeline:** Custom pipeline parsing 460+ real resumes across 80 job roles

## Project Structure
├── api/          # FastAPI endpoints
├── core/         # Parser, feature extraction, feedback engine
├── ml/           # Model training
├── data/         # Jobs data and ideal profiles
├── frontend/     # HTML/CSS/JS interface

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Set environment variable: `GROQ_API_KEY=your_key`
3. Train model: `python ml/train.py`
4. Run API: `uvicorn api.app:app --reload`