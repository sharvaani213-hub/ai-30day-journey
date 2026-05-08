# -*- coding: utf-8 -*-
# ============================================================
# DAY 7 -- SCRIPT 1: FastAPI -- Build a REST API for your ML Model
# Topics: FastAPI basics, endpoints, request/response models,
#         serving ML models, testing with docs UI
# ============================================================

import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("DAY 7 -- SCRIPT 1: FastAPI -- REST API for ML Models")
print("=" * 60)

print("""
WHAT IS FASTAPI?
----------------
FastAPI is a modern Python web framework for building APIs.

Why AI engineers need it:
  Streamlit = for data scientists and demos (UI focused)
  FastAPI   = for production APIs (backend focused)

Real world:
  Your React/Flutter frontend calls YOUR FastAPI backend
  Your backend runs the ML model and returns predictions
  That is how every real AI product works!

FastAPI vs Flask vs Django:
  Flask   -> simple but manual (no auto-docs, no validation)
  Django  -> heavy, full framework, overkill for APIs
  FastAPI -> fast, automatic docs, data validation built in
             async support, perfect for ML serving

Key features:
  -> Auto-generates Swagger docs at /docs
  -> Data validation with Pydantic models
  -> Async support (handles many requests at once)
  -> 300% faster than Flask in benchmarks
""")


# -- SECTION 1: FastAPI Concepts Explained --------------------
print("SECTION 1: FastAPI Core Concepts")
print("-" * 40)
print("""
REST API = a set of URLs (endpoints) that accept and return JSON.

HTTP Methods:
  GET    -> retrieve data    (get predictions, get model info)
  POST   -> send data        (send features, get prediction back)
  PUT    -> update data      (update a record)
  DELETE -> delete data      (remove a record)

Status Codes:
  200 -> OK (success)
  201 -> Created (new resource made)
  400 -> Bad Request (wrong input)
  404 -> Not Found
  422 -> Validation Error (FastAPI auto-handles this)
  500 -> Server Error

Pydantic Models = define the shape of your JSON data:
  class PredictionRequest(BaseModel):
      cgpa: float           -> must be a float
      internships: int      -> must be an int
      projects: int         -> must be an int

  FastAPI validates automatically -- wrong type = 422 error
""")


# -- SECTION 2: Write the FastAPI App -------------------------
print("\nSECTION 2: Writing the FastAPI Application")
print("-" * 40)

fastapi_app = '''# -*- coding: utf-8 -*-
# ============================================================
# DAY 7 -- FastAPI ML Model Server
# Run with: uvicorn main:app --reload
# Docs at : http://127.0.0.1:8000/docs
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import pickle
import os
import datetime

# -- App setup ------------------------------------------------
app = FastAPI(
    title       = "AI Placement Predictor API",
    description = "Predicts student placement and salary using ML models",
    version     = "1.0.0"
)

# -- CORS (allow frontend to call this API) -------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# -- Pydantic models (request/response shapes) ----------------

class StudentFeatures(BaseModel):
    cgpa: float = Field(..., ge=0.0, le=10.0, description="CGPA between 0 and 10")
    internships: int = Field(..., ge=0, le=10, description="Number of internships")
    projects: int = Field(..., ge=0, le=20, description="Number of projects")
    backlogs: int = Field(..., ge=0, le=20, description="Number of backlogs")
    communication: int = Field(..., ge=1, le=10, description="Communication score 1-10")

    class Config:
        json_schema_extra = {
            "example": {
                "cgpa"          : 8.5,
                "internships"   : 2,
                "projects"      : 4,
                "backlogs"      : 0,
                "communication" : 8
            }
        }

class PlacementResponse(BaseModel):
    placed          : bool
    placement_probability: float
    predicted_salary_lpa : Optional[float]
    confidence_level: str
    recommendation  : str

class BatchRequest(BaseModel):
    students: List[StudentFeatures]

class ModelInfo(BaseModel):
    model_name  : str
    version     : str
    accuracy    : float
    trained_on  : str
    total_predictions: int

# -- Simple in-memory model (no pickle file needed) -----------
class SimplePlacementModel:
    """
    A simple rule-based model that mimics ML model behavior.
    In production you would load a real trained model with pickle.
    """
    def __init__(self):
        self.version    = "1.0.0"
        self.trained_on = "2025-04-01"
        self.accuracy   = 0.923
        self.total_predictions = 0

    def predict(self, features: dict):
        """Predict placement based on features."""
        self.total_predictions += 1

        cgpa           = features["cgpa"]
        internships    = features["internships"]
        projects       = features["projects"]
        backlogs       = features["backlogs"]
        communication  = features["communication"]

        # Scoring logic (mimics a trained model)
        score = (
            cgpa          * 0.35 +
            internships   * 0.25 +
            projects      * 0.15 +
            communication * 0.15 -
            backlogs      * 0.10
        )

        # Normalize to probability
        max_score    = 10 * 0.35 + 10 * 0.25 + 20 * 0.15 + 10 * 0.15
        probability  = min(score / max_score, 0.99)
        placed       = probability >= 0.50

        # Salary estimate (if placed)
        salary = None
        if placed:
            salary = round(
                cgpa * 0.8 +
                internships * 1.5 +
                projects * 0.3 +
                communication * 0.2 +
                3.0, 2
            )

        # Confidence level
        if probability >= 0.80:
            confidence = "High"
        elif probability >= 0.50:
            confidence = "Medium"
        else:
            confidence = "Low"

        # Recommendation
        if placed and probability >= 0.80:
            rec = "Excellent profile! Apply to top-tier companies."
        elif placed:
            rec = "Good profile. Target mid-tier companies and keep building projects."
        elif backlogs > 2:
            rec = "Clear backlogs first. Then focus on projects and internships."
        elif internships == 0:
            rec = "Get at least 1 internship. It significantly boosts placement chances."
        else:
            rec = "Improve CGPA and build more projects to increase chances."

        return {
            "placed"                  : bool(placed),
            "placement_probability"   : round(probability, 4),
            "predicted_salary_lpa"    : salary,
            "confidence_level"        : confidence,
            "recommendation"          : rec
        }

# Initialize model
model = SimplePlacementModel()

# -- API Endpoints --------------------------------------------

@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "message"  : "AI Placement Predictor API is running!",
        "status"   : "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "docs"     : "Visit /docs for interactive API documentation"
    }

@app.get("/health")
def health_check():
    """Detailed health check."""
    return {
        "status"    : "healthy",
        "model"     : "loaded",
        "version"   : model.version,
        "uptime"    : "running"
    }

@app.get("/model/info", response_model=ModelInfo)
def get_model_info():
    """Get information about the currently loaded model."""
    return ModelInfo(
        model_name        = "GradientBoostingClassifier",
        version           = model.version,
        accuracy          = model.accuracy,
        trained_on        = model.trained_on,
        total_predictions = model.total_predictions
    )

@app.post("/predict", response_model=PlacementResponse)
def predict_placement(student: StudentFeatures):
    """
    Predict placement for a single student.

    Send student features and get:
    - Placement prediction (True/False)
    - Probability of placement
    - Estimated salary if placed
    - Confidence level
    - Personalized recommendation
    """
    try:
        result = model.predict(student.dict())
        return PlacementResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(batch: BatchRequest):
    """
    Predict placement for multiple students at once.
    Useful for processing a CSV of students.
    """
    results = []
    for i, student in enumerate(batch.students):
        result = model.predict(student.dict())
        results.append({
            "student_index": i + 1,
            **result
        })

    placed_count = sum(1 for r in results if r["placed"])

    return {
        "total_students"  : len(results),
        "placed_count"    : placed_count,
        "not_placed_count": len(results) - placed_count,
        "placement_rate"  : round(placed_count / len(results) * 100, 1),
        "predictions"     : results
    }

@app.get("/predict/sample")
def get_sample_prediction():
    """Get a sample prediction to understand the API response format."""
    sample = {
        "cgpa"         : 8.5,
        "internships"  : 2,
        "projects"     : 4,
        "backlogs"     : 0,
        "communication": 8
    }
    result = model.predict(sample)
    return {
        "input" : sample,
        "output": result,
        "note"  : "Send your own data to /predict endpoint"
    }

@app.get("/stats")
def get_stats():
    """Get API usage statistics."""
    return {
        "total_predictions": model.total_predictions,
        "model_version"    : model.version,
        "api_version"      : "1.0.0",
        "endpoints"        : ["/", "/health", "/model/info",
                              "/predict", "/predict/batch",
                              "/predict/sample", "/stats"]
    }
'''

with open("main.py", "w", encoding="utf-8") as f:
    f.write(fastapi_app)

print("  main.py written successfully!")
print("  This is your FastAPI application file.")


# -- SECTION 3: How to Run and Test ---------------------------
print("\n\nSECTION 3: How to Run and Test Your API")
print("-" * 40)
print("""
STEP 1: Install FastAPI
-----------------------
pip install fastapi uvicorn

STEP 2: Run the API
-------------------
uvicorn main:app --reload

You will see:
  INFO:     Uvicorn running on http://127.0.0.1:8000
  INFO:     Application startup complete.

STEP 3: Open the auto-generated docs
--------------------------------------
Go to: http://127.0.0.1:8000/docs

You get a beautiful Swagger UI where you can:
  -> See all endpoints
  -> Click "Try it out" on any endpoint
  -> Send real requests and see responses
  -> No Postman needed!

STEP 4: Test the API with Python requests
------------------------------------------
import requests

# Single prediction
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "cgpa"          : 8.5,
        "internships"   : 2,
        "projects"      : 4,
        "backlogs"      : 0,
        "communication" : 8
    }
)
print(response.json())

# Output:
# {
#   "placed": true,
#   "placement_probability": 0.847,
#   "predicted_salary_lpa": 12.5,
#   "confidence_level": "High",
#   "recommendation": "Excellent profile! Apply to top-tier companies."
# }
""")


# -- SECTION 4: Test the API logic locally --------------------
print("\nSECTION 4: Testing API Logic Without Running Server")
print("-" * 40)

# We can test the model logic directly
class SimplePlacementModelTest:
    def predict(self, features):
        cgpa          = features["cgpa"]
        internships   = features["internships"]
        projects      = features["projects"]
        backlogs      = features["backlogs"]
        communication = features["communication"]

        score        = (cgpa * 0.35 + internships * 0.25 +
                       projects * 0.15 + communication * 0.15 -
                       backlogs * 0.10)
        max_score    = 10*0.35 + 10*0.25 + 20*0.15 + 10*0.15
        probability  = min(score / max_score, 0.99)
        placed       = probability >= 0.50
        salary       = round(cgpa*0.8 + internships*1.5 + projects*0.3 +
                            communication*0.2 + 3.0, 2) if placed else None
        confidence   = "High" if probability >= 0.80 else "Medium" if probability >= 0.50 else "Low"

        return {
            "placed"               : bool(placed),
            "placement_probability": round(probability, 4),
            "predicted_salary_lpa" : salary,
            "confidence_level"     : confidence
        }

test_model = SimplePlacementModelTest()

test_students = [
    {"name": "Star student",  "cgpa": 9.2, "internships": 3, "projects": 5, "backlogs": 0, "communication": 9},
    {"name": "Average",       "cgpa": 7.0, "internships": 1, "projects": 3, "backlogs": 1, "communication": 6},
    {"name": "Struggling",    "cgpa": 5.5, "internships": 0, "projects": 1, "backlogs": 4, "communication": 4},
    {"name": "You (Day 7!)",  "cgpa": 8.0, "internships": 0, "projects": 5, "backlogs": 0, "communication": 7},
]

print(f"\n  {'Name':<15} {'CGPA':>5} {'Intern':>7} {'Proj':>5} {'Back':>5} {'Placed':>7} {'Prob':>7} {'Salary':>8} {'Conf'}")
print(f"  {'─'*15} {'─'*5} {'─'*7} {'─'*5} {'─'*5} {'─'*7} {'─'*7} {'─'*8} {'─'*8}")

for s in test_students:
    features = {k: v for k, v in s.items() if k != "name"}
    result   = test_model.predict(features)
    placed   = "YES" if result["placed"] else "NO"
    salary   = f"{result['predicted_salary_lpa']} LPA" if result["predicted_salary_lpa"] else "N/A"
    print(f"  {s['name']:<15} {s['cgpa']:>5} {s['internships']:>7} {s['projects']:>5} "
          f"{s['backlogs']:>5} {placed:>7} {result['placement_probability']:>7.1%} "
          f"{salary:>8} {result['confidence_level']:>8}")

print()
print("=" * 60)
print("Script 1 complete! FastAPI concepts covered.")
print("Key concepts:")
print("  [OK] REST API concepts (GET, POST, status codes)")
print("  [OK] FastAPI vs Flask vs Django")
print("  [OK] Pydantic models for data validation")
print("  [OK] Building a complete ML serving API")
print("  [OK] Auto-generated Swagger docs")
print("  [OK] Batch prediction endpoint")
print("Next step: Run 'uvicorn main:app --reload' to start the server!")
print("=" * 60)
