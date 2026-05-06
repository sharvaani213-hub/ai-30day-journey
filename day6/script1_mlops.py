# -*- coding: utf-8 -*-
# ============================================================
# DAY 6 -- SCRIPT 1: MLOps & Experiment Tracking
# Topics: What is MLOps, MLflow tracking, model versioning,
#         experiment comparison, model registry concepts
# ============================================================

import numpy as np
import pandas as pd
import warnings
import os
import json
import datetime
warnings.filterwarnings("ignore")

print("=" * 60)
print("DAY 6 -- SCRIPT 1: MLOps & Experiment Tracking")
print("=" * 60)

print("""
WHAT IS MLOPS?
--------------
MLOps = ML + DevOps

In software: you write code -> test -> deploy -> monitor
In ML      : you train model -> evaluate -> deploy -> monitor

Without MLOps:
  -> You try 50 models, forget which one was best
  -> You deploy the wrong model version
  -> Model breaks in production, you don't know why
  -> Can't reproduce results from 3 weeks ago

With MLOps:
  -> Every experiment is tracked automatically
  -> You can compare 50 runs side by side
  -> One-click rollback to previous model version
  -> Alerts when model performance degrades

The 3 core MLOps concepts:
  1. Experiment Tracking  -> log params, metrics, artifacts
  2. Model Registry       -> version and stage models
  3. Model Monitoring     -> detect drift in production
""")


# -- SECTION 1: Manual Experiment Tracker (No MLflow needed) --
print("SECTION 1: Building a Manual Experiment Tracker")
print("-" * 40)
print("Before using MLflow, let's understand what it does under the hood\n")

class ExperimentTracker:
    """
    A simple experiment tracker that saves runs to JSON.
    This is basically what MLflow does, but simplified.
    """

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.runs            = []
        self.current_run     = None
        self.run_id          = 0

    def start_run(self, run_name):
        self.run_id += 1
        self.current_run = {
            "run_id"    : self.run_id,
            "run_name"  : run_name,
            "timestamp" : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "params"    : {},
            "metrics"   : {},
            "tags"      : {}
        }
        print(f"  Started run: {run_name} (ID: {self.run_id})")
        return self

    def log_param(self, key, value):
        """Log a hyperparameter."""
        if self.current_run:
            self.current_run["params"][key] = value

    def log_metric(self, key, value):
        """Log a metric (accuracy, loss, etc.)."""
        if self.current_run:
            self.current_run["metrics"][key] = round(float(value), 4)

    def log_tag(self, key, value):
        """Log a tag (model type, author, etc.)."""
        if self.current_run:
            self.current_run["tags"][key] = value

    def end_run(self):
        """Finish the current run and save it."""
        if self.current_run:
            self.runs.append(self.current_run.copy())
            self.current_run = None

    def get_best_run(self, metric, higher_is_better=True):
        """Find the best run based on a metric."""
        if not self.runs:
            return None
        sorted_runs = sorted(
            self.runs,
            key    = lambda r: r["metrics"].get(metric, 0),
            reverse= higher_is_better
        )
        return sorted_runs[0]

    def compare_runs(self):
        """Print a comparison table of all runs."""
        if not self.runs:
            print("  No runs to compare.")
            return

        print(f"\n  Experiment: {self.experiment_name}")
        print(f"  Total runs : {len(self.runs)}\n")

        # Header
        print(f"  {'Run':<5} {'Name':<25} {'Model':<20} {'LR':>8} {'Acc':>8} {'F1':>8} {'Time(s)':>9}")
        print(f"  {'─'*5} {'─'*25} {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*9}")

        for run in self.runs:
            p = run["params"]
            m = run["metrics"]
            print(f"  {run['run_id']:<5} "
                  f"{run['run_name']:<25} "
                  f"{p.get('model','─'):<20} "
                  f"{p.get('learning_rate','─'):>8} "
                  f"{m.get('accuracy','─'):>8} "
                  f"{m.get('f1_score','─'):>8} "
                  f"{m.get('train_time','─'):>9}")

    def save(self, filepath="experiment_runs.json"):
        """Save all runs to JSON."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "experiment": self.experiment_name,
                "runs"      : self.runs
            }, f, indent=2)
        print(f"\n  Saved {len(self.runs)} runs to {filepath}")


# Now run actual ML experiments and track them
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time

# Create dataset
X, y = make_classification(
    n_samples    = 1000,
    n_features   = 10,
    n_informative= 7,
    n_redundant  = 2,
    random_state = 42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Define experiments
experiments = [
    {
        "name"  : "logistic_baseline",
        "model" : LogisticRegression(C=1.0),
        "params": {"model": "LogisticRegression", "C": 1.0, "learning_rate": "lbfgs"}
    },
    {
        "name"  : "logistic_high_C",
        "model" : LogisticRegression(C=10.0),
        "params": {"model": "LogisticRegression", "C": 10.0, "learning_rate": "lbfgs"}
    },
    {
        "name"  : "decision_tree_d3",
        "model" : DecisionTreeClassifier(max_depth=3),
        "params": {"model": "DecisionTree", "max_depth": 3, "learning_rate": "N/A"}
    },
    {
        "name"  : "decision_tree_d7",
        "model" : DecisionTreeClassifier(max_depth=7),
        "params": {"model": "DecisionTree", "max_depth": 7, "learning_rate": "N/A"}
    },
    {
        "name"  : "random_forest_100",
        "model" : RandomForestClassifier(n_estimators=100, random_state=42),
        "params": {"model": "RandomForest", "n_estimators": 100, "learning_rate": "N/A"}
    },
    {
        "name"  : "gradient_boosting",
        "model" : GradientBoostingClassifier(n_estimators=100, learning_rate=0.1),
        "params": {"model": "GradientBoosting", "n_estimators": 100, "learning_rate": 0.1}
    },
]

tracker = ExperimentTracker("placement_prediction_v1")

print("Running 6 experiments and tracking all results...\n")

for exp in experiments:
    tracker.start_run(exp["name"])

    # Log params
    for k, v in exp["params"].items():
        tracker.log_param(k, v)
    tracker.log_tag("author", "Sharvaani")
    tracker.log_tag("dataset", "synthetic_placement")

    # Train
    start = time.time()
    exp["model"].fit(X_train_s, y_train)
    train_time = round(time.time() - start, 3)

    # Evaluate
    y_pred  = exp["model"].predict(X_test_s)
    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred)
    rec     = recall_score(y_test, y_pred)

    # Log metrics
    tracker.log_metric("accuracy",   acc)
    tracker.log_metric("f1_score",   f1)
    tracker.log_metric("precision",  prec)
    tracker.log_metric("recall",     rec)
    tracker.log_metric("train_time", train_time)

    tracker.end_run()

# Compare all runs
tracker.compare_runs()

# Find best model
best = tracker.get_best_run("accuracy")
print(f"\n  BEST RUN: {best['run_name']}")
print(f"  Accuracy : {best['metrics']['accuracy']}")
print(f"  F1 Score : {best['metrics']['f1_score']}")
print(f"  Model    : {best['params']['model']}")

tracker.save("experiment_runs.json")


# -- SECTION 2: MLflow (if installed) -------------------------
print("\n\nSECTION 2: MLflow -- Industry Standard Tracking")
print("-" * 40)
print("""
MLflow is the most popular open-source MLOps tool.
Used by Netflix, Airbnb, Microsoft, and thousands of companies.

Core MLflow concepts:
  mlflow.start_run()         -> start tracking
  mlflow.log_param()         -> log hyperparameter
  mlflow.log_metric()        -> log evaluation metric
  mlflow.log_artifact()      -> log file (model, chart, CSV)
  mlflow.sklearn.log_model() -> log the actual model
  mlflow ui                  -> opens a beautiful web dashboard
""")

try:
    import mlflow
    import mlflow.sklearn

    print("  MLflow is installed! Running full experiment tracking...\n")

    mlflow.set_experiment("day6_placement_prediction")

    for exp in experiments[:3]:  # run 3 experiments with MLflow
        with mlflow.start_run(run_name=exp["name"]):
            # Log params
            for k, v in exp["params"].items():
                mlflow.log_param(k, str(v))

            # Train
            exp["model"].fit(X_train_s, y_train)
            y_pred = exp["model"].predict(X_test_s)

            # Log metrics
            mlflow.log_metric("accuracy",  accuracy_score(y_test, y_pred))
            mlflow.log_metric("f1_score",  f1_score(y_test, y_pred))
            mlflow.log_metric("precision", precision_score(y_test, y_pred))
            mlflow.log_metric("recall",    recall_score(y_test, y_pred))

            # Log the model itself
            mlflow.sklearn.log_model(exp["model"], "model")

            print(f"  Logged run: {exp['name']}")

    print("\n  All runs logged to MLflow!")
    print("  Run 'mlflow ui' in your terminal to see the dashboard at http://localhost:5000")

except ImportError:
    print("  MLflow not installed. Run: pip install mlflow")
    print("""
  What the MLflow dashboard looks like:
  -> All runs listed in a table
  -> Click any run to see full details
  -> Compare runs side by side
  -> Download the best model
  -> One-click deploy to cloud
  """)


# -- SECTION 3: Model Versioning ──────────────────────────────
print("\n\nSECTION 3: Model Versioning")
print("-" * 40)
print("""
Model versioning = tracking different versions of your model over time.

Like Git for code, but for ML models.

Model lifecycle stages:
  Staging    -> tested, ready for review
  Production -> live, serving real users
  Archived   -> old version, kept for rollback

Why it matters:
  v1.0 -> trained on 1000 samples, accuracy 85%
  v1.1 -> added feature engineering, accuracy 88%
  v2.0 -> switched to gradient boosting, accuracy 92%
  v2.1 -> bug fix in preprocessing, accuracy 91%  <- REGRESSION!
  -> Rollback to v2.0 instantly because it was versioned!
""")

# Simulate model registry
model_registry = {
    "placement_predictor": [
        {"version": "1.0", "accuracy": 0.85, "stage": "Archived",    "date": "2025-01-01", "model": "LogisticRegression"},
        {"version": "1.1", "accuracy": 0.88, "stage": "Archived",    "date": "2025-02-01", "model": "LogisticRegression"},
        {"version": "2.0", "accuracy": 0.92, "stage": "Production",  "date": "2025-03-01", "model": "GradientBoosting"},
        {"version": "2.1", "accuracy": 0.91, "stage": "Staging",     "date": "2025-04-01", "model": "GradientBoosting"},
    ]
}

print("\n  Model Registry: placement_predictor")
print(f"\n  {'Version':<10} {'Model':<22} {'Accuracy':>10} {'Stage':<12} {'Date'}")
print(f"  {'─'*10} {'─'*22} {'─'*10} {'─'*12} {'─'*12}")

for v in model_registry["placement_predictor"]:
    marker = " <-- LIVE" if v["stage"] == "Production" else ""
    print(f"  {v['version']:<10} {v['model']:<22} {v['accuracy']:>10.2%} {v['stage']:<12} {v['date']}{marker}")


# -- SECTION 4: Model Saving and Loading ----------------------
print("\n\nSECTION 4: Saving and Loading Models")
print("-" * 40)
print("In production you save the best model to disk and load it for predictions\n")

import pickle

# Train the best model
best_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
best_model.fit(X_train_s, y_train)

# Save model + scaler together
model_package = {
    "model"      : best_model,
    "scaler"     : scaler,
    "version"    : "2.0",
    "trained_on" : datetime.datetime.now().strftime("%Y-%m-%d"),
    "accuracy"   : accuracy_score(y_test, best_model.predict(X_test_s)),
    "features"   : ["feature_" + str(i) for i in range(10)]
}

with open("best_model_v2.pkl", "wb") as f:
    pickle.dump(model_package, f)

print("  Model saved to best_model_v2.pkl")

# Load and use
with open("best_model_v2.pkl", "rb") as f:
    loaded = pickle.load(f)

print(f"  Loaded model version : {loaded['version']}")
print(f"  Trained on           : {loaded['trained_on']}")
print(f"  Accuracy             : {loaded['accuracy']:.2%}")

# Make prediction with loaded model
sample = np.random.randn(1, 10)
sample_scaled = loaded["scaler"].transform(sample)
prediction    = loaded["model"].predict(sample_scaled)
probability   = loaded["model"].predict_proba(sample_scaled)

print(f"\n  New prediction:")
print(f"  Raw features (sample): {sample[0][:4].round(3)}...")
print(f"  Prediction           : {'Placed' if prediction[0] == 1 else 'Not Placed'}")
print(f"  Confidence           : {max(probability[0]):.2%}")


# -- SECTION 5: Data Drift Detection --------------------------
print("\n\nSECTION 5: Data Drift -- Why Models Degrade")
print("-" * 40)
print("""
Data drift = the distribution of incoming data changes over time.
Your model was trained on old data, but the real world changed.

Example:
  2023: trained on job market data -> 90% accuracy
  2024: AI boom, job requirements changed -> model sees new patterns
  2025: model accuracy drops to 70% because of drift

How to detect drift:
  -> Compare mean/std of features between training and production data
  -> If difference > threshold -> retrain the model
""")

# Simulate drift detection
np.random.seed(42)
train_data = np.random.normal(0, 1, (1000, 5))     # training distribution

# Production data with slight drift
prod_data_ok   = np.random.normal(0.1, 1.1, (200, 5))   # slight drift, acceptable
prod_data_drift= np.random.normal(1.5, 2.0, (200, 5))   # significant drift!

def detect_drift(train, production, threshold=0.3):
    """Detect if production data has drifted from training data."""
    results = []
    for i in range(train.shape[1]):
        train_mean = np.mean(train[:, i])
        prod_mean  = np.mean(production[:, i])
        drift_score= abs(prod_mean - train_mean) / (np.std(train[:, i]) + 1e-8)
        drifted    = drift_score > threshold
        results.append({
            "feature"    : f"feature_{i}",
            "train_mean" : round(train_mean, 3),
            "prod_mean"  : round(prod_mean, 3),
            "drift_score": round(drift_score, 3),
            "drifted"    : drifted
        })
    return results

print("\n  Drift Report -- Normal production data:")
results_ok = detect_drift(train_data, prod_data_ok)
for r in results_ok:
    status = "DRIFT DETECTED!" if r["drifted"] else "OK"
    print(f"    {r['feature']}: train_mean={r['train_mean']:>7} | prod_mean={r['prod_mean']:>7} | drift={r['drift_score']:>6} | {status}")

print("\n  Drift Report -- Drifted production data:")
results_drift = detect_drift(train_data, prod_data_drift)
drifted_count = 0
for r in results_drift:
    status = "DRIFT DETECTED!" if r["drifted"] else "OK"
    if r["drifted"]:
        drifted_count += 1
    print(f"    {r['feature']}: train_mean={r['train_mean']:>7} | prod_mean={r['prod_mean']:>7} | drift={r['drift_score']:>6} | {status}")

if drifted_count > 0:
    print(f"\n  ALERT: {drifted_count} features have drifted! Consider retraining the model.")

print()
print("=" * 60)
print("Script 1 complete! MLOps & Experiment Tracking covered.")
print("Key concepts:")
print("  [OK] Experiment tracking (manual + MLflow)")
print("  [OK] Comparing multiple model runs")
print("  [OK] Model versioning and registry")
print("  [OK] Saving and loading models (pickle)")
print("  [OK] Data drift detection")
print("=" * 60)
