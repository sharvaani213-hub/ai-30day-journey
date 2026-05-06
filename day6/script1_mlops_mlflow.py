# -*- coding: utf-8 -*-
# ============================================================
# DAY 6 -- SCRIPT 1: MLOps & Experiment Tracking
# Topics: What is MLOps, MLflow, experiment tracking,
#         model versioning, metrics logging, model registry
# ============================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("DAY 6 -- SCRIPT 1: MLOps & Experiment Tracking")
print("=" * 60)

print("""
WHAT IS MLOps?
--------------
MLOps = ML + DevOps = practices to deploy and maintain ML models

The problem without MLOps:
  -> You train 50 models, forget which one was best
  -> You change code, model gets worse, you don't know why
  -> You deploy a model, it breaks in production, no idea why
  -> Team members can't reproduce your results

MLOps solves this with:
  1. Experiment Tracking  -> log every run, compare results
  2. Model Versioning     -> save every model with metadata
  3. Data Versioning      -> track what data you trained on
  4. CI/CD Pipelines      -> auto test and deploy models
  5. Model Monitoring     -> detect when model degrades

TODAY we cover steps 1 and 2 using MLflow -- the most popular
free and open-source MLOps tool used by real companies.
""")


# -- SECTION 1: Setup MLflow ----------------------------------
print("SECTION 1: Setting up MLflow")
print("-" * 40)

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
    print(f"  MLflow version: {mlflow.__version__}")
    print("  MLflow installed successfully!")
except ImportError:
    MLFLOW_AVAILABLE = False
    print("  MLflow not installed. Run: pip install mlflow")
    print("  Showing code walkthroughs instead.\n")


# -- SECTION 2: The Problem -- Manual Experiment Tracking -----
print("\n\nSECTION 2: The Problem -- Manual Tracking is Messy")
print("-" * 40)
print("""
Most beginners track experiments like this:

  Run 1: accuracy = 0.82, model saved as model_v1.pkl
  Run 2: accuracy = 0.85, model saved as model_v2_lr0.01.pkl
  Run 3: accuracy = 0.79, model saved as model_final.pkl
  Run 4: accuracy = 0.88, model saved as model_final_FINAL.pkl
  Run 5: accuracy = 0.91, model saved as model_USE_THIS_ONE.pkl

After 20 runs you have no idea:
  -> Which hyperparameters gave 0.91?
  -> What data was used for that run?
  -> Can you reproduce it?
  -> Which model is actually in production?

MLflow solves all of this automatically.
""")


# -- SECTION 3: Create Dataset --------------------------------
print("SECTION 3: Creating Dataset for Experiments")
print("-" * 40)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

np.random.seed(42)

# Create a classification dataset
X, y = make_classification(
    n_samples     = 1000,
    n_features    = 15,
    n_informative = 10,
    n_redundant   = 3,
    random_state  = 42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler       = StandardScaler()
X_train_sc   = scaler.fit_transform(X_train)
X_test_sc    = scaler.transform(X_test)

print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"  Train  : {X_train.shape[0]} samples")
print(f"  Test   : {X_test.shape[0]} samples")
print(f"  Class balance: {np.bincount(y)}")


# -- SECTION 4: MLflow Experiment Tracking --------------------
print("\n\nSECTION 4: MLflow Experiment Tracking")
print("-" * 40)

# Define experiments to run
experiments = [
    {
        "name"  : "Logistic Regression",
        "model" : LogisticRegression(C=1.0, max_iter=1000),
        "params": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000}
    },
    {
        "name"  : "Logistic Regression (high C)",
        "model" : LogisticRegression(C=10.0, max_iter=1000),
        "params": {"C": 10.0, "solver": "lbfgs", "max_iter": 1000}
    },
    {
        "name"  : "Random Forest (50 trees)",
        "model" : RandomForestClassifier(n_estimators=50, random_state=42),
        "params": {"n_estimators": 50, "max_depth": "None", "min_samples_split": 2}
    },
    {
        "name"  : "Random Forest (100 trees)",
        "model" : RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "params": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2}
    },
    {
        "name"  : "Gradient Boosting",
        "model" : GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}
    },
]

results = []

if MLFLOW_AVAILABLE:
    # Set up MLflow
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("Day6_Classification_Experiments")

    print("  Running experiments and logging to MLflow...\n")
    print(f"  {'Experiment':<35} {'Accuracy':>10} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print(f"  {'─'*35} {'─'*10} {'─'*8} {'─'*10} {'─'*8}")

    for exp in experiments:
        with mlflow.start_run(run_name=exp["name"]):

            # Train
            model = exp["model"]
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)

            # Calculate metrics
            accuracy  = accuracy_score(y_test, y_pred)
            f1        = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall    = recall_score(y_test, y_pred)

            # Log parameters
            mlflow.log_params(exp["params"])

            # Log metrics
            mlflow.log_metrics({
                "accuracy" : accuracy,
                "f1_score" : f1,
                "precision": precision,
                "recall"   : recall
            })

            # Log the model itself
            mlflow.sklearn.log_model(model, "model")

            # Log dataset info as tags
            mlflow.set_tags({
                "dataset"   : "make_classification",
                "n_samples" : 1000,
                "developer" : "Sharvaani",
                "day"       : "6"
            })

            results.append({
                "name"     : exp["name"],
                "accuracy" : accuracy,
                "f1"       : f1,
                "precision": precision,
                "recall"   : recall
            })

            print(f"  {exp['name']:<35} {accuracy:>10.4f} {f1:>8.4f} {precision:>10.4f} {recall:>8.4f}")

    print("\n  All experiments logged to MLflow!")
    print("  Run 'mlflow ui' to see the dashboard at http://localhost:5000")

else:
    # Show results without MLflow
    print("  Running experiments (without MLflow logging)...\n")
    print(f"  {'Experiment':<35} {'Accuracy':>10} {'F1':>8}")
    print(f"  {'─'*35} {'─'*10} {'─'*8}")

    for exp in experiments:
        model = exp["model"]
        model.fit(X_train_sc, y_train)
        y_pred    = model.predict(X_test_sc)
        accuracy  = accuracy_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        results.append({
            "name": exp["name"], "accuracy": accuracy,
            "f1": f1, "precision": precision, "recall": recall
        })
        print(f"  {exp['name']:<35} {accuracy:>10.4f} {f1:>8.4f}")


# -- SECTION 5: Compare Results -------------------------------
print("\n\nSECTION 5: Comparing All Experiments")
print("-" * 40)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("accuracy", ascending=False)

print("  Experiments ranked by accuracy:\n")
print(f"  {'Rank':<5} {'Model':<35} {'Accuracy':>10} {'F1':>8} {'Precision':>10} {'Recall':>8}")
print(f"  {'─'*5} {'─'*35} {'─'*10} {'─'*8} {'─'*10} {'─'*8}")

for rank, (_, row) in enumerate(results_df.iterrows(), 1):
    marker = " <-- BEST" if rank == 1 else ""
    print(f"  {rank:<5} {row['name']:<35} {row['accuracy']:>10.4f} {row['f1']:>8.4f} {row['precision']:>10.4f} {row['recall']:>8.4f}{marker}")

best = results_df.iloc[0]
print(f"\n  Best model    : {best['name']}")
print(f"  Best accuracy : {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")
print(f"  Best F1 score : {best['f1']:.4f}")


# -- SECTION 6: Save Best Model -------------------------------
print("\n\nSECTION 6: Saving Best Model")
print("-" * 40)

import pickle

# Find and retrain best model
best_exp = next(e for e in experiments if e["name"] == best["name"])
best_model = best_exp["model"]
best_model.fit(X_train_sc, y_train)

# Save with pickle
model_info = {
    "model"      : best_model,
    "scaler"     : scaler,
    "model_name" : best["name"],
    "accuracy"   : best["accuracy"],
    "f1"         : best["f1"],
    "features"   : 15,
    "trained_on" : "make_classification 1000 samples"
}

with open("best_model.pkl", "wb") as f:
    pickle.dump(model_info, f)

print(f"  Best model saved as 'best_model.pkl'")
print(f"  Model: {best['name']}")
print(f"  Accuracy: {best['accuracy']:.4f}")

# Test loading
with open("best_model.pkl", "rb") as f:
    loaded = pickle.load(f)

test_pred = loaded["model"].predict(loaded["scaler"].transform(X_test[:5]))
print(f"\n  Loaded model test predictions: {test_pred}")
print(f"  Actual labels                : {y_test[:5]}")
print(f"  Model loaded and working correctly!")


# -- SECTION 7: How to Use MLflow UI --------------------------
print("\n\nSECTION 7: How to Use MLflow Dashboard")
print("-" * 40)
print("""
After running this script, open MLflow UI:

  Step 1: Open a NEW Command Prompt window
  Step 2: Go to your project folder:
          cd "C:\\Users\\Sharvaani K\\ai-30day-journey\\day6"
  Step 3: Run:
          mlflow ui
  Step 4: Open browser at: http://localhost:5000

What you will see:
  -> All your experiment runs listed
  -> Click any run to see parameters + metrics
  -> Compare runs side by side
  -> Download any saved model
  -> Filter and sort by any metric

This is what data scientists use at companies like
Google, Microsoft, and Amazon to manage their ML experiments.
""")


# -- SECTION 8: MLflow Key Concepts ---------------------------
print("SECTION 8: MLflow Key Concepts")
print("-" * 40)
print("""
  Experiment  -> a group of related runs (e.g. "Placement Prediction v1")
  Run         -> one training session with specific hyperparameters
  Parameters  -> inputs to the model (learning_rate=0.01, n_estimators=100)
  Metrics     -> outputs/results (accuracy=0.91, f1=0.89)
  Artifacts   -> files saved (model.pkl, plots, datasets)
  Tags        -> metadata (developer name, dataset version, notes)
  Model Registry -> production-ready model store with versioning

MLflow commands you will use daily:
  mlflow.set_experiment("name")  -> create/select experiment
  mlflow.start_run()             -> begin logging a run
  mlflow.log_param("lr", 0.01)   -> log one parameter
  mlflow.log_params(dict)        -> log multiple parameters
  mlflow.log_metric("acc", 0.91) -> log one metric
  mlflow.log_metrics(dict)       -> log multiple metrics
  mlflow.log_artifact("file.png")-> save a file
  mlflow.sklearn.log_model(m,"m")-> save a sklearn model
  mlflow ui                      -> launch the dashboard
""")

print("=" * 60)
print("Script 1 complete! MLOps + MLflow covered.")
print("Key concepts:")
print("  [OK] What MLOps is and why it matters")
print("  [OK] MLflow experiment tracking")
print("  [OK] Logging parameters, metrics, models")
print("  [OK] Comparing multiple experiments")
print("  [OK] Saving and loading best model")
print("  [OK] MLflow UI dashboard")
print("=" * 60)
