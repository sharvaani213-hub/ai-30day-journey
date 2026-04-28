# ============================================================
# DAY 2 — SCRIPT 3: Classification Models
# Topics: Logistic Regression, Decision Tree, confusion matrix,
#         precision, recall, F1 score, ROC curve
# Dataset: Predicting whether a student gets placed or not
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, roc_curve, auc)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("DAY 2 — SCRIPT 3: Classification Models")
print("Predicting Student Placement (Yes/No)")
print("=" * 60)


# ── STEP 1: Create Dataset ────────────────────────────────────
print("\nSTEP 1: Creating dataset")
print("-" * 40)

np.random.seed(42)
n = 300

cgpa        = np.round(np.random.normal(7.2, 1.1, n).clip(4.5, 10.0), 2)
internships = np.random.randint(0, 4, n)
projects    = np.random.randint(1, 6, n)
backlogs    = np.random.randint(0, 6, n)
communication = np.random.randint(1, 11, n)     # 1-10 rating

# Placement probability based on features
placement_score = (
    cgpa * 0.4 +
    internships * 0.8 +
    projects * 0.3 +
    communication * 0.2 -
    backlogs * 0.5
)

# Convert score to binary placement (0 or 1)
prob      = 1 / (1 + np.exp(-( placement_score - 5)))  # sigmoid
placed    = (np.random.rand(n) < prob).astype(int)

df = pd.DataFrame({
    "cgpa"          : cgpa,
    "internships"   : internships,
    "projects"      : projects,
    "backlogs"      : backlogs,
    "communication" : communication,
    "placed"        : placed
})

print(f"Dataset        : {df.shape}")
print(f"Placed         : {placed.sum()} students ({placed.mean()*100:.1f}%)")
print(f"Not placed     : {(1-placed).sum()} students ({(1-placed).mean()*100:.1f}%)")
print(f"\nFirst 5 rows:\n{df.head().to_string()}")


# ── STEP 2: Understand the Difference from Regression ─────────
print("\n\nSTEP 2: Regression vs Classification")
print("-" * 40)
print("Regression  → predicts a NUMBER  (salary = 8.5 LPA)")
print("Classification → predicts a CLASS (placed = Yes/No)")
print()
print("Logistic Regression: finds probability of being in a class")
print("Decision Tree      : asks yes/no questions to classify")


# ── STEP 3: Prepare Data ─────────────────────────────────────
print("\n\nSTEP 3: Preparing data")
print("-" * 40)

X = df[["cgpa", "internships", "projects", "backlogs", "communication"]]
y = df["placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify = keep same class ratio
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
print(f"Train placed: {y_train.sum()} | Test placed: {y_test.sum()}")


# ── STEP 4: Model 1 — Logistic Regression ─────────────────────
print("\n\nSTEP 4: Model 1 — Logistic Regression")
print("-" * 40)
print("Works by: finding a probability boundary between classes")

lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_s, y_train)

lr_pred      = lr_model.predict(X_test_s)
lr_pred_prob = lr_model.predict_proba(X_test_s)[:, 1]  # probability of being placed
lr_accuracy  = accuracy_score(y_test, lr_pred)

print(f"\nLogistic Regression Accuracy: {lr_accuracy:.2%}")
print(f"\nFeature coefficients (impact on placement):")
for feat, coef in zip(X.columns, lr_model.coef_[0]):
    direction = "helps placement ↑" if coef > 0 else "hurts placement ↓"
    print(f"  {feat:<15}: {coef:+.3f}  {direction}")


# ── STEP 5: Model 2 — Decision Tree ───────────────────────────
print("\n\nSTEP 5: Model 2 — Decision Tree")
print("-" * 40)
print("Works by: asking yes/no questions like 'CGPA > 7.5?'")

dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)   # Decision tree doesn't need scaling!

dt_pred     = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

print(f"\nDecision Tree Accuracy: {dt_accuracy:.2%}")

# Print the actual decision rules
print("\nDecision Tree Rules (first few levels):")
tree_rules = export_text(dt_model, feature_names=list(X.columns), max_depth=3)
# Print first 20 lines only
for line in tree_rules.split("\n")[:20]:
    print(f"  {line}")
print("  ...")


# ── STEP 6: Confusion Matrix — Most Important Metric ──────────
print("\n\nSTEP 6: Confusion Matrix")
print("-" * 40)
print("The confusion matrix shows EXACTLY where the model goes wrong\n")

def print_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"  {model_name}:")
    print(f"  {'':20} Predicted NOT placed  Predicted Placed")
    print(f"  {'Actual NOT placed':<20} {tn:^22} {fp:^16}")
    print(f"  {'Actual Placed':<20} {fn:^22} {tp:^16}")
    print()
    print(f"  TP (True Positive)  = {tp}  → correctly predicted placed")
    print(f"  TN (True Negative)  = {tn}  → correctly predicted not placed")
    print(f"  FP (False Positive) = {fp}  → said placed but wasn't (Type I error)")
    print(f"  FN (False Negative) = {fn}  → said not placed but was (Type II error)")

print_confusion_matrix(y_test, lr_pred, "Logistic Regression")


# ── STEP 7: Precision, Recall, F1 ────────────────────────────
print("\n\nSTEP 7: Precision, Recall, F1 Score")
print("-" * 40)
print("Accuracy alone can be misleading — use these 3 metrics together\n")
print("  Precision = of all predicted PLACED, how many actually were placed?")
print("  Recall    = of all actually PLACED, how many did we catch?")
print("  F1        = harmonic mean of precision and recall\n")

print("Logistic Regression report:")
print(classification_report(y_test, lr_pred, target_names=["Not Placed", "Placed"]))

print("Decision Tree report:")
print(classification_report(y_test, dt_pred, target_names=["Not Placed", "Placed"]))


# ── STEP 8: Cross-Validation ─────────────────────────────────
print("\nSTEP 8: Cross-Validation (more reliable evaluation)")
print("-" * 40)
print("K-Fold CV trains & tests the model K times on different data splits")
print("This gives a more reliable accuracy than one train/test split\n")

lr_cv = cross_val_score(LogisticRegression(), X_train_s, y_train, cv=5, scoring="accuracy")
dt_cv = cross_val_score(DecisionTreeClassifier(max_depth=4), X_train, y_train, cv=5, scoring="accuracy")

print(f"Logistic Regression CV scores : {lr_cv.round(3)}")
print(f"  Mean: {lr_cv.mean():.3f} ± {lr_cv.std():.3f}")

print(f"\nDecision Tree CV scores       : {dt_cv.round(3)}")
print(f"  Mean: {dt_cv.mean():.3f} ± {dt_cv.std():.3f}")

winner = "Logistic Regression" if lr_cv.mean() > dt_cv.mean() else "Decision Tree"
print(f"\n✓ Better model: {winner}")


# ── STEP 9: Predict for New Students ──────────────────────────
print("\n\nSTEP 9: Predicting placement for new students")
print("-" * 40)

new_students = pd.DataFrame({
    "cgpa"          : [9.1, 6.0, 7.5, 5.0],
    "internships"   : [3,   0,   1,   0  ],
    "projects"      : [5,   1,   3,   2  ],
    "backlogs"      : [0,   3,   0,   5  ],
    "communication" : [9,   5,   7,   4  ]
})

new_scaled = scaler.transform(new_students)
new_pred   = lr_model.predict(new_scaled)
new_prob   = lr_model.predict_proba(new_scaled)[:, 1]

profiles = ["Star student", "Struggling", "Average", "At-risk"]
print(f"\n{'Profile':<14} {'CGPA':>5} {'Intern':>7} {'Proj':>5} {'Back':>5} {'Comm':>5} {'Placed?':>8} {'Prob':>6}")
print("─" * 65)
for i, (_, row) in enumerate(new_students.iterrows()):
    result = "✓ YES" if new_pred[i] == 1 else "✗ NO"
    print(f"{profiles[i]:<14} {row['cgpa']:>5} {row['internships']:>7} "
          f"{row['projects']:>5} {row['backlogs']:>5} {row['communication']:>5} "
          f"{result:>8} {new_prob[i]:>5.1%}")


# ── STEP 10: Visualise ────────────────────────────────────────
print("\n\nSTEP 10: Saving visualisation")
print("-" * 40)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Day 2 — Classification: Student Placement Prediction", fontsize=13, fontweight="bold")

BLUE = "#378ADD"; GREEN = "#1D9E75"; ORANGE = "#BA7517"; RED = "#E24B4A"

# Chart 1: Confusion matrix heatmap
ax1 = axes[0]
cm = confusion_matrix(y_test, lr_pred)
im = ax1.imshow(cm, cmap="Blues")
ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
ax1.set_xticklabels(["Not Placed", "Placed"])
ax1.set_yticklabels(["Not Placed", "Placed"])
ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")
ax1.set_title("Confusion Matrix\n(Logistic Regression)")
for i in range(2):
    for j in range(2):
        ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14, fontweight="bold")

# Chart 2: ROC Curve
ax2 = axes[1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_pred_prob)
auc_lr = auc(fpr_lr, tpr_lr)
dt_pred_prob = dt_model.predict_proba(X_test)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_pred_prob)
auc_dt = auc(fpr_dt, tpr_dt)
ax2.plot(fpr_lr, tpr_lr, color=BLUE,   linewidth=2, label=f"Logistic Reg (AUC={auc_lr:.3f})")
ax2.plot(fpr_dt, tpr_dt, color=ORANGE, linewidth=2, label=f"Decision Tree (AUC={auc_dt:.3f})")
ax2.plot([0,1],[0,1], "k--", linewidth=1, label="Random (AUC=0.5)")
ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve\n(closer to top-left = better)")
ax2.legend(fontsize=8); ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

# Chart 3: Feature importance (Decision Tree)
ax3 = axes[2]
importances = dt_model.feature_importances_
sorted_idx  = np.argsort(importances)
feat_names  = [list(X.columns)[i] for i in sorted_idx]
feat_vals   = importances[sorted_idx]
colors = [GREEN if v > 0.15 else BLUE for v in feat_vals]
ax3.barh(feat_names, feat_vals, color=colors, edgecolor="white")
ax3.set_xlabel("Feature Importance")
ax3.set_title("Decision Tree\nFeature Importance")
ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("day2_classification.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Chart saved as 'day2_classification.png'")

print()
print("=" * 60)
print("Script 3 complete! Classification models done.")
print(f"  Logistic Regression accuracy: {lr_accuracy:.2%}")
print(f"  Decision Tree accuracy      : {dt_accuracy:.2%}")
print("Key concepts covered:")
print("  ✓ Classification vs Regression")
print("  ✓ Logistic Regression & Decision Tree")
print("  ✓ Confusion Matrix (TP, TN, FP, FN)")
print("  ✓ Precision, Recall, F1 Score")
print("  ✓ Cross-Validation")
print("  ✓ ROC Curve & AUC")
print("=" * 60)
