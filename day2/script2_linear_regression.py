# ============================================================
# DAY 2 — SCRIPT 2: Your First ML Model with scikit-learn
# Topics: train/test split, linear regression, evaluation
#         metrics, feature engineering, model persistence
# Dataset: Predicting student placement salary
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("DAY 2 — SCRIPT 2: Your First ML Model")
print("Predicting Placement Salary from Student Features")
print("=" * 60)


# ── STEP 1: Create Dataset ────────────────────────────────────
print("\nSTEP 1: Creating dataset")
print("-" * 40)

np.random.seed(42)
n = 200   # 200 students

# Features that affect placement salary
cgpa        = np.round(np.random.normal(7.5, 1.0, n).clip(5.0, 10.0), 2)
internships = np.random.randint(0, 4, n)          # 0 to 3 internships
projects    = np.random.randint(1, 6, n)          # 1 to 5 projects
backlogs    = np.random.randint(0, 5, n)          # 0 to 4 backlogs

# Salary formula (realistic): higher CGPA + internships = higher salary
# Adding some noise to make it realistic
salary_lpa = (
    cgpa * 1.5 +
    internships * 1.2 +
    projects * 0.5 -
    backlogs * 0.8 +
    np.random.normal(0, 1.0, n)
).clip(3.0, 25.0)

df = pd.DataFrame({
    "cgpa"        : cgpa,
    "internships" : internships,
    "projects"    : projects,
    "backlogs"    : backlogs,
    "salary_lpa"  : np.round(salary_lpa, 2)
})

# Only keep placed students (salary > 4 LPA)
df = df[df["salary_lpa"] > 4].reset_index(drop=True)

print(f"Dataset shape  : {df.shape}")
print(f"\nFirst 5 rows:\n{df.head().to_string()}")
print(f"\nBasic stats:\n{df.describe().round(2).to_string()}")


# ── STEP 2: Explore the Data ──────────────────────────────────
print("\n\nSTEP 2: Exploring the data")
print("-" * 40)

print(f"Missing values : {df.isnull().sum().sum()}")
print(f"Avg salary     : ₹{df['salary_lpa'].mean():.2f} LPA")
print(f"Max salary     : ₹{df['salary_lpa'].max():.2f} LPA")
print(f"Min salary     : ₹{df['salary_lpa'].min():.2f} LPA")

# Correlation — which feature affects salary most?
print(f"\nCorrelation with salary:")
corr = df.corr()["salary_lpa"].drop("salary_lpa").sort_values(ascending=False)
for feat, val in corr.items():
    bar = "█" * int(abs(val) * 20)
    sign = "+" if val > 0 else "-"
    print(f"  {feat:<15}: {sign}{abs(val):.3f}  {bar}")


# ── STEP 3: Prepare Features & Target ────────────────────────
print("\n\nSTEP 3: Preparing features (X) and target (y)")
print("-" * 40)

# X = input features, y = what we want to predict
X = df[["cgpa", "internships", "projects", "backlogs"]]
y = df["salary_lpa"]

print(f"X shape (features)  : {X.shape}")
print(f"y shape (target)    : {y.shape}")
print(f"\nFeature columns: {list(X.columns)}")
print(f"Target         : salary_lpa")


# ── STEP 4: Train / Test Split ────────────────────────────────
print("\n\nSTEP 4: Train/Test Split (80% train, 20% test)")
print("-" * 40)
print("WHY: We train on 80% and test on the remaining 20%")
print("     so we know if the model works on UNSEEN data.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42     # reproducible split
)

print(f"\nTraining set   : {X_train.shape[0]} students")
print(f"Testing set    : {X_test.shape[0]} students")
print(f"\nFirst 3 training samples:\n{X_train.head(3).to_string()}")


# ── STEP 5: Feature Scaling ───────────────────────────────────
print("\n\nSTEP 5: Feature Scaling (StandardScaler)")
print("-" * 40)
print("WHY: CGPA is 5-10, internships is 0-3 — very different scales.")
print("     Scaling brings them to the same range for fair comparison.")

scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train, transform train
X_test_scaled  = scaler.transform(X_test)        # only transform test (never fit!)

print(f"\nBefore scaling — CGPA range: {X_train['cgpa'].min():.1f} to {X_train['cgpa'].max():.1f}")
print(f"After scaling  — CGPA range: {X_train_scaled[:,0].min():.2f} to {X_train_scaled[:,0].max():.2f}")
print("\nIMPORTANT: Always fit scaler on TRAIN data only, then transform test.")
print("           Fitting on test data = data leakage = wrong results!")


# ── STEP 6: Train the Model ───────────────────────────────────
print("\n\nSTEP 6: Training Linear Regression Model")
print("-" * 40)
print("Linear Regression finds the best line: salary = m1*cgpa + m2*internships + ...")

model = LinearRegression()
model.fit(X_train_scaled, y_train)    # This is the ENTIRE training process!

print(f"\nModel trained successfully!")
print(f"\nCoefficients (how much each feature contributes):")
for feature, coef in zip(X.columns, model.coef_):
    direction = "↑ increases" if coef > 0 else "↓ decreases"
    print(f"  {feature:<15}: {coef:+.4f}  ({direction} salary)")
print(f"  {'intercept':<15}: {model.intercept_:.4f}")


# ── STEP 7: Make Predictions ──────────────────────────────────
print("\n\nSTEP 7: Making Predictions")
print("-" * 40)

y_pred = model.predict(X_test_scaled)

print("Actual vs Predicted salary (first 10 students):")
print(f"\n  {'Student':<10} {'Actual (LPA)':>13} {'Predicted (LPA)':>16} {'Error':>8}")
print(f"  {'─'*10} {'─'*13} {'─'*16} {'─'*8}")
for i, (actual, pred) in enumerate(zip(y_test[:10], y_pred[:10])):
    error = actual - pred
    print(f"  {i+1:<10} {actual:>13.2f} {pred:>16.2f} {error:>+8.2f}")


# ── STEP 8: Evaluate the Model ────────────────────────────────
print("\n\nSTEP 8: Evaluating Model Performance")
print("-" * 40)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"MSE  (Mean Squared Error)   : {mse:.4f}")
print(f"RMSE (Root MSE)             : {rmse:.4f} LPA  ← average error")
print(f"MAE  (Mean Absolute Error)  : {mae:.4f} LPA")
print(f"R²   (R-squared score)      : {r2:.4f}")
print()
print("What these mean:")
print(f"  → On average, predictions are off by ₹{rmse:.2f} LPA")
print(f"  → R² of {r2:.2f} means model explains {r2*100:.1f}% of salary variation")
print(f"  → R² of 1.0 = perfect, 0.0 = useless model")

# Train vs test score — check for overfitting
train_pred = model.predict(X_train_scaled)
train_r2   = r2_score(y_train, train_pred)
print(f"\nTrain R²: {train_r2:.4f}   Test R²: {r2:.4f}")
gap = train_r2 - r2
if gap < 0.05:
    print("✓ Good! Train and test scores are close — no overfitting.")
else:
    print("⚠ Warning: Gap is large — possible overfitting.")


# ── STEP 9: Predict for a New Student ─────────────────────────
print("\n\nSTEP 9: Predicting salary for NEW students")
print("-" * 40)

new_students = pd.DataFrame({
    "cgpa"        : [9.2, 6.5, 7.8, 5.5],
    "internships" : [3,   0,   1,   0  ],
    "projects"    : [5,   2,   3,   1  ],
    "backlogs"    : [0,   2,   0,   4  ]
})

new_scaled   = scaler.transform(new_students)
new_pred     = model.predict(new_scaled)

print(f"\n{'Profile':<10} {'CGPA':>6} {'Intern':>7} {'Projects':>9} {'Backlogs':>9} {'Pred Salary':>12}")
print(f"{'─'*10} {'─'*6} {'─'*7} {'─'*9} {'─'*9} {'─'*12}")
profiles = ["Star student", "Average", "Good", "Struggling"]
for i, (_, row) in enumerate(new_students.iterrows()):
    print(f"{profiles[i]:<10} {row['cgpa']:>6} {row['internships']:>7} "
          f"{row['projects']:>9} {row['backlogs']:>9} ₹{new_pred[i]:>9.2f} LPA")


# ── STEP 10: Visualise ────────────────────────────────────────
print("\n\nSTEP 10: Saving visualisation")
print("-" * 40)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Day 2 — Linear Regression: Predicting Placement Salary", fontsize=13, fontweight="bold")

BLUE = "#378ADD"; GREEN = "#1D9E75"; ORANGE = "#BA7517"

# Chart 1: Actual vs Predicted
ax1 = axes[0]
ax1.scatter(y_test, y_pred, alpha=0.6, color=BLUE, s=40, edgecolors="none")
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect prediction")
ax1.set_xlabel("Actual Salary (LPA)")
ax1.set_ylabel("Predicted Salary (LPA)")
ax1.set_title(f"Actual vs Predicted\nR² = {r2:.3f}")
ax1.legend(fontsize=9)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Chart 2: Residuals (errors)
ax2 = axes[1]
residuals = y_test - y_pred
ax2.scatter(y_pred, residuals, alpha=0.6, color=ORANGE, s=40, edgecolors="none")
ax2.axhline(y=0, color="red", linestyle="--", linewidth=1.5)
ax2.set_xlabel("Predicted Salary (LPA)")
ax2.set_ylabel("Residual (Actual - Predicted)")
ax2.set_title("Residual Plot\n(should be random around 0)")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Chart 3: Feature importance (coefficients)
ax3 = axes[2]
features = list(X.columns)
coefs    = model.coef_
colors   = [GREEN if c > 0 else ORANGE for c in coefs]
bars = ax3.barh(features, coefs, color=colors, edgecolor="white")
ax3.axvline(x=0, color="gray", linewidth=0.8)
ax3.set_xlabel("Coefficient value")
ax3.set_title("Feature Importance\n(coefficient magnitude)")
for bar, coef in zip(bars, coefs):
    ax3.text(coef + (0.02 if coef > 0 else -0.02),
             bar.get_y() + bar.get_height()/2,
             f"{coef:+.3f}", va="center",
             ha="left" if coef > 0 else "right", fontsize=9)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("day2_linear_regression.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Chart saved as 'day2_linear_regression.png'")

print()
print("=" * 60)
print("Script 2 complete! Your first ML model is built!")
print(f"Model accuracy (R²): {r2:.2%}")
print("Key concepts covered:")
print("  ✓ Feature selection & target variable")
print("  ✓ Train/test split (and WHY it matters)")
print("  ✓ Feature scaling (StandardScaler)")
print("  ✓ Training a model (.fit)")
print("  ✓ Predictions (.predict)")
print("  ✓ Evaluation: RMSE, MAE, R²")
print("  ✓ Checking for overfitting")
print("=" * 60)
