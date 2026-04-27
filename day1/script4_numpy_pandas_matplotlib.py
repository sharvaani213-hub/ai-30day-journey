# ============================================================
# DAY 1 — SCRIPT 4: NumPy + Pandas + Matplotlib
# Topics: arrays, dataframes, data cleaning, visualisation
# Goal : Analyse student data, plot survival-rate style charts
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

print("Libraries loaded successfully!")
print(f"NumPy  version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print()


# ╔══════════════════════════════════════════════════════════╗
# ║  PART A — NumPy Fundamentals                            ║
# ╚══════════════════════════════════════════════════════════╝

print("=" * 55)
print("PART A: NumPy — Why It Beats Plain Python Loops")
print("=" * 55)

# Plain Python list vs NumPy array
py_list  = [85, 92, 78, 95, 88, 76, 91, 89, 84, 97]
np_array = np.array(py_list)

print(f"\nPython list : {py_list}")
print(f"NumPy array : {np_array}")
print(f"Data type   : {np_array.dtype}")
print(f"Shape       : {np_array.shape}")

# Element-wise operations — NumPy does this in ONE line
print("\n--- Element-wise operations ---")
print(f"  +10 to all  : {np_array + 10}")
print(f"  *2 all marks: {np_array * 2}")
print(f"  Above 85    : {np_array[np_array > 85]}")   # Boolean indexing

# Statistical functions
print("\n--- NumPy statistics ---")
print(f"  Mean   : {np.mean(np_array):.2f}")
print(f"  Std dev: {np.std(np_array):.2f}")
print(f"  Max    : {np.max(np_array)}")
print(f"  Min    : {np.min(np_array)}")
print(f"  Median : {np.median(np_array)}")
print(f"  Sum    : {np.sum(np_array)}")
print(f"  Percentile 75: {np.percentile(np_array, 75)}")

# 2D arrays — basis of matrix math in ML
print("\n--- 2D arrays (like a matrix) ---")
marks_2d = np.array([
    [85, 92, 78],   # Student 1 — 3 subjects
    [55, 62, 38],   # Student 2
    [97, 99, 91],   # Student 3
    [72, 68, 80],   # Student 4
])
print(f"Shape          : {marks_2d.shape}  (4 students, 3 subjects)")
print(f"Student 1 marks: {marks_2d[0]}")
print(f"Subject 2 marks: {marks_2d[:, 1]}")     # all students, column 1
print(f"Row averages   : {np.mean(marks_2d, axis=1).round(2)}")  # per student
print(f"Col averages   : {np.mean(marks_2d, axis=0).round(2)}")  # per subject


# ╔══════════════════════════════════════════════════════════╗
# ║  PART B — Pandas: Load + Explore + Clean Data           ║
# ╚══════════════════════════════════════════════════════════╝

print("\n" + "=" * 55)
print("PART B: Pandas — Data Analysis")
print("=" * 55)

# ── Create a realistic student dataset (Titanic-style structure) ──
# In a real project you'd do: df = pd.read_csv("titanic.csv")
# Here we build one so the script is self-contained

np.random.seed(42)   # reproducible random numbers

n = 120   # 120 students

data = {
    "name"        : [f"Student_{i:03d}" for i in range(1, n + 1)],
    "gender"      : np.random.choice(["Male", "Female"], n, p=[0.55, 0.45]),
    "branch"      : np.random.choice(["CS", "EC", "ME", "CE", "EE"], n,
                                      p=[0.35, 0.25, 0.15, 0.15, 0.10]),
    "year"        : np.random.choice([1, 2, 3, 4], n),
    "hostel"      : np.random.choice([True, False], n, p=[0.60, 0.40]),
    "cgpa"        : np.round(np.random.normal(7.2, 1.1, n).clip(4.0, 10.0), 2),
    "attendance"  : np.round(np.random.normal(78, 12, n).clip(40, 100), 1),
    "placed"      : None    # will compute below
}

df = pd.DataFrame(data)

# Add some missing values (real data is messy!)
missing_idx = np.random.choice(df.index, size=8, replace=False)
df.loc[missing_idx, "cgpa"] = np.nan

missing_att = np.random.choice(df.index, size=5, replace=False)
df.loc[missing_att, "attendance"] = np.nan

# Placement depends on CGPA + attendance (realistic logic)
df["placed"] = (
    (df["cgpa"].fillna(0) >= 6.5) &
    (df["attendance"].fillna(0) >= 65)
).astype(int)   # 1 = placed, 0 = not placed

print("\n--- Basic exploration ---")
print(f"\ndf.shape     : {df.shape}   (rows, columns)")
print(f"\ndf.dtypes:\n{df.dtypes}")
print(f"\ndf.head(5):\n{df.head(5).to_string()}")

print(f"\n--- Missing values ---")
print(df.isnull().sum())

print(f"\n--- Basic statistics (df.describe()) ---")
print(df[["cgpa", "attendance"]].describe().round(2))

# ── Data Cleaning ─────────────────────────────────────────────
print("\n--- Data cleaning ---")
before = df.isnull().sum().sum()

# Fill missing CGPA with the mean
df["cgpa"].fillna(df["cgpa"].mean(), inplace=True)

# Fill missing attendance with median
df["attendance"].fillna(df["attendance"].median(), inplace=True)

after = df.isnull().sum().sum()
print(f"  Missing values before: {before}")
print(f"  Missing values after : {after}")

# ── Analysis ──────────────────────────────────────────────────
print("\n--- Key insights ---")

placement_rate = df["placed"].mean() * 100
print(f"  Overall placement rate: {placement_rate:.1f}%")

print(f"\n  Placement by branch:")
branch_stats = df.groupby("branch").agg(
    total    = ("placed", "count"),
    placed   = ("placed", "sum"),
    avg_cgpa = ("cgpa",   "mean")
).round(2)
branch_stats["placement_pct"] = (branch_stats["placed"] / branch_stats["total"] * 100).round(1)
print(branch_stats.to_string())

print(f"\n  Placement by gender:")
gender_stats = df.groupby("gender")["placed"].agg(["sum", "count", "mean"])
gender_stats.columns = ["placed", "total", "rate"]
gender_stats["rate"] = (gender_stats["rate"] * 100).round(1)
print(gender_stats.to_string())

print(f"\n  Hostel vs Day scholar placement:")
hostel_stats = df.groupby("hostel")["placed"].mean() * 100
hostel_stats.index = ["Day Scholar", "Hostel"]
print(hostel_stats.round(1).to_string())


# ╔══════════════════════════════════════════════════════════╗
# ║  PART C — Matplotlib: 4 Charts in one figure            ║
# ╚══════════════════════════════════════════════════════════╝

print("\n" + "=" * 55)
print("PART C: Matplotlib — Visualisation")
print("=" * 55)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Day 1 — Student Data Analysis Dashboard", fontsize=15, fontweight="bold", y=0.98)

BLUE   = "#378ADD"
GREEN  = "#1D9E75"
ORANGE = "#BA7517"
PURPLE = "#7F77DD"
GRAY   = "#888780"
RED    = "#E24B4A"

# ── Chart 1: Placement rate by branch (bar chart) ─────────────
ax1 = axes[0, 0]
branches = branch_stats.index.tolist()
rates    = branch_stats["placement_pct"].tolist()
colors   = [BLUE if r >= 70 else ORANGE if r >= 50 else RED for r in rates]

bars = ax1.bar(branches, rates, color=colors, edgecolor="white", linewidth=0.5)
ax1.set_title("Placement Rate by Branch", fontsize=12, fontweight="bold")
ax1.set_ylabel("Placement %")
ax1.set_ylim(0, 100)
ax1.axhline(y=70, color=GRAY, linestyle="--", linewidth=0.8, label="70% target")
ax1.legend(fontsize=9)

for bar, rate in zip(bars, rates):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
             f"{rate}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ── Chart 2: CGPA distribution (histogram) ────────────────────
ax2 = axes[0, 1]
placed_cgpa   = df[df["placed"] == 1]["cgpa"]
unplaced_cgpa = df[df["placed"] == 0]["cgpa"]

ax2.hist(placed_cgpa,   bins=15, alpha=0.7, color=GREEN,  label="Placed",     edgecolor="white")
ax2.hist(unplaced_cgpa, bins=15, alpha=0.7, color=RED,    label="Not placed", edgecolor="white")
ax2.axvline(x=6.5, color=GRAY, linestyle="--", linewidth=1.2, label="6.5 cutoff")
ax2.set_title("CGPA Distribution: Placed vs Not Placed", fontsize=12, fontweight="bold")
ax2.set_xlabel("CGPA")
ax2.set_ylabel("Number of students")
ax2.legend(fontsize=9)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# ── Chart 3: CGPA vs Attendance scatter plot ──────────────────
ax3 = axes[1, 0]
scatter_colors = [GREEN if p == 1 else RED for p in df["placed"]]
ax3.scatter(df["attendance"], df["cgpa"], c=scatter_colors, alpha=0.6, s=30, edgecolors="none")
ax3.axvline(x=65, color=GRAY, linestyle="--", linewidth=0.8)
ax3.axhline(y=6.5, color=GRAY, linestyle="--", linewidth=0.8)
ax3.set_title("CGPA vs Attendance (Placement Outcome)", fontsize=12, fontweight="bold")
ax3.set_xlabel("Attendance %")
ax3.set_ylabel("CGPA")
legend_elements = [
    mpatches.Patch(color=GREEN, label="Placed"),
    mpatches.Patch(color=RED,   label="Not placed")
]
ax3.legend(handles=legend_elements, fontsize=9)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# ── Chart 4: Placement by gender (grouped bar) ────────────────
ax4 = axes[1, 1]
genders = ["Male", "Female"]
placed_counts   = [df[(df["gender"] == g) & (df["placed"] == 1)].shape[0] for g in genders]
unplaced_counts = [df[(df["gender"] == g) & (df["placed"] == 0)].shape[0] for g in genders]

x      = np.arange(len(genders))
width  = 0.35
bars1 = ax4.bar(x - width / 2, placed_counts,   width, label="Placed",     color=BLUE,   edgecolor="white")
bars2 = ax4.bar(x + width / 2, unplaced_counts, width, label="Not placed", color=PURPLE, edgecolor="white")

ax4.set_title("Placement Count by Gender", fontsize=12, fontweight="bold")
ax4.set_xticks(x)
ax4.set_xticklabels(genders)
ax4.set_ylabel("Number of students")
ax4.legend(fontsize=9)

for bar in list(bars1) + list(bars2):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             str(int(bar.get_height())), ha="center", va="bottom", fontsize=10)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

plt.tight_layout()

# Save the chart
output_path = "day1_analysis_dashboard.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"\n  Chart saved as '{output_path}'")
print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")

print("\n" + "=" * 55)
print("Script 4 complete! NumPy + Pandas + Matplotlib done.")
print("=" * 55)
print()
print("  What you built today:")
print("  1. Loaded and cleaned a 120-row dataset")
print("  2. Handled missing values (fillna)")
print("  3. Grouped and aggregated data (groupby)")
print("  4. Plotted 4 different chart types")
print("  5. Saved a professional dashboard PNG")
print()
print("  Push everything to GitHub. Post on LinkedIn. You're 1/30 done!")
