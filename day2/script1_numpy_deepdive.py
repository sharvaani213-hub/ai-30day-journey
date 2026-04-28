# ============================================================
# DAY 2 — SCRIPT 1: NumPy Deep Dive
# Topics: arrays, indexing, broadcasting, linear algebra,
#         random module, reshaping — all used in real ML
# ============================================================

import numpy as np

print("NumPy version:", np.__version__)
print("=" * 55)


# ── SECTION 1: Creating Arrays (many ways) ───────────────────
print("\nSECTION 1: Creating Arrays")
print("-" * 40)

a = np.array([1, 2, 3, 4, 5])                  # from list
b = np.zeros((3, 4))                            # all zeros  — used to init weights in ML
c = np.ones((2, 3))                             # all ones
d = np.eye(4)                                   # identity matrix
e = np.arange(0, 20, 2)                         # like range() → [0,2,4,...18]
f = np.linspace(0, 1, 6)                        # 6 evenly spaced between 0 and 1
g = np.full((3, 3), 7)                          # fill with a constant

print(f"np.array       : {a}")
print(f"np.zeros(3x4):\n{b}")
print(f"np.ones(2x3):\n{c}")
print(f"np.eye(4):\n{d}")
print(f"np.arange      : {e}")
print(f"np.linspace    : {f}")
print(f"np.full(3x3,7):\n{g}")


# ── SECTION 2: Array Properties ──────────────────────────────
print("\nSECTION 2: Array Properties")
print("-" * 40)

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(f"Matrix:\n{matrix}")
print(f"Shape    : {matrix.shape}")      # (rows, cols) — used constantly in ML
print(f"Ndim     : {matrix.ndim}")       # number of dimensions
print(f"Size     : {matrix.size}")       # total elements
print(f"Dtype    : {matrix.dtype}")      # data type
print(f"Itemsize : {matrix.itemsize} bytes per element")


# ── SECTION 3: Indexing & Slicing ────────────────────────────
print("\nSECTION 3: Indexing & Slicing")
print("-" * 40)

arr = np.array([[10, 20, 30, 40],
                [50, 60, 70, 80],
                [90, 100, 110, 120]])

print(f"Array:\n{arr}")
print(f"Row 0          : {arr[0]}")
print(f"Row 1, Col 2   : {arr[1, 2]}")          # = 70
print(f"All rows, Col 1: {arr[:, 1]}")           # = [20, 60, 100]
print(f"Rows 0-1       : \n{arr[0:2]}")
print(f"Bottom-right 2x2:\n{arr[1:, 2:]}")       # submatrix

# Boolean indexing — very common in ML for filtering
print(f"\nElements > 50  : {arr[arr > 50]}")
print(f"Elements % 20==0: {arr[arr % 20 == 0]}")

# Fancy indexing
rows = np.array([0, 2])
cols = np.array([1, 3])
print(f"Fancy index [0,1] and [2,3]: {arr[rows, cols]}")  # [20, 120]


# ── SECTION 4: Reshaping — Critical for ML ───────────────────
print("\nSECTION 4: Reshaping (Critical for ML)")
print("-" * 40)

flat = np.arange(1, 25)                         # 24 elements
print(f"Flat array (24 elements): {flat}")

r1 = flat.reshape(4, 6)                         # 4 rows, 6 cols
r2 = flat.reshape(2, 3, 4)                      # 3D — like image batches
r3 = flat.reshape(24, 1)                        # column vector
r4 = flat.reshape(1, 24)                        # row vector

print(f"\nReshape to (4,6):\n{r1}")
print(f"\nReshape to (2,3,4) [3D]:\n{r2}")
print(f"\nReshape to (24,1) — column vector:\n{r3.T}")  # .T = transpose

# -1 lets NumPy figure out the dimension automatically
auto = flat.reshape(-1, 4)                      # NumPy calculates rows = 6
print(f"\nReshape(-1, 4) — auto rows:\n{auto}")

# Flatten back
print(f"\nFlatten back: {r1.flatten()}")


# ── SECTION 5: Broadcasting ───────────────────────────────────
print("\nSECTION 5: Broadcasting (NumPy's Superpower)")
print("-" * 40)
print("Broadcasting = apply operations between arrays of different shapes")

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

scalar = 10
row    = np.array([1, 2, 3])       # shape (3,)
col    = np.array([[1], [2], [3]]) # shape (3,1)

print(f"\nMatrix A:\n{A}")
print(f"\nA + scalar(10):\n{A + scalar}")      # adds 10 to EVERY element
print(f"\nA + row[1,2,3]:\n{A + row}")          # adds row to EACH row of A
print(f"\nA + col[[1],[2],[3]]:\n{A + col}")    # adds col to EACH column of A
print(f"\nA * A (element-wise):\n{A * A}")      # NOT matrix multiply


# ── SECTION 6: Linear Algebra — Heart of ML ──────────────────
print("\nSECTION 6: Linear Algebra (Heart of ML)")
print("-" * 40)

X = np.array([[1, 2],
              [3, 4],
              [5, 6]])   # shape (3, 2) — think: 3 samples, 2 features

W = np.array([[0.5, 0.1, 0.3],
              [0.2, 0.4, 0.6]])  # shape (2, 3) — weight matrix

# Matrix multiplication — this IS forward pass in neural networks
result = X @ W                   # shape (3, 3)
print(f"X (3x2):\n{X}")
print(f"W (2x3):\n{W}")
print(f"X @ W  (3x3) — matrix multiply:\n{result}")

# Dot product
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(f"\nDot product {v1} · {v2} = {np.dot(v1, v2)}")   # 1*4+2*5+3*6 = 32

# Transpose
print(f"\nTranspose of X:\n{X.T}")              # flips rows and cols

# Determinant & inverse (square matrix only)
sq = np.array([[2, 1], [5, 3]])
print(f"\nSquare matrix:\n{sq}")
print(f"Determinant  : {np.linalg.det(sq):.2f}")
print(f"Inverse:\n{np.linalg.inv(sq)}")

# Eigenvalues — used in PCA (dimensionality reduction)
eigvals, eigvecs = np.linalg.eig(sq)
print(f"\nEigenvalues  : {eigvals.round(3)}")


# ── SECTION 7: Random Module — Used in All ML ────────────────
print("\nSECTION 7: Random Module")
print("-" * 40)

np.random.seed(42)      # seed = reproducible results (always set this in ML!)

print(f"Random float [0,1)      : {np.random.rand():.4f}")
print(f"Random array (3,3):\n{np.random.rand(3,3).round(3)}")
print(f"Random integers [0,10)  : {np.random.randint(0, 10, size=8)}")
print(f"Normal dist (mean=0,std=1): {np.random.randn(5).round(3)}")
print(f"Normal (mean=70, std=10) : {np.random.normal(70, 10, 5).round(1)}")

# Shuffle & choice — used in train/test split
data = np.arange(1, 11)
np.random.shuffle(data)
print(f"\nShuffled [1..10]   : {data}")
print(f"Random choice of 3 : {np.random.choice(data, size=3, replace=False)}")


# ── SECTION 8: Aggregations & Axis Operations ─────────────────
print("\nSECTION 8: Aggregations & Axis Operations")
print("-" * 40)

scores = np.array([[85, 92, 78, 95],   # Student 1
                   [72, 68, 80, 75],   # Student 2
                   [91, 88, 94, 82],   # Student 3
                   [60, 55, 70, 65]])  # Student 4

print(f"Scores matrix (4 students x 4 subjects):\n{scores}")
print(f"\nOverall mean  : {np.mean(scores):.2f}")
print(f"Student avgs (axis=1): {np.mean(scores, axis=1)}")   # across columns
print(f"Subject avgs (axis=0): {np.mean(scores, axis=0)}")   # across rows
print(f"Max per student      : {np.max(scores, axis=1)}")
print(f"Min per subject      : {np.min(scores, axis=0)}")
print(f"Sum of all scores    : {np.sum(scores)}")

# Sorting
print(f"\nSorted scores (each row): \n{np.sort(scores, axis=1)}")
print(f"Index of max per student: {np.argmax(scores, axis=1)}")  # which subject


# ── SECTION 9: Real ML Use Case — Manual Linear Regression ───
print("\nSECTION 9: Real ML — Manual Linear Regression with NumPy")
print("-" * 40)
print("Predicting salary (LPA) from years of experience")

np.random.seed(0)
experience = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
salary     = experience * 1.2 + np.random.normal(0, 0.5, 10) + 3

# Linear regression formula: y = mx + b
# Using least squares: m = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²
x_mean = np.mean(experience)
y_mean = np.mean(salary)

m = np.sum((experience - x_mean) * (salary - y_mean)) / np.sum((experience - x_mean)**2)
b = y_mean - m * x_mean

print(f"Slope (m)     : {m:.4f}")
print(f"Intercept (b) : {b:.4f}")
print(f"Equation      : salary = {m:.2f} * experience + {b:.2f}")

# Predictions
predictions = m * experience + b
errors      = salary - predictions
mse         = np.mean(errors**2)
rmse        = np.sqrt(mse)

print(f"\nPredictions   : {predictions.round(2)}")
print(f"Actual        : {salary.round(2)}")
print(f"MSE           : {mse:.4f}")
print(f"RMSE          : {rmse:.4f} LPA average error")

# Predict for new values
for exp in [3, 5, 12]:
    pred = m * exp + b
    print(f"  {exp} years experience → ₹{pred:.2f} LPA predicted salary")

print()
print("=" * 55)
print("Script 1 complete! NumPy deep dive done.")
print("Key takeaway: NumPy arrays + matrix ops = foundation of ALL ML")
print("=" * 55)
