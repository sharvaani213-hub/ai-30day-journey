# ============================================================
# DAY 1 — SCRIPT 3: File I/O
# Topics: write/read files, CSV with plain Python, csv module
# Goal : Create a CSV of marks, read it, find top scorer
# ============================================================

import csv
import os

# ── STEP 1: Create a CSV file using plain Python ─────────────
# We write it manually first so you understand what a CSV is

csv_filename = "student_marks.csv"

# Raw data — list of dicts (easy to read and maintain)
students_data = [
    {"name": "Arjun Sharma",   "roll": "20CS001", "maths": 88, "dsa": 92, "os": 74, "dbms": 81, "networks": 69, "aiml": 95},
    {"name": "Priya Reddy",    "roll": "20CS042", "maths": 55, "dsa": 62, "os": 38, "dbms": 70, "networks": 48, "aiml": 59},
    {"name": "Rahul Verma",    "roll": "20CS089", "maths": 35, "dsa": 41, "os": 52, "dbms": 30, "networks": 28, "aiml": 45},
    {"name": "Sneha Patel",    "roll": "20CS007", "maths": 97, "dsa": 99, "os": 91, "dbms": 94, "networks": 88, "aiml": 98},
    {"name": "Kiran Naidu",    "roll": "20CS055", "maths": 72, "dsa": 68, "os": 80, "dbms": 75, "networks": 66, "aiml": 83},
    {"name": "Divya Iyer",     "roll": "20CS033", "maths": 91, "dsa": 85, "os": 88, "dbms": 79, "networks": 93, "aiml": 87},
    {"name": "Mohit Singh",    "roll": "20CS071", "maths": 45, "dsa": 52, "os": 48, "dbms": 60, "networks": 55, "aiml": 50},
    {"name": "Ananya Rao",     "roll": "20CS018", "maths": 78, "dsa": 82, "os": 71, "dbms": 88, "networks": 76, "aiml": 90},
]

subjects = ["maths", "dsa", "os", "dbms", "networks", "aiml"]


# ── WRITE to CSV ──────────────────────────────────────────────
print("=" * 55)
print("STEP 1: Writing CSV file")
print("=" * 55)

# Method A: plain Python write (so you see what's inside)
with open(csv_filename, "w", newline="") as f:
    # Header row
    header = "name,roll," + ",".join(subjects) + "\n"
    f.write(header)

    # Data rows
    for s in students_data:
        row_values = [s["name"], s["roll"]] + [str(s[sub]) for sub in subjects]
        f.write(",".join(row_values) + "\n")

print(f"  Created '{csv_filename}' with {len(students_data)} student records.")
print(f"  File size: {os.path.getsize(csv_filename)} bytes\n")


# ── READ with plain Python (no library) ──────────────────────
print("=" * 55)
print("STEP 2: Reading CSV with plain Python (no csv module)")
print("=" * 55)

raw_records = []

with open(csv_filename, "r") as f:
    lines = f.readlines()               # read all lines into a list

header_line = lines[0].strip().split(",")
print(f"  Columns found: {header_line}")
print()

for line in lines[1:]:                  # skip header
    values = line.strip().split(",")    # split on comma
    record = dict(zip(header_line, values))   # pair headers with values
    raw_records.append(record)

print(f"  Loaded {len(raw_records)} records.")
print(f"\n  First record (raw dict):")
for k, v in raw_records[0].items():
    print(f"    {k}: {v}")
print()


# ── READ with csv module (the right way) ─────────────────────
print("=" * 55)
print("STEP 3: Reading CSV with Python's csv module")
print("=" * 55)

students = []

with open(csv_filename, "r") as f:
    reader = csv.DictReader(f)          # auto-parses header

    for row in reader:
        # Convert mark strings to integers
        student = {
            "name"  : row["name"],
            "roll"  : row["roll"],
            "marks" : {sub: int(row[sub]) for sub in subjects}
        }
        # Compute total & average
        student["total"]   = sum(student["marks"].values())
        student["average"] = round(student["total"] / len(subjects), 2)
        students.append(student)

print(f"  Loaded {len(students)} students with marks converted to integers.\n")


# ── ANALYSIS ─────────────────────────────────────────────────
print("=" * 55)
print("STEP 4: Analysing the data")
print("=" * 55)

# Sort by average descending
students_sorted = sorted(students, key=lambda s: s["average"], reverse=True)

# Top scorer
top_scorer = students_sorted[0]
print(f"\n  TOP SCORER: {top_scorer['name']} (Roll: {top_scorer['roll']})")
print(f"  Total    : {top_scorer['total']} / {len(subjects) * 100}")
print(f"  Average  : {top_scorer['average']}")

# Lowest scorer
low_scorer = students_sorted[-1]
print(f"\n  LOWEST  : {low_scorer['name']} — Avg: {low_scorer['average']}")

# Class average
class_avg = round(sum(s["average"] for s in students) / len(students), 2)
print(f"\n  Class average: {class_avg}")

# Pass/fail (passing = avg >= 40)
passed = [s for s in students if s["average"] >= 40]
failed = [s for s in students if s["average"] < 40]
print(f"  Passed: {len(passed)} students")
print(f"  Failed: {len(failed)} students")

# Subject-wise averages
print("\n  Subject-wise class averages:")
for sub in subjects:
    sub_avg = round(sum(s["marks"][sub] for s in students) / len(students), 2)
    bar = "█" * int(sub_avg // 5)
    print(f"    {sub:<12}: {sub_avg:>6}  {bar}")

# Full rankings table
print("\n  FULL RANKINGS:")
print(f"\n  {'Rank':<5} {'Name':<18} {'Roll':<10} {'Total':>6}  {'Avg':>6}")
print(f"  {'─'*5} {'─'*18} {'─'*10} {'─'*6}  {'─'*6}")
for rank, stu in enumerate(students_sorted, start=1):
    print(f"  {rank:<5} {stu['name']:<18} {stu['roll']:<10} {stu['total']:>6}  {stu['average']:>6}")


# ── WRITE a results CSV ───────────────────────────────────────
print()
print("=" * 55)
print("STEP 5: Writing a results summary CSV")
print("=" * 55)

results_file = "results_summary.csv"

with open(results_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["rank", "name", "roll", "total", "average", "result"])
    writer.writeheader()

    for rank, stu in enumerate(students_sorted, start=1):
        writer.writerow({
            "rank"   : rank,
            "name"   : stu["name"],
            "roll"   : stu["roll"],
            "total"  : stu["total"],
            "average": stu["average"],
            "result" : "PASS" if stu["average"] >= 40 else "FAIL"
        })

print(f"\n  Results written to '{results_file}'")
print(f"  Open it in Excel or Google Sheets to see it formatted!\n")

print("=" * 55)
print("Script 3 complete! File I/O + CSV handling covered.")
print("=" * 55)
