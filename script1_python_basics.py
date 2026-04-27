# ============================================================
# DAY 1 — SCRIPT 1: Python Basics
# Topics: variables, strings, lists, dicts, loops, functions
# Goal : a function that returns mean, max, min of a list
# ============================================================

# ── 1. Variables & data types ────────────────────────────────
name    = "Arjun"          # string
age     = 22               # integer
cgpa    = 8.5              # float
is_placed = False          # boolean

print("=" * 50)
print("SECTION 1: Variables & Data Types")
print("=" * 50)
print(f"Name    : {name}")
print(f"Age     : {age}")
print(f"CGPA    : {cgpa}")
print(f"Placed? : {is_placed}")
print()

# ── 2. Strings ───────────────────────────────────────────────
print("=" * 50)
print("SECTION 2: String Operations")
print("=" * 50)

greeting = "Hello, World!"
print(f"Original     : {greeting}")
print(f"Uppercase    : {greeting.upper()}")
print(f"Lowercase    : {greeting.lower()}")
print(f"Length       : {len(greeting)}")
print(f"Replace      : {greeting.replace('World', 'Python')}")
print(f"Slice [0:5]  : {greeting[0:5]}")
print(f"Split on ',' : {greeting.split(',')}")
print()

# ── 3. Lists ─────────────────────────────────────────────────
print("=" * 50)
print("SECTION 3: Lists")
print("=" * 50)

marks = [85, 92, 78, 95, 88, 76, 91]
print(f"Marks list    : {marks}")
print(f"First mark    : {marks[0]}")
print(f"Last mark     : {marks[-1]}")
print(f"Slice [1:4]   : {marks[1:4]}")

marks.append(89)           # add to end
marks.remove(76)           # remove value
marks.sort()               # sort in place
print(f"After sort    : {marks}")
print(f"Reversed      : {marks[::-1]}")

# List comprehension — squares of all marks above 85
high_marks = [m for m in marks if m > 85]
print(f"Marks > 85    : {high_marks}")
print()

# ── 4. Dictionaries ──────────────────────────────────────────
print("=" * 50)
print("SECTION 4: Dictionaries")
print("=" * 50)

student = {
    "name"    : "Arjun",
    "age"     : 22,
    "cgpa"    : 8.5,
    "branch"  : "Computer Engineering",
    "skills"  : ["Python", "C++", "SQL"]
}

print(f"Student name  : {student['name']}")
print(f"Branch        : {student['branch']}")
print(f"Skills        : {student['skills']}")

# Add a new key
student["city"] = "Hyderabad"
print(f"All keys      : {list(student.keys())}")
print(f"All values    : {list(student.values())}")
print()

# Loop through dict
print("--- Looping through student dict ---")
for key, value in student.items():
    print(f"  {key:10} : {value}")
print()

# ── 5. Loops ─────────────────────────────────────────────────
print("=" * 50)
print("SECTION 5: Loops")
print("=" * 50)

# for loop
subjects = ["Maths", "DSA", "OS", "DBMS", "AI"]
print("Subjects (for loop):")
for i, sub in enumerate(subjects, start=1):
    print(f"  {i}. {sub}")

# while loop
print("\nCountdown (while loop):")
count = 5
while count > 0:
    print(f"  {count}...")
    count -= 1
print("  Go!")
print()

# ── 6. Functions ─────────────────────────────────────────────
print("=" * 50)
print("SECTION 6: Functions")
print("=" * 50)

def greet_student(name, course="AI/ML"):
    """Returns a welcome message for the student."""
    return f"Welcome {name}! You are learning {course}."

def calculate_stats(numbers):
    """
    Takes a list of numbers.
    Returns a dict with mean, maximum, minimum, and total.
    This is the CORE function of Script 1.
    """
    if not numbers:
        return None

    total   = sum(numbers)
    mean    = total / len(numbers)
    maximum = max(numbers)
    minimum = min(numbers)

    return {
        "count"  : len(numbers),
        "total"  : total,
        "mean"   : round(mean, 2),
        "maximum": maximum,
        "minimum": minimum,
        "range"  : maximum - minimum
    }

# Test greet function
print(greet_student("Arjun"))
print(greet_student("Priya", "Deep Learning"))
print()

# Test stats function with different datasets
datasets = {
    "Exam marks"    : [85, 92, 78, 95, 88, 76, 91, 89],
    "Monthly hours" : [120, 145, 98, 160, 135],
    "Salary (LPA)"  : [8, 12, 10, 15, 9, 11]
}

for label, data in datasets.items():
    stats = calculate_stats(data)
    print(f"--- {label} ---")
    for key, val in stats.items():
        print(f"  {key:10}: {val}")
    print()

# ── 7. Bonus: List comprehensions + lambda ───────────────────
print("=" * 50)
print("SECTION 7: List Comprehensions & Lambda")
print("=" * 50)

numbers = list(range(1, 11))   # [1, 2, 3, ..., 10]

squares      = [n ** 2 for n in numbers]
evens        = [n for n in numbers if n % 2 == 0]
even_squares = [n ** 2 for n in numbers if n % 2 == 0]

print(f"Numbers      : {numbers}")
print(f"Squares      : {squares}")
print(f"Evens only   : {evens}")
print(f"Even squares : {even_squares}")

# Lambda = tiny one-line function
double = lambda x: x * 2
is_even = lambda x: x % 2 == 0

print(f"\ndouble(7)    : {double(7)}")
print(f"is_even(4)   : {is_even(4)}")
print(f"is_even(7)   : {is_even(7)}")

print()
print("=" * 50)
print("Script 1 complete! All basics covered.")
print("=" * 50)