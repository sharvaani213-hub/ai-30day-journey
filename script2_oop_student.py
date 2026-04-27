# ============================================================
# DAY 1 — SCRIPT 2: Object-Oriented Programming (OOP)
# Topics: classes, objects, __init__, methods, inheritance
# Goal : Student class with grade & is_passed methods
# ============================================================


# ── CLASS DEFINITION ─────────────────────────────────────────
class Student:
    """
    Represents a student with their marks and basic stats.

    Attributes:
        name    (str)   : Student's name
        roll_no (str)   : Roll number
        marks   (dict)  : Subject -> marks mapping
        branch  (str)   : Engineering branch
    """

    # Class variable — shared by ALL students
    passing_mark = 40
    total_subjects = 6

    def __init__(self, name, roll_no, branch, marks):
        """
        Constructor — runs automatically when you create a Student object.
        'self' refers to the specific object being created.
        """
        self.name    = name
        self.roll_no = roll_no
        self.branch  = branch
        self.marks   = marks        # dict: {"Maths": 85, "DSA": 90, ...}

    # ── Basic methods ─────────────────────────────────────────

    def total_marks(self):
        """Returns sum of all subject marks."""
        return sum(self.marks.values())

    def average(self):
        """Returns average marks rounded to 2 decimal places."""
        return round(self.total_marks() / len(self.marks), 2)

    def highest_subject(self):
        """Returns the subject with the highest marks."""
        return max(self.marks, key=self.marks.get)

    def lowest_subject(self):
        """Returns the subject with the lowest marks."""
        return min(self.marks, key=self.marks.get)

    def failed_subjects(self):
        """Returns list of subjects where marks < passing_mark."""
        return [sub for sub, mark in self.marks.items()
                if mark < Student.passing_mark]

    def is_passed(self):
        """Returns True if student passed ALL subjects."""
        return len(self.failed_subjects()) == 0

    def grade(self):
        """
        Returns grade based on average marks.
        Standard Indian university grading.
        """
        avg = self.average()
        if avg >= 90:
            return "O (Outstanding)"
        elif avg >= 80:
            return "A+ (Excellent)"
        elif avg >= 70:
            return "A (Very Good)"
        elif avg >= 60:
            return "B+ (Good)"
        elif avg >= 50:
            return "B (Average)"
        elif avg >= 40:
            return "C (Pass)"
        else:
            return "F (Fail)"

    def cgpa(self):
        """Converts average marks to approx CGPA on 10-point scale."""
        return round(self.average() / 10, 2)

    def report_card(self):
        """Prints a formatted report card for this student."""
        line = "─" * 45
        print(f"\n{line}")
        print(f"  REPORT CARD")
        print(f"{line}")
        print(f"  Name     : {self.name}")
        print(f"  Roll No  : {self.roll_no}")
        print(f"  Branch   : {self.branch}")
        print(f"{line}")
        print(f"  {'Subject':<20} {'Marks':>6}  {'Status':>8}")
        print(f"  {'─'*20} {'─'*6}  {'─'*8}")
        for subject, mark in self.marks.items():
            status = "PASS" if mark >= Student.passing_mark else "FAIL"
            marker = "✗" if status == "FAIL" else " "
            print(f"  {marker} {subject:<19} {mark:>6}  {status:>8}")
        print(f"{line}")
        print(f"  Total    : {self.total_marks()} / {len(self.marks) * 100}")
        print(f"  Average  : {self.average()}")
        print(f"  CGPA     : {self.cgpa()} / 10")
        print(f"  Grade    : {self.grade()}")
        print(f"  Result   : {'PASSED ✓' if self.is_passed() else 'FAILED ✗'}")
        if not self.is_passed():
            print(f"  Failed in: {', '.join(self.failed_subjects())}")
        print(f"  Best sub : {self.highest_subject()} ({self.marks[self.highest_subject()]})")
        print(f"  Weak sub : {self.lowest_subject()} ({self.marks[self.lowest_subject()]})")
        print(f"{line}\n")

    def __str__(self):
        """Called when you print a Student object — e.g. print(student1)"""
        return f"Student({self.name}, Roll: {self.roll_no}, Avg: {self.average()})"

    def __repr__(self):
        """Developer-friendly representation."""
        return f"Student(name='{self.name}', roll='{self.roll_no}')"


# ── INHERITANCE: Topper class extends Student ─────────────────
class Topper(Student):
    """
    A special Student who qualifies for a scholarship.
    Inherits everything from Student, adds scholarship logic.
    """

    SCHOLARSHIP_THRESHOLD = 80

    def __init__(self, name, roll_no, branch, marks, scholarship_amount):
        # Call parent class __init__ first
        super().__init__(name, roll_no, branch, marks)
        self.scholarship_amount = scholarship_amount

    def is_eligible_for_scholarship(self):
        return self.average() >= Topper.SCHOLARSHIP_THRESHOLD

    def report_card(self):
        # Call the parent's report_card, then add scholarship info
        super().report_card()
        if self.is_eligible_for_scholarship():
            print(f"  ★ Scholarship: ₹{self.scholarship_amount:,} awarded!\n")
        else:
            print(f"  Scholarship: Not eligible (need avg ≥ {self.SCHOLARSHIP_THRESHOLD})\n")


# ── CREATE STUDENT OBJECTS ────────────────────────────────────

student1 = Student(
    name    = "Arjun Sharma",
    roll_no = "20CS001",
    branch  = "Computer Engineering",
    marks   = {
        "Maths"    : 88,
        "DSA"      : 92,
        "OS"       : 74,
        "DBMS"     : 81,
        "Networks" : 69,
        "AI/ML"    : 95
    }
)

student2 = Student(
    name    = "Priya Reddy",
    roll_no = "20CS042",
    branch  = "Computer Engineering",
    marks   = {
        "Maths"    : 55,
        "DSA"      : 62,
        "OS"       : 38,      # FAIL
        "DBMS"     : 70,
        "Networks" : 48,
        "AI/ML"    : 59
    }
)

student3 = Student(
    name    = "Rahul Verma",
    roll_no = "20CS089",
    branch  = "Computer Engineering",
    marks   = {
        "Maths"    : 35,      # FAIL
        "DSA"      : 41,
        "OS"       : 52,
        "DBMS"     : 30,      # FAIL
        "Networks" : 28,      # FAIL
        "AI/ML"    : 45
    }
)

topper = Topper(
    name                = "Sneha Patel",
    roll_no             = "20CS007",
    branch              = "Computer Engineering",
    marks               = {
        "Maths"    : 97,
        "DSA"      : 99,
        "OS"       : 91,
        "DBMS"     : 94,
        "Networks" : 88,
        "AI/ML"    : 98
    },
    scholarship_amount  = 50000
)


# ── PRINT INDIVIDUAL REPORT CARDS ────────────────────────────
print("=" * 50)
print("  INDIVIDUAL STUDENT REPORT CARDS")
print("=" * 50)

student1.report_card()
student2.report_card()
student3.report_card()
topper.report_card()


# ── CLASS-LEVEL ANALYSIS ─────────────────────────────────────
print("=" * 50)
print("  CLASS ANALYSIS")
print("=" * 50)

all_students = [student1, student2, student3, topper]

# Sort by average descending — using lambda (from Script 1!)
ranked = sorted(all_students, key=lambda s: s.average(), reverse=True)

print(f"\n{'Rank':<6} {'Name':<20} {'Avg':>6}  {'Grade':<20} {'Pass?':>6}")
print(f"{'─'*6} {'─'*20} {'─'*6}  {'─'*20} {'─'*6}")
for rank, stu in enumerate(ranked, start=1):
    result = "Yes" if stu.is_passed() else "No"
    print(f"{rank:<6} {stu.name:<20} {stu.average():>6}  {stu.grade():<20} {result:>6}")

# Quick stats
averages = [s.average() for s in all_students]
class_avg = round(sum(averages) / len(averages), 2)
passed    = sum(1 for s in all_students if s.is_passed())

print(f"\nClass average : {class_avg}")
print(f"Pass rate     : {passed}/{len(all_students)} students")
print(f"Class topper  : {ranked[0].name} ({ranked[0].average()} avg)")
print(f"Needs help    : {ranked[-1].name} ({ranked[-1].average()} avg)")

# Using __str__
print("\n--- Student objects (using __str__) ---")
for s in all_students:
    print(f"  {s}")

print("\nScript 2 complete! OOP covered: classes, objects, inheritance.")
