# -*- coding: utf-8 -*-
# ============================================================
# DAY 8 -- SCRIPT 3: Real Project -- Job Application Tracker
# Topics: building a full-stack AI app, Streamlit + AI features,
#         local data storage, AI-powered job analysis
# This is a REAL portfolio project!
# ============================================================

import json
import os
import datetime
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("DAY 8 -- SCRIPT 3: Real Project -- Job Application Tracker")
print("=" * 60)

print("""
WHAT WE ARE BUILDING
---------------------
A complete Job Application Tracker with AI features.

Features:
  -> Add job applications (company, role, date, status)
  -> AI analyzes each job description automatically
  -> Extracts required skills from job description
  -> Compares with your skills -> shows match score
  -> Tracks application status (applied, interview, offer, rejected)
  -> Dashboard with statistics and charts
  -> Export data to CSV

Why this is a great portfolio project:
  -> Solves a REAL problem you actually have right now
  -> Uses AI in a practical way (not just a chatbot demo)
  -> Shows full-stack thinking (data + UI + AI)
  -> Recruiters understand it immediately
  -> You use it yourself during your job search
""")


# -- SECTION 1: Core Data Logic ------------------------------
print("SECTION 1: Core Data Logic")
print("-" * 40)

class JobApplicationTracker:
    """Core tracker logic -- works without any API."""

    # Skills database for matching
    AI_SKILLS = [
        "python", "machine learning", "deep learning", "nlp",
        "langchain", "rag", "openai", "huggingface", "tensorflow",
        "pytorch", "scikit-learn", "numpy", "pandas", "fastapi",
        "streamlit", "docker", "git", "sql", "transformers",
        "llm", "vector database", "chromadb", "pinecone",
        "prompt engineering", "fine-tuning", "mlops", "aws",
        "gcp", "azure", "kubernetes", "spark", "hadoop",
        "computer vision", "object detection", "bert", "gpt",
        "agents", "langraph", "llamaindex", "embeddings",
    ]

    def __init__(self, data_file="job_applications.json"):
        self.data_file   = data_file
        self.applications= self.load()

    def load(self):
        """Load applications from JSON file."""
        if os.path.exists(self.data_file):
            with open(self.data_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save(self):
        """Save applications to JSON file."""
        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(self.applications, f, indent=2, ensure_ascii=False)

    def add_application(self, company, role, job_desc, your_skills,
                        location="", salary_range="", job_url="", notes=""):
        """Add a new job application."""
        app_id   = len(self.applications) + 1
        analysis = self.analyze_job(job_desc, your_skills)

        application = {
            "id"           : app_id,
            "company"      : company,
            "role"         : role,
            "location"     : location,
            "salary_range" : salary_range,
            "job_url"      : job_url,
            "job_desc"     : job_desc,
            "your_skills"  : your_skills,
            "notes"        : notes,
            "status"       : "Applied",
            "applied_date" : datetime.datetime.now().strftime("%Y-%m-%d"),
            "last_updated" : datetime.datetime.now().strftime("%Y-%m-%d"),
            "analysis"     : analysis,
            "interviews"   : [],
            "offer"        : None
        }

        self.applications.append(application)
        self.save()
        return application

    def analyze_job(self, job_desc, your_skills):
        """Extract skills and calculate match score."""
        job_lower   = job_desc.lower()
        your_lower  = your_skills.lower()

        # Extract required skills from job description
        required = [s for s in self.AI_SKILLS if s in job_lower]

        # Find skills you have
        you_have = [s for s in required if s in your_lower]
        missing  = [s for s in required if s not in your_lower]

        # Calculate match score
        match_score = round(len(you_have) / len(required) * 100, 1) if required else 0

        # Estimate difficulty
        senior_keywords = ["senior", "lead", "principal", "architect", "5+ years", "7+ years"]
        is_senior       = any(kw in job_lower for kw in senior_keywords)

        # Detect job type
        job_types = {
            "AI Engineer"       : ["ai engineer", "artificial intelligence"],
            "ML Engineer"       : ["ml engineer", "machine learning engineer"],
            "Data Scientist"    : ["data scientist"],
            "NLP Engineer"      : ["nlp", "natural language"],
            "GenAI Developer"   : ["genai", "generative ai", "llm"],
            "Prompt Engineer"   : ["prompt engineer"],
        }
        detected_type = "AI/ML Role"
        for jtype, keywords in job_types.items():
            if any(kw in job_lower for kw in keywords):
                detected_type = jtype
                break

        return {
            "required_skills": required,
            "you_have"       : you_have,
            "missing"        : missing,
            "match_score"    : match_score,
            "is_senior"      : is_senior,
            "job_type"       : detected_type,
            "total_required" : len(required),
            "recommendation" : self.get_recommendation(match_score, is_senior)
        }

    def get_recommendation(self, match_score, is_senior):
        """Get application recommendation."""
        if is_senior and match_score < 60:
            return "Skip -- senior role, experience gap too large"
        elif match_score >= 80:
            return "Apply immediately -- strong match!"
        elif match_score >= 60:
            return "Good match -- apply with tailored resume"
        elif match_score >= 40:
            return "Decent match -- apply and highlight transferable skills"
        else:
            return "Weak match -- focus on learning missing skills first"

    def update_status(self, app_id, new_status, notes=""):
        """Update application status."""
        for app in self.applications:
            if app["id"] == app_id:
                app["status"]       = new_status
                app["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d")
                if notes:
                    app["notes"] += f"\n[{datetime.datetime.now().strftime('%Y-%m-%d')}] {notes}"
                self.save()
                return True
        return False

    def add_interview(self, app_id, interview_type, interview_date, notes=""):
        """Add an interview record."""
        for app in self.applications:
            if app["id"] == app_id:
                app["interviews"].append({
                    "type" : interview_type,
                    "date" : interview_date,
                    "notes": notes
                })
                app["status"]       = "Interview"
                app["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d")
                self.save()
                return True
        return False

    def get_stats(self):
        """Get dashboard statistics."""
        total     = len(self.applications)
        if total == 0:
            return {}

        statuses  = {}
        for app in self.applications:
            s = app["status"]
            statuses[s] = statuses.get(s, 0) + 1

        avg_match = sum(
            app["analysis"]["match_score"]
            for app in self.applications
        ) / total

        top_matches = sorted(
            self.applications,
            key    = lambda a: a["analysis"]["match_score"],
            reverse= True
        )[:3]

        response_rate = (
            (statuses.get("Interview", 0) + statuses.get("Offer", 0)) /
            total * 100
        ) if total > 0 else 0

        return {
            "total_applications": total,
            "status_breakdown"  : statuses,
            "avg_match_score"   : round(avg_match, 1),
            "response_rate"     : round(response_rate, 1),
            "top_matches"       : [(a["company"], a["analysis"]["match_score"])
                                   for a in top_matches],
        }

    def export_csv(self, filepath="applications_export.csv"):
        """Export applications to CSV."""
        import csv
        if not self.applications:
            return

        fieldnames = ["id", "company", "role", "location", "salary_range",
                      "status", "applied_date", "match_score",
                      "required_skills", "missing_skills", "job_url"]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for app in self.applications:
                writer.writerow({
                    "id"             : app["id"],
                    "company"        : app["company"],
                    "role"           : app["role"],
                    "location"       : app["location"],
                    "salary_range"   : app["salary_range"],
                    "status"         : app["status"],
                    "applied_date"   : app["applied_date"],
                    "match_score"    : app["analysis"]["match_score"],
                    "required_skills": ", ".join(app["analysis"]["required_skills"]),
                    "missing_skills" : ", ".join(app["analysis"]["missing"]),
                    "job_url"        : app["job_url"],
                })
        print(f"  Exported to {filepath}")


# -- SECTION 2: Test the Tracker Logic -----------------------
print("\nSECTION 2: Testing the Tracker")
print("-" * 40)

tracker = JobApplicationTracker("test_applications.json")

your_skills = """
Python, NumPy, Pandas, scikit-learn, TensorFlow, Keras,
LangChain, OpenAI API, ChromaDB, RAG, Streamlit, FastAPI,
Git, Docker, Prompt Engineering, HuggingFace, Neural Networks,
SQL, Matplotlib, Data Analysis
"""

# Sample job applications
sample_jobs = [
    {
        "company"     : "Sarvam AI",
        "role"        : "AI Engineer",
        "location"    : "Bangalore",
        "salary_range": "10-15 LPA",
        "job_url"     : "https://sarvam.ai/careers",
        "job_desc"    : """
        We are looking for an AI Engineer with:
        - Strong Python and machine learning fundamentals
        - Experience with LangChain and RAG pipelines
        - Knowledge of OpenAI API and prompt engineering
        - Familiarity with ChromaDB or Pinecone vector databases
        - Streamlit or FastAPI for deployment
        - Git and Docker experience
        - Understanding of transformers and LLMs
        - HuggingFace experience preferred
        """,
        "notes"       : "Referral from LinkedIn connection"
    },
    {
        "company"     : "TCS iON",
        "role"        : "GenAI Developer",
        "location"    : "Hyderabad",
        "salary_range": "8-12 LPA",
        "job_url"     : "https://tcs.com/careers",
        "job_desc"    : """
        GenAI Developer role:
        - Python programming (mandatory)
        - LLM integration using OpenAI or similar APIs
        - Prompt engineering skills
        - FastAPI for building AI microservices
        - SQL for data management
        - Git version control
        - Understanding of NLP and deep learning
        """,
        "notes"       : "Applied through careers portal"
    },
    {
        "company"     : "Google DeepMind",
        "role"        : "Senior ML Research Engineer",
        "location"    : "Bangalore",
        "salary_range": "40-60 LPA",
        "job_url"     : "https://deepmind.com/careers",
        "job_desc"    : """
        Senior ML Research Engineer (5+ years experience):
        - PhD or Masters in ML/AI preferred
        - PyTorch and TensorFlow expertise
        - Published research in top conferences (NeurIPS, ICML)
        - Kubernetes and distributed training experience
        - Spark for large-scale data processing
        - Computer vision and object detection experience
        - Deep understanding of transformers architecture
        """,
        "notes"       : "Stretch application -- learning experience"
    },
    {
        "company"     : "Freshworks",
        "role"        : "ML Engineer",
        "location"    : "Chennai",
        "salary_range": "12-18 LPA",
        "job_url"     : "https://freshworks.com/careers",
        "job_desc"    : """
        ML Engineer position:
        - Python with scikit-learn and TensorFlow
        - Pandas and NumPy for data processing
        - Machine learning model development
        - FastAPI for model serving
        - Docker containerization
        - SQL database knowledge
        - Git for version control
        - NLP experience is a plus
        """,
        "notes"       : "Good culture, applied cold"
    },
]

print("  Adding 4 sample applications...\n")
for job in sample_jobs:
    app = tracker.add_application(
        company      = job["company"],
        role         = job["role"],
        job_desc     = job["job_desc"],
        your_skills  = your_skills,
        location     = job["location"],
        salary_range = job["salary_range"],
        job_url      = job["job_url"],
        notes        = job["notes"]
    )
    analysis = app["analysis"]
    print(f"  {app['company']:<20} | {app['role']:<30} | Match: {analysis['match_score']}%")
    print(f"    Skills found: {len(analysis['you_have'])} | Missing: {len(analysis['missing'])}")
    print(f"    Recommendation: {analysis['recommendation']}")
    print()

# Update some statuses
tracker.update_status(1, "Interview", "HR screening call scheduled")
tracker.add_interview(1, "Technical Round 1", "2025-05-15", "DSA + ML concepts")
tracker.update_status(3, "Rejected", "Too senior, missing research experience")

# Show stats
stats = tracker.get_stats()
print("\n  Dashboard Stats:")
print(f"  Total applications : {stats['total_applications']}")
print(f"  Avg match score    : {stats['avg_match_score']}%")
print(f"  Response rate      : {stats['response_rate']}%")
print(f"  Status breakdown   : {stats['status_breakdown']}")
print(f"\n  Top matches:")
for company, score in stats["top_matches"]:
    print(f"    {company:<20}: {score}%")

tracker.export_csv()


# -- SECTION 3: Write the Streamlit App ----------------------
print("\n\nSECTION 3: Writing the Streamlit App")
print("-" * 40)

app_code = '''# -*- coding: utf-8 -*-
# ============================================================
# Job Application Tracker -- AI-Powered Portfolio Project
# Run: streamlit run job_tracker_app.py
# ============================================================

import streamlit as st
import json
import os
import datetime
import pandas as pd

st.set_page_config(
    page_title = "Job Application Tracker",
    page_icon  = "briefcase",
    layout     = "wide"
)

# -- Skills database ------------------------------------------
AI_SKILLS = [
    "python", "machine learning", "deep learning", "nlp",
    "langchain", "rag", "openai", "huggingface", "tensorflow",
    "pytorch", "scikit-learn", "numpy", "pandas", "fastapi",
    "streamlit", "docker", "git", "sql", "transformers",
    "llm", "vector database", "chromadb", "prompt engineering",
    "fine-tuning", "mlops", "aws", "agents", "embeddings",
]

DATA_FILE = "my_job_applications.json"

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def analyze_job(job_desc, your_skills):
    job_lower  = job_desc.lower()
    your_lower = your_skills.lower()
    required   = [s for s in AI_SKILLS if s in job_lower]
    you_have   = [s for s in required if s in your_lower]
    missing    = [s for s in required if s not in your_lower]
    score      = round(len(you_have) / len(required) * 100, 1) if required else 0

    if score >= 80:
        rec = "Apply immediately -- strong match!"
    elif score >= 60:
        rec = "Good match -- apply with tailored resume"
    elif score >= 40:
        rec = "Decent match -- highlight transferable skills"
    else:
        rec = "Weak match -- learn missing skills first"

    return {"required": required, "you_have": you_have,
            "missing": missing, "score": score, "recommendation": rec}

# -- Initialize session state ---------------------------------
if "applications" not in st.session_state:
    st.session_state.applications = load_data()

apps = st.session_state.applications

# -- Header ---------------------------------------------------
st.title("Job Application Tracker")
st.caption("AI-powered job search manager -- built by Sharvaani K")

# -- Sidebar --------------------------------------------------
with st.sidebar:
    st.header("Your Skills")
    your_skills = st.text_area(
        "Paste your skills here (used for match scoring)",
        value  = "Python, NumPy, Pandas, scikit-learn, TensorFlow, LangChain, OpenAI API, ChromaDB, RAG, Streamlit, FastAPI, Git, Docker, Prompt Engineering, HuggingFace, SQL",
        height = 150,
        help   = "Update this with your actual skills"
    )
    st.divider()
    st.markdown(f"**Total applications:** {len(apps)}")
    if apps:
        statuses = {}
        for a in apps:
            statuses[a["status"]] = statuses.get(a["status"], 0) + 1
        for status, count in statuses.items():
            st.markdown(f"- {status}: **{count}**")

# -- Tabs -----------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Add Application",
    "My Applications",
    "Dashboard",
    "Skills Gap"
])

# -- TAB 1: Add Application ----------------------------------
with tab1:
    st.subheader("Add New Job Application")

    col1, col2 = st.columns(2)
    with col1:
        company      = st.text_input("Company Name *", placeholder="Sarvam AI")
        role         = st.text_input("Job Role *",     placeholder="AI Engineer")
        location     = st.text_input("Location",       placeholder="Bangalore")
        salary_range = st.text_input("Salary Range",   placeholder="10-15 LPA")
    with col2:
        job_url      = st.text_input("Job URL",        placeholder="https://...")
        notes        = st.text_area("Notes",           placeholder="How you found it, referral, etc.", height=100)

    job_desc = st.text_area(
        "Job Description *",
        placeholder = "Paste the full job description here...",
        height      = 200
    )

    if st.button("Analyze and Add Application", type="primary", use_container_width=True):
        if not company or not role or not job_desc:
            st.error("Please fill in Company, Role, and Job Description!")
        else:
            analysis = analyze_job(job_desc, your_skills)
            new_app  = {
                "id"           : len(apps) + 1,
                "company"      : company,
                "role"         : role,
                "location"     : location,
                "salary_range" : salary_range,
                "job_url"      : job_url,
                "job_desc"     : job_desc,
                "notes"        : notes,
                "status"       : "Applied",
                "applied_date" : datetime.datetime.now().strftime("%Y-%m-%d"),
                "analysis"     : analysis
            }
            apps.append(new_app)
            save_data(apps)
            st.session_state.applications = apps

            st.success(f"Added application for {role} at {company}!")

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Match Score",    f"{analysis['score']}%")
            col_b.metric("Skills You Have", len(analysis['you_have']))
            col_c.metric("Skills Missing",  len(analysis['missing']))

            if analysis["score"] >= 70:
                st.success(f"Recommendation: {analysis['recommendation']}")
            elif analysis["score"] >= 40:
                st.warning(f"Recommendation: {analysis['recommendation']}")
            else:
                st.error(f"Recommendation: {analysis['recommendation']}")

            if analysis["missing"]:
                st.markdown("**Skills to learn before applying:**")
                cols = st.columns(4)
                for i, skill in enumerate(analysis["missing"]):
                    cols[i % 4].error(skill)

# -- TAB 2: My Applications ----------------------------------
with tab2:
    st.subheader(f"My Applications ({len(apps)} total)")

    if not apps:
        st.info("No applications yet! Add your first one in the Add Application tab.")
    else:
        status_filter = st.selectbox(
            "Filter by status",
            ["All", "Applied", "Interview", "Offer", "Rejected"]
        )

        filtered = apps if status_filter == "All" else [
            a for a in apps if a["status"] == status_filter
        ]

        for app in reversed(filtered):
            score  = app["analysis"]["score"]
            color  = "green" if score >= 70 else "orange" if score >= 40 else "red"
            status = app["status"]

            with st.expander(f"{app['company']} -- {app['role']} | {status} | Match: {score}%"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Match Score",  f"{score}%")
                col2.metric("Applied",      app["applied_date"])
                col3.metric("Location",     app.get("location", "N/A"))
                col4.metric("Salary",       app.get("salary_range", "N/A"))

                # Status update
                new_status = st.selectbox(
                    "Update status",
                    ["Applied", "Interview", "Offer", "Rejected"],
                    index=["Applied", "Interview", "Offer", "Rejected"].index(status),
                    key=f"status_{app['id']}"
                )
                if new_status != status:
                    for a in apps:
                        if a["id"] == app["id"]:
                            a["status"] = new_status
                    save_data(apps)
                    st.success("Status updated!")
                    st.rerun()

                if app.get("job_url"):
                    st.markdown(f"[View Job Posting]({app['job_url']})")

                if app.get("notes"):
                    st.markdown(f"**Notes:** {app['notes']}")

                skills_col1, skills_col2 = st.columns(2)
                with skills_col1:
                    st.markdown("**Skills you have:**")
                    for s in app["analysis"]["you_have"]:
                        st.success(s, icon=None)
                with skills_col2:
                    st.markdown("**Skills to learn:**")
                    for s in app["analysis"]["missing"]:
                        st.error(s, icon=None)

# -- TAB 3: Dashboard ----------------------------------------
with tab3:
    st.subheader("Application Dashboard")

    if not apps:
        st.info("Add applications to see your dashboard!")
    else:
        col1, col2, col3, col4 = st.columns(4)
        statuses     = {}
        match_scores = []
        for a in apps:
            statuses[a["status"]] = statuses.get(a["status"], 0) + 1
            match_scores.append(a["analysis"]["score"])

        col1.metric("Total Applied",   len(apps))
        col2.metric("Interviews",      statuses.get("Interview", 0))
        col3.metric("Offers",          statuses.get("Offer", 0))
        col4.metric("Avg Match Score", f"{sum(match_scores)/len(match_scores):.1f}%")

        st.divider()

        # Status chart
        if statuses:
            st.subheader("Application Status")
            status_df = pd.DataFrame(
                list(statuses.items()),
                columns=["Status", "Count"]
            )
            st.bar_chart(status_df.set_index("Status"))

        # Match score distribution
        st.subheader("Match Score Distribution")
        score_df = pd.DataFrame({
            "Company": [a["company"] for a in apps],
            "Match Score": [a["analysis"]["score"] for a in apps]
        })
        st.bar_chart(score_df.set_index("Company"))

# -- TAB 4: Skills Gap ----------------------------------------
with tab4:
    st.subheader("Skills Gap Analysis")

    if not apps:
        st.info("Add applications to see your skills gap!")
    else:
        # Aggregate missing skills across all applications
        all_missing = {}
        for app in apps:
            for skill in app["analysis"]["missing"]:
                all_missing[skill] = all_missing.get(skill, 0) + 1

        if all_missing:
            sorted_missing = sorted(all_missing.items(), key=lambda x: x[1], reverse=True)

            st.markdown("**Skills you are missing most often across all applications:**")
            st.markdown("(Learn these first for maximum impact on your job search)")
            st.divider()

            for skill, count in sorted_missing[:10]:
                col1, col2 = st.columns([3, 1])
                col1.progress(count / len(apps), text=skill.title())
                col2.markdown(f"Missing in **{count}** jobs")

        # Export button
        st.divider()
        if st.button("Export to CSV"):
            df = pd.DataFrame([{
                "Company"     : a["company"],
                "Role"        : a["role"],
                "Status"      : a["status"],
                "Match Score" : a["analysis"]["score"],
                "Applied Date": a["applied_date"],
                "Missing Skills": ", ".join(a["analysis"]["missing"])
            } for a in apps])
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "job_applications.csv",
                "text/csv"
            )

st.divider()
st.caption("Built by Sharvaani K | Day 8 of 30-Day AI Engineering Journey")
st.caption("GitHub: https://github.com/sharvaani213-hub/ai-30day-journey")
'''

with open("job_tracker_app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

print("  job_tracker_app.py written successfully!")
print("  Run with: streamlit run job_tracker_app.py")

print()
print("=" * 60)
print("Script 3 complete! Job Application Tracker built.")
print("Key features:")
print("  [OK] Add and track job applications")
print("  [OK] AI skill extraction from job descriptions")
print("  [OK] Match score calculation")
print("  [OK] Status tracking (Applied/Interview/Offer/Rejected)")
print("  [OK] Dashboard with charts")
print("  [OK] Skills gap analysis across all applications")
print("  [OK] CSV export")
print()
print("FILES CREATED:")
print("  [OK] script1_advanced_rag.py")
print("  [OK] script2_llm_evaluation.py")
print("  [OK] script3_job_tracker.py (this file)")
print("  [OK] job_tracker_app.py      <- run this with Streamlit!")
print("  [OK] advanced_rag_config.json")
print("  [OK] rag_evaluation_results.json")
print("=" * 60)
