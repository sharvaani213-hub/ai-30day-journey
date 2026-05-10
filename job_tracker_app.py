# -*- coding: utf-8 -*-
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
