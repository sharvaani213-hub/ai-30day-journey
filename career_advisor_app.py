# -*- coding: utf-8 -*-
# ============================================================
# AI Career Advisor -- Complete Portfolio Project
# Run: streamlit run career_advisor_app.py
# ============================================================

import streamlit as st
import os

st.set_page_config(
    page_title = "AI Career Advisor",
    page_icon  = "briefcase",
    layout     = "wide"
)

# -- Skill database -------------------------------------------
AI_SKILLS_DB = {
    "programming": {
        "python": {"level": "essential", "learn_days": 7},
        "sql": {"level": "important", "learn_days": 5},
        "bash/linux": {"level": "important", "learn_days": 3},
    },
    "ml_frameworks": {
        "scikit-learn": {"level": "essential", "learn_days": 5},
        "tensorflow": {"level": "important", "learn_days": 7},
        "pytorch": {"level": "important", "learn_days": 7},
        "huggingface": {"level": "essential", "learn_days": 5},
    },
    "llm_tools": {
        "openai api": {"level": "essential", "learn_days": 2},
        "langchain": {"level": "essential", "learn_days": 5},
        "rag": {"level": "essential", "learn_days": 5},
        "vector database": {"level": "important", "learn_days": 3},
        "chromadb": {"level": "important", "learn_days": 2},
        "fine-tuning": {"level": "important", "learn_days": 7},
        "prompt engineering": {"level": "essential", "learn_days": 3},
    },
    "deployment": {
        "streamlit": {"level": "essential", "learn_days": 2},
        "fastapi": {"level": "important", "learn_days": 4},
        "docker": {"level": "important", "learn_days": 5},
        "git/github": {"level": "essential", "learn_days": 2},
    },
    "data_skills": {
        "numpy": {"level": "essential", "learn_days": 3},
        "pandas": {"level": "essential", "learn_days": 4},
        "matplotlib": {"level": "important", "learn_days": 2},
    }
}

def analyze_job(job_desc):
    job_lower = job_desc.lower()
    found = []
    for category, skills in AI_SKILLS_DB.items():
        for skill, info in skills.items():
            if skill in job_lower:
                found.append({"skill": skill, "category": category, **info})
    return found

def analyze_candidate(skills_text):
    skills_lower = skills_text.lower()
    found = []
    for category, skills in AI_SKILLS_DB.items():
        for skill in skills:
            if skill in skills_lower:
                found.append(skill)
    return found

def gap_analysis(job_skills, candidate_skills):
    job_names     = [s["skill"] for s in job_skills]
    candidate_set = set(candidate_skills)
    job_set       = set(job_names)
    matching      = candidate_set.intersection(job_set)
    missing_names = job_set - candidate_set
    missing_info  = [s for s in job_skills if s["skill"] in missing_names]
    score         = round(len(matching) / len(job_set) * 100, 1) if job_set else 0
    return {
        "score"   : score,
        "matching": list(matching),
        "missing" : missing_info,
        "days"    : sum(s["learn_days"] for s in missing_info)
    }

# -- Header ---------------------------------------------------
st.title("AI Career Advisor")
st.caption("Paste a job description and your skills -- get a personalized action plan")

st.divider()

# -- Input columns --------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Job Description")
    job_desc = st.text_area(
        "Paste the job description here",
        height      = 300,
        placeholder = "We are looking for an AI Engineer with Python, LangChain, RAG..."
    )

with col2:
    st.subheader("Your Current Skills")
    your_skills = st.text_area(
        "List your current skills",
        height      = 300,
        placeholder = "Python, NumPy, Pandas, scikit-learn, TensorFlow, LangChain..."
    )

    api_key = st.text_input(
        "OpenAI API Key (optional -- for AI-powered analysis)",
        type        = "password",
        placeholder = "sk-... (leave blank for basic analysis)"
    )

# -- Analyze button -------------------------------------------
if st.button("Analyze My Profile", type="primary", use_container_width=True):

    if not job_desc or not your_skills:
        st.error("Please fill in both the job description and your skills!")
    else:
        with st.spinner("Analyzing your profile..."):

            job_skills      = analyze_job(job_desc)
            candidate_found = analyze_candidate(your_skills)
            gap             = gap_analysis(job_skills, candidate_found)

        st.divider()
        st.subheader("Your Analysis Results")

        # -- Score ------------------------------------------------
        score = gap["score"]
        col_score, col_match, col_missing, col_days = st.columns(4)

        color = "green" if score >= 70 else "orange" if score >= 40 else "red"
        col_score.metric("Match Score", f"{score}%")
        col_match.metric("Skills You Have", len(gap["matching"]))
        col_missing.metric("Skills to Learn", len(gap["missing"]))
        col_days.metric("Days to Get Ready", gap["days"])

        # Progress bar
        st.progress(int(score) / 100)

        if score >= 70:
            st.success("Great match! You are well qualified for this role. Apply now!")
        elif score >= 40:
            st.warning(f"Decent match. Learn {len(gap['missing'])} more skills to be fully ready.")
        else:
            st.error(f"Skill gap detected. Focus on learning the essentials first.")

        st.divider()

        # -- Matching skills ----------------------------------------
        col_have, col_learn = st.columns(2)

        with col_have:
            st.subheader("Skills You Already Have")
            if gap["matching"]:
                for skill in sorted(gap["matching"]):
                    st.success(f"  {skill.title()}")
            else:
                st.write("No matching skills detected. Make sure to list your skills clearly.")

        with col_learn:
            st.subheader("Skills to Learn")
            if gap["missing"]:
                essential = [s for s in gap["missing"] if s["level"] == "essential"]
                important = [s for s in gap["missing"] if s["level"] == "important"]

                if essential:
                    st.markdown("**Essential (learn first):**")
                    for s in essential:
                        st.error(f"  {s['skill'].title()} -- {s['learn_days']} days")

                if important:
                    st.markdown("**Important:**")
                    for s in important:
                        st.warning(f"  {s['skill'].title()} -- {s['learn_days']} days")
            else:
                st.write("You have all required skills!")

        st.divider()

        # -- Learning Plan ------------------------------------------
        st.subheader("Your Personalized Learning Plan")
        if gap["missing"]:
            day  = 1
            data = []
            for s in gap["missing"]:
                end_day = day + s["learn_days"] - 1
                data.append({
                    "Priority"   : s["level"].title(),
                    "Skill"      : s["skill"].title(),
                    "Start Day"  : day,
                    "End Day"    : end_day,
                    "Days Needed": s["learn_days"]
                })
                day = end_day + 1

            import pandas as pd
            st.dataframe(pd.DataFrame(data), use_container_width=True)
            st.info(f"Total time to be job-ready: {gap['days']} days of focused study")
        else:
            st.success("You are already job-ready! Start applying today.")

        # -- AI Analysis (if API key) -------------------------------
        if api_key:
            st.divider()
            st.subheader("AI-Powered Resume Advice")
            with st.spinner("Getting AI advice..."):
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)

                    prompt = f"""
                    Job Description: {job_desc[:500]}

                    Candidate Skills: {your_skills[:300]}

                    Matching skills: {gap['matching']}
                    Missing skills: {[s['skill'] for s in gap['missing'][:5]]}

                    Give 3 specific, actionable resume tips for this candidate
                    to improve their chances for this role. Be direct and specific.
                    """

                    response = client.chat.completions.create(
                        model    = "gpt-3.5-turbo",
                        messages = [
                            {"role": "system", "content": "You are an expert AI career coach for engineering freshers in India."},
                            {"role": "user",   "content": prompt}
                        ],
                        max_tokens  = 400,
                        temperature = 0.7
                    )
                    st.write(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"API error: {e}")

# -- Footer ---------------------------------------------------
st.divider()
st.caption("Built by Sharvaani K | Day 5 of 30-day AI Engineering Journey")
st.caption("GitHub: https://github.com/sharvaani213-hub/ai-30day-journey")
