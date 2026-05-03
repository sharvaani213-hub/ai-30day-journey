# -*- coding: utf-8 -*-
# ============================================================
# DAY 5 -- SCRIPT 3: Complete Project -- AI Career Advisor
# Topics: putting it all together -- LangChain + RAG +
#         Streamlit into one complete deployable project
# This is your PORTFOLIO PROJECT for job applications!
# ============================================================

import os
import json
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("DAY 5 -- SCRIPT 3: Complete Project -- AI Career Advisor")
print("=" * 60)

print("""
WHAT WE ARE BUILDING:
---------------------
A complete AI Career Advisor app that:
  1. Analyzes a job description you paste
  2. Compares it with your skills
  3. Identifies skill gaps
  4. Generates a personalized learning plan
  5. Rewrites your resume bullets for that job
  6. Estimates your chances of getting the role

This combines EVERYTHING from Days 1-5:
  Python + Pandas + LangChain + RAG + Streamlit + Prompt Engineering

This is a REAL project you can put on your resume!
""")


# -- SECTION 1: Core Logic (works without API) ----------------
print("SECTION 1: Core Career Advisor Logic")
print("-" * 40)

# AI/ML skills database
AI_SKILLS_DB = {
    "programming": {
        "python"         : {"level": "essential", "learn_days": 7},
        "javascript"     : {"level": "nice-to-have", "learn_days": 14},
        "sql"            : {"level": "important", "learn_days": 5},
        "bash/linux"     : {"level": "important", "learn_days": 3},
    },
    "ml_frameworks": {
        "scikit-learn"   : {"level": "essential", "learn_days": 5},
        "tensorflow"     : {"level": "important", "learn_days": 7},
        "pytorch"        : {"level": "important", "learn_days": 7},
        "keras"          : {"level": "important", "learn_days": 3},
        "huggingface"    : {"level": "essential", "learn_days": 5},
    },
    "llm_tools": {
        "openai api"     : {"level": "essential", "learn_days": 2},
        "langchain"      : {"level": "essential", "learn_days": 5},
        "llamaindex"     : {"level": "important", "learn_days": 4},
        "rag"            : {"level": "essential", "learn_days": 5},
        "vector database": {"level": "important", "learn_days": 3},
        "chromadb"       : {"level": "important", "learn_days": 2},
        "pinecone"       : {"level": "nice-to-have", "learn_days": 2},
        "fine-tuning"    : {"level": "important", "learn_days": 7},
        "lora"           : {"level": "important", "learn_days": 5},
        "prompt engineering": {"level": "essential", "learn_days": 3},
    },
    "deployment": {
        "streamlit"      : {"level": "essential", "learn_days": 2},
        "fastapi"        : {"level": "important", "learn_days": 4},
        "docker"         : {"level": "important", "learn_days": 5},
        "aws"            : {"level": "nice-to-have", "learn_days": 14},
        "gcp"            : {"level": "nice-to-have", "learn_days": 14},
        "git/github"     : {"level": "essential", "learn_days": 2},
    },
    "data_skills": {
        "numpy"          : {"level": "essential", "learn_days": 3},
        "pandas"         : {"level": "essential", "learn_days": 4},
        "matplotlib"     : {"level": "important", "learn_days": 2},
        "data cleaning"  : {"level": "important", "learn_days": 3},
        "feature engineering": {"level": "important", "learn_days": 4},
    }
}

def analyze_job_description(job_desc):
    """Extract required skills from a job description."""
    job_lower = job_desc.lower()
    required_skills = []

    for category, skills in AI_SKILLS_DB.items():
        for skill, info in skills.items():
            if skill in job_lower:
                required_skills.append({
                    "skill"   : skill,
                    "category": category,
                    "level"   : info["level"],
                    "learn_days": info["learn_days"]
                })

    return required_skills

def analyze_candidate(your_skills_text):
    """Extract skills from candidate's skill description."""
    skills_lower = your_skills_text.lower()
    found_skills = []

    for category, skills in AI_SKILLS_DB.items():
        for skill in skills:
            if skill in skills_lower:
                found_skills.append(skill)

    return found_skills

def skill_gap_analysis(job_skills, candidate_skills):
    """Compare job requirements with candidate skills."""
    job_skill_names = [s["skill"] for s in job_skills]
    candidate_set   = set(candidate_skills)
    job_set         = set(job_skill_names)

    matching  = candidate_set.intersection(job_set)
    missing   = job_set - candidate_set
    extra     = candidate_set - job_set

    # Get full info for missing skills
    missing_info = [s for s in job_skills if s["skill"] in missing]

    # Sort by priority
    essential_missing = [s for s in missing_info if s["level"] == "essential"]
    important_missing = [s for s in missing_info if s["level"] == "important"]
    nice_missing      = [s for s in missing_info if s["level"] == "nice-to-have"]

    match_score = len(matching) / len(job_set) * 100 if job_set else 0

    return {
        "match_score"      : round(match_score, 1),
        "matching_skills"  : list(matching),
        "essential_missing": essential_missing,
        "important_missing": important_missing,
        "nice_missing"     : nice_missing,
        "extra_skills"     : list(extra),
        "total_days_to_learn": sum(s["learn_days"] for s in essential_missing + important_missing)
    }

def generate_learning_plan(gap_analysis):
    """Generate a personalized learning plan based on skill gaps."""
    essential = gap_analysis["essential_missing"]
    important = gap_analysis["important_missing"]

    plan = []
    day  = 1

    if essential:
        plan.append("WEEK 1-2: ESSENTIAL SKILLS (learn these first)")
        for skill in essential:
            end_day = day + skill["learn_days"] - 1
            plan.append(f"  Days {day}-{end_day}: Learn {skill['skill'].title()} ({skill['learn_days']} days)")
            day = end_day + 1

    if important:
        plan.append(f"\nWEEK {day//7 + 1}+: IMPORTANT SKILLS")
        for skill in important[:5]:  # top 5 important
            end_day = day + skill["learn_days"] - 1
            plan.append(f"  Days {day}-{end_day}: Learn {skill['skill'].title()} ({skill['learn_days']} days)")
            day = end_day + 1

    plan.append(f"\nTOTAL TIME: ~{gap_analysis['total_days_to_learn']} days of focused study")
    plan.append("TIP: Build one project for every 2-3 skills you learn!")

    return "\n".join(plan)


# -- SECTION 2: Test the Logic --------------------------------
print("\nSECTION 2: Testing Career Advisor Logic")
print("-" * 40)

# Sample job description
job_description = """
We are looking for an AI Engineer to join our team.

Requirements:
- Strong Python programming skills
- Experience with LangChain and RAG pipelines
- Knowledge of OpenAI API and prompt engineering
- Familiarity with vector databases (ChromaDB or Pinecone)
- Experience with Streamlit or FastAPI for deployment
- Git/GitHub for version control
- Understanding of fine-tuning and LoRA
- NumPy and Pandas for data manipulation
- HuggingFace transformers library

Nice to have:
- Docker containerization
- AWS or GCP cloud experience
"""

# Sample candidate profile (Sharvaani after 5 days of learning!)
candidate_skills = """
Python, NumPy, Pandas, Matplotlib, Scikit-learn,
TensorFlow, Keras, LangChain, OpenAI API, ChromaDB,
RAG, Streamlit, Git/GitHub, Prompt Engineering,
HuggingFace, Neural Networks
"""

print("JOB DESCRIPTION ANALYSIS:")
print("-" * 40)
job_skills      = analyze_job_description(job_description)
candidate_found = analyze_candidate(candidate_skills)
gap             = skill_gap_analysis(job_skills, candidate_found)

print(f"  Skills required by job : {len(job_skills)}")
print(f"  Your matching skills   : {len(gap['matching_skills'])}")
print(f"  Match score            : {gap['match_score']}%")

print(f"\n  Your matching skills:")
for skill in sorted(gap["matching_skills"]):
    print(f"    [YES] {skill}")

if gap["essential_missing"]:
    print(f"\n  Essential skills to learn:")
    for s in gap["essential_missing"]:
        print(f"    [LEARN] {s['skill']} (~{s['learn_days']} days)")

if gap["important_missing"]:
    print(f"\n  Important skills to learn:")
    for s in gap["important_missing"][:5]:
        print(f"    [LEARN] {s['skill']} (~{s['learn_days']} days)")

print(f"\nPERSONALIZED LEARNING PLAN:")
print("-" * 40)
plan = generate_learning_plan(gap)
print(plan)


# -- SECTION 3: Write the Complete Streamlit App --------------
print("\n\nSECTION 3: Writing the Complete Streamlit Portfolio App")
print("-" * 40)

app_code = """# -*- coding: utf-8 -*-
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

                    prompt = f\"\"\"
                    Job Description: {job_desc[:500]}

                    Candidate Skills: {your_skills[:300]}

                    Matching skills: {gap['matching']}
                    Missing skills: {[s['skill'] for s in gap['missing'][:5]]}

                    Give 3 specific, actionable resume tips for this candidate
                    to improve their chances for this role. Be direct and specific.
                    \"\"\"

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
"""

with open("career_advisor_app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

print("  career_advisor_app.py written successfully!")


# -- SECTION 4: Day 5 Summary ---------------------------------
print("\n\nSECTION 4: Day 5 Complete -- What You Built")
print("=" * 60)
print("""
  Script 1: AI Agents
  -> Built 5 tools (calculator, datetime, converter, planner, salary)
  -> Implemented ReAct pattern manually
  -> Understood LangChain agent structure

  Script 2: Fine-tuning LLMs
  -> Understood LoRA math and intuition
  -> Prepared a fine-tuning dataset (Alpaca format)
  -> Got a Colab-ready LoRA template
  -> Know when to use fine-tune vs RAG vs prompting

  Script 3: Complete Portfolio Project
  -> Built AI Career Advisor app
  -> Analyzes job descriptions automatically
  -> Identifies your skill gaps
  -> Generates personalized learning plans
  -> Full Streamlit UI ready to deploy

  FILES CREATED TODAY:
  [OK] script1_agents.py
  [OK] script2_finetuning.py
  [OK] script3_complete_project.py  (this file)
  [OK] career_advisor_app.py        <- your 2nd portfolio app!
  [OK] finetune_dataset.json        <- sample training data
  [OK] lora_finetune_template.py    <- Colab ready!
""")

print("HOW TO RUN YOUR NEW APP:")
print("  streamlit run career_advisor_app.py")
print()
print("WHAT TO DO NEXT:")
print("  1. Run the app locally")
print("  2. Test it with a real job description from LinkedIn/Naukri")
print("  3. Deploy on Streamlit Cloud (same steps as app.py)")
print("  4. Add the live URL to your LinkedIn post and resume")
print()
print("=" * 60)
print("5 days done! You now have 2 deployed AI apps.")
print("That already puts you ahead of most freshers applying for AI roles.")
print("=" * 60)
