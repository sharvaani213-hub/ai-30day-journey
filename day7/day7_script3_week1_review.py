# -*- coding: utf-8 -*-
# ============================================================
# DAY 7 -- SCRIPT 3: Week 1 Review + Portfolio Polish
# Topics: Week 1 summary, GitHub polish, LinkedIn strategy,
#         interview prep, what companies actually look for,
#         week 2 preview
# ============================================================

import os
import json
import datetime
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("DAY 7 -- SCRIPT 3: Week 1 Complete -- Portfolio Polish")
print("=" * 60)


# -- SECTION 1: Week 1 Complete Summary ----------------------
print("\nSECTION 1: Everything You Learned in Week 1")
print("-" * 40)

week1_summary = {
    "Day 1": {
        "topics": ["Python basics", "OOP", "File I/O", "NumPy", "Pandas", "Matplotlib"],
        "scripts": 4,
        "key_skill": "Data manipulation and visualization",
        "project": "Student data analysis dashboard"
    },
    "Day 2": {
        "topics": ["NumPy deep dive", "Linear Regression", "Classification", "scikit-learn"],
        "scripts": 3,
        "key_skill": "Building and evaluating ML models",
        "project": "Salary predictor + Placement classifier"
    },
    "Day 3": {
        "topics": ["Neural Networks", "TensorFlow", "Transformers", "HuggingFace", "OpenAI API"],
        "scripts": 3,
        "key_skill": "Deep learning and LLM APIs",
        "project": "MNIST digit recognizer + CLI chatbot"
    },
    "Day 4": {
        "topics": ["LangChain", "RAG Pipeline", "ChromaDB", "Streamlit"],
        "scripts": 3,
        "key_skill": "Building LLM applications",
        "project": "AI Study Buddy chatbot (deployed!)"
    },
    "Day 5": {
        "topics": ["AI Agents", "ReAct pattern", "Fine-tuning", "LoRA", "PEFT"],
        "scripts": 3,
        "key_skill": "Agents and model customization",
        "project": "AI Career Advisor (deployed!)"
    },
    "Day 6": {
        "topics": ["MLOps", "Experiment tracking", "MLflow", "Model versioning", "Data drift"],
        "scripts": 3,
        "key_skill": "Production ML best practices",
        "project": "Experiment tracker + Model registry"
    },
    "Day 7": {
        "topics": ["FastAPI", "REST APIs", "Docker", "Containerization", "Deployment"],
        "scripts": 3,
        "key_skill": "Production deployment",
        "project": "ML serving API + Dockerized app"
    },
}

total_scripts = sum(d["scripts"] for d in week1_summary.values())
print(f"\n  {'Day':<8} {'Key Skill':<35} {'Project'}")
print(f"  {'─'*8} {'─'*35} {'─'*35}")
for day, info in week1_summary.items():
    print(f"  {day:<8} {info['key_skill']:<35} {info['project']}")

print(f"\n  Total scripts written : {total_scripts}")
print(f"  Topics covered        : {sum(len(d['topics']) for d in week1_summary.values())}")
print(f"  Deployed apps         : 2 (AI Study Buddy + Career Advisor)")
print(f"  GitHub commits        : 7+")


# -- SECTION 2: Skills You Can Now Put on Your Resume --------
print("\n\nSECTION 2: Your Resume Skills After Week 1")
print("-" * 40)

resume_skills = {
    "Programming Languages": ["Python (Advanced)", "SQL (Basic)"],
    "ML/DL Frameworks"     : ["scikit-learn", "TensorFlow", "Keras", "NumPy", "Pandas"],
    "LLM & GenAI"          : ["OpenAI API", "LangChain", "HuggingFace Transformers",
                              "RAG Pipelines", "Prompt Engineering", "ChromaDB"],
    "Web & APIs"           : ["Streamlit", "FastAPI", "REST APIs"],
    "MLOps & DevOps"       : ["MLflow", "Docker", "Git/GitHub", "Experiment Tracking"],
    "AI Concepts"          : ["Neural Networks", "Transformers", "Fine-tuning", "LoRA",
                              "AI Agents", "ReAct Pattern", "Vector Databases"],
}

print("\n  Skills section for your resume:\n")
for category, skills in resume_skills.items():
    skills_str = " | ".join(skills)
    print(f"  {category}")
    print(f"    {skills_str}")
    print()


# -- SECTION 3: GitHub Profile Polish ------------------------
print("\nSECTION 3: Polish Your GitHub Profile")
print("-" * 40)
print("""
Your GitHub is your coding resume. Here is exactly how to polish it.

1. PROFILE README (most important!)
-------------------------------------
Create a file: README.md in a repo named exactly as your username.
Example: if username is sharvaani213-hub,
create repo: sharvaani213-hub with README.md

Template:
---------
# Hi, I am Sharvaani K!

BE Computer Engineering | AI/ML Engineer in the making

## What I am Building
- AI-powered web apps with LangChain, RAG, and Streamlit
- Production ML APIs with FastAPI and Docker
- 30-day AI learning sprint -> landing an AI Engineer role

## Tech Stack
Python | LangChain | OpenAI API | HuggingFace | Streamlit
FastAPI | Docker | scikit-learn | TensorFlow | ChromaDB

## Live Projects
- AI Study Buddy Chatbot: [your streamlit link]
- AI Career Advisor    : [your streamlit link]
- ML Serving API       : [your render.com link]

## Currently Learning
Week 2: Advanced LLM techniques, n8n automation, real project builds

## Connect
LinkedIn: [your linkedin]
Location: Hyderabad, India
Open to  : AI Engineer, ML Engineer, Prompt Engineer roles

2. PIN YOUR BEST REPOS
------------------------
Go to your profile -> click "Customize your pins"
Pin these repos in this order:
  1. ai-30day-journey       (your main learning repo)
  2. career-advisor-app     (deployed project)
  3. ai-study-buddy         (deployed project)

3. REPO README QUALITY
------------------------
Every repo needs:
  - What the project does (2 sentences)
  - Screenshot or demo GIF
  - Tech stack badges
  - How to run locally
  - Live demo link

4. CONTRIBUTION GRAPH
----------------------
The green squares on your profile.
Commit every single day -- even if it is just a small update.
Recruiters look at this to see consistency.

5. STAR RELEVANT REPOS
-----------------------
Star repos for: LangChain, HuggingFace, FastAPI, Streamlit
It shows you follow the ecosystem.
""")


# -- SECTION 4: Interview Prep -- Questions You Will Get -----
print("\nSECTION 4: Interview Questions You Will Definitely Get")
print("-" * 40)

interview_qa = [
    {
        "q": "Explain RAG in simple terms",
        "a": "RAG = Retrieval Augmented Generation. Instead of relying only on the LLM's training knowledge, we first search a vector database for relevant document chunks, then pass those chunks along with the user's question to the LLM. This way the LLM answers based on YOUR specific data, reducing hallucinations and allowing up-to-date information."
    },
    {
        "q": "What is the difference between fine-tuning and RAG?",
        "a": "RAG retrieves external documents at query time -- good for frequently changing data and when you need citations. Fine-tuning bakes knowledge into the model weights -- good for consistent style/tone changes and when data is stable. For most enterprise apps, RAG is cheaper and faster to set up."
    },
    {
        "q": "What is an embedding?",
        "a": "An embedding is a dense vector of numbers that represents the semantic meaning of text. Similar meanings have similar vectors (measurable by cosine similarity). Used in RAG to find relevant documents by comparing question vectors with document vectors."
    },
    {
        "q": "Explain the transformer attention mechanism",
        "a": "Attention computes a weighted sum of value vectors, where weights come from the similarity between query and key vectors (Q, K, V). This lets each token attend to all other tokens simultaneously, unlike RNNs which process sequentially. Multi-head attention runs this in parallel across multiple subspaces."
    },
    {
        "q": "What is overfitting and how do you prevent it?",
        "a": "Overfitting is when a model memorizes training data instead of learning general patterns, resulting in high training accuracy but low test accuracy. Prevention: dropout layers, L1/L2 regularization, more training data, early stopping, cross-validation, and reducing model complexity."
    },
    {
        "q": "What is a vector database and when do you use it?",
        "a": "A vector database stores high-dimensional embeddings and enables fast similarity search (cosine or dot product). Used in RAG systems, semantic search, and recommendation systems. Examples: ChromaDB (local, free), Pinecone (cloud), FAISS (Facebook, fastest). Use when you need to find similar content at scale."
    },
    {
        "q": "How would you deploy an ML model to production?",
        "a": "1. Wrap model in FastAPI endpoint with Pydantic validation. 2. Write Dockerfile to containerize the app. 3. Push to GitHub. 4. Deploy to Render/Railway/AWS App Runner. 5. Set up monitoring for data drift and model performance degradation. 6. Implement versioning with MLflow for rollback capability."
    },
    {
        "q": "What is prompt engineering and give examples of techniques",
        "a": "Prompt engineering is designing effective LLM inputs. Key techniques: zero-shot (just ask), few-shot (give examples first), chain-of-thought (add 'think step by step'), role prompting (assign a persona in system prompt), and output anchoring (specify exact format). Good prompts dramatically improve output quality without any model training."
    },
    {
        "q": "Explain the difference between precision and recall",
        "a": "Precision = of all predicted positives, how many were actually positive (reduces false positives). Recall = of all actual positives, how many did we catch (reduces false negatives). F1 = harmonic mean of both. Use precision when false positives are costly (spam filter). Use recall when false negatives are costly (cancer detection)."
    },
    {
        "q": "What is LangChain and what problem does it solve?",
        "a": "LangChain is a framework for building LLM-powered applications. It solves the problem of manually managing prompts, API calls, memory, and output parsing. It provides: prompt templates, chains (connect LLM steps), memory (conversation history), agents (LLM that uses tools), and output parsers (structured responses). Think of it as React for AI apps."
    },
]

for i, qa in enumerate(interview_qa, 1):
    print(f"\n  Q{i}: {qa['q']}")
    print(f"  A : {qa['a'][:150]}...")

# Save full Q&A to file
with open("interview_prep.json", "w", encoding="utf-8") as f:
    json.dump(interview_qa, f, indent=2, ensure_ascii=False)
print(f"\n  Full Q&A saved to interview_prep.json ({len(interview_qa)} questions)")


# -- SECTION 5: Week 2 Preview --------------------------------
print("\n\nSECTION 5: Week 2 Preview -- What is Coming")
print("-" * 40)
print("""
Week 1 was: LEARNING the tools
Week 2 is : BUILDING real projects that get you hired

Day  8: Advanced RAG -- multi-doc, re-ranking, hybrid search
Day  9: LLM evaluation -- how to measure if your app is good
Day 10: n8n automation -- build 3 real workflows
Day 11: Real project -- Job Application Tracker (full stack)
Day 12: Real project -- AI Resume Analyzer (with PDF upload)
Day 13: Interview prep -- mock technical interview simulation
Day 14: Week 2 review + apply to 20 companies

By end of Week 2:
  -> 4 deployed projects on your portfolio
  -> 20 job applications sent
  -> Technical interview practice done
  -> LinkedIn with 200+ new connections in AI space
""")


# -- SECTION 6: Generate Week 1 Certificate ------------------
print("\n\nSECTION 6: Your Week 1 Achievement Summary")
print("-" * 40)

certificate = {
    "name"          : "Sharvaani K",
    "achievement"   : "Completed Week 1 of 30-Day AI Engineering Sprint",
    "date"          : datetime.datetime.now().strftime("%d %B %Y"),
    "skills_gained" : list(resume_skills.keys()),
    "projects_built": [
        "Student Data Analysis Dashboard",
        "Salary Predictor + Placement Classifier",
        "MNIST Neural Network (97%+ accuracy)",
        "AI Study Buddy Chatbot (deployed)",
        "AI Career Advisor App (deployed)",
        "ML Experiment Tracker",
        "FastAPI ML Serving API",
        "Dockerized Application"
    ],
    "github"        : "https://github.com/sharvaani213-hub/ai-30day-journey",
    "next_milestone": "Week 2 -- Build 4 production projects and apply to 20 companies"
}

with open("week1_summary.json", "w", encoding="utf-8") as f:
    json.dump(certificate, f, indent=2, ensure_ascii=False)

print(f"\n  Week 1 Achievement Summary")
print(f"  {'─'*40}")
print(f"  Name          : {certificate['name']}")
print(f"  Date          : {certificate['date']}")
print(f"  Scripts written: {total_scripts}")
print(f"  Projects built : {len(certificate['projects_built'])}")
print(f"  Apps deployed  : 2")
print(f"  GitHub commits : 7+")
print(f"\n  Projects completed:")
for p in certificate["projects_built"]:
    print(f"    [DONE] {p}")

print(f"\n  Next milestone: {certificate['next_milestone']}")

print()
print("=" * 60)
print("Script 3 complete! Week 1 Done!")
print()
print("  You just completed 7 days of intense AI learning.")
print("  Most people quit by Day 3. You are still here.")
print("  That is already the top 10 percent.")
print()
print("  Week 2 starts tomorrow. Get some rest tonight.")
print("  You have earned it.")
print("=" * 60)
