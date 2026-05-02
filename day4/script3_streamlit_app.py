# -*- coding: utf-8 -*-
# ============================================================
# DAY 4 -- SCRIPT 3: Streamlit -- Deploy Your AI App
# Topics: Streamlit UI, deploy RAG chatbot, file upload,
#         session state, chat interface, deploy to cloud
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("DAY 4 -- SCRIPT 3: Streamlit -- Deploy Your AI App")
print("=" * 60)

print("""
WHAT IS STREAMLIT?
------------------
Streamlit converts Python scripts into web apps instantly.
No HTML, CSS, or JavaScript needed.

Why it matters for AI engineers:
  -> Deploy your AI projects as live web apps in minutes
  -> Share with recruiters -- they can USE your project, not just read code
  -> Standard tool used by data scientists and AI engineers
  -> Free hosting at streamlit.io/cloud

Today we'll build a complete AI chatbot web app.
""")


# -- SECTION 1: Streamlit Basics (Print version) ---------------
print("SECTION 1: Streamlit Core Components")
print("-" * 40)
print("""
Key Streamlit components:

  st.title("Hello")          -> big heading
  st.write("text")           -> write any text or data
  st.text_input("label")     -> text input box
  st.button("Click me")      -> clickable button
  st.selectbox("Pick", list) -> dropdown
  st.slider("Val", 0, 100)   -> slider
  st.file_uploader("Upload") -> file upload
  st.chat_input("Message")   -> chat input box
  st.chat_message("user")    -> chat bubble
  st.spinner("Loading...")   -> loading animation
  st.success("Done!")        -> green success message
  st.error("Failed!")        -> red error message
  st.sidebar.write("text")   -> left sidebar

  Session State (VERY IMPORTANT):
  st.session_state["key"] = value  -> persists across reruns
""")


# -- SECTION 2: Write the Streamlit App to a File --------------
print("\nSECTION 2: Writing the Streamlit App")
print("-" * 40)

app_code = """# ============================================================
# DAY 4 -- AI Chatbot Web App with Streamlit
# Run with: streamlit run app.py
# ============================================================

import streamlit as st
import os

# Page config -- must be FIRST streamlit command
st.set_page_config(
    page_title = "AI Study Buddy",
    page_icon  = "robot",
    layout     = "wide"
)

# -- App Title & Description -----------------------------------
st.title("AI Study Buddy")
st.caption("Your personal AI tutor for Computer Engineering students")

# -- Sidebar ---------------------------------------------------
with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "OpenAI API Key",
        type        = "password",
        help        = "Get yours at platform.openai.com",
        placeholder = "sk-..."
    )

    model = st.selectbox(
        "Model",
        ["gpt-3.5-turbo", "gpt-4"],
        index = 0
    )

    temperature = st.slider(
        "Temperature (creativity)",
        min_value = 0.0,
        max_value = 2.0,
        value     = 0.7,
        step      = 0.1,
        help      = "0 = focused, 2 = very creative"
    )

    persona = st.selectbox(
        "AI Persona",
        [
            "Helpful tutor",
            "Strict professor",
            "Encouraging coach",
            "Senior AI engineer"
        ]
    )

    st.divider()

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**Built by:** Sharvaani K")
    st.markdown("**Day 4** of 30-day AI journey")
    st.markdown("[GitHub](https://github.com/sharvaani213-hub/ai-30day-journey)")

# -- System Prompt based on persona ---------------------------
persona_prompts = {
    "Helpful tutor"      : "You are a friendly and helpful AI tutor for computer engineering students. Explain concepts clearly with examples.",
    "Strict professor"   : "You are a strict professor. Give precise, technical answers. No hand-holding.",
    "Encouraging coach"  : "You are an encouraging career coach. Motivate the student while giving practical advice.",
    "Senior AI engineer" : "You are a senior AI engineer with 10 years experience. Give practical, real-world advice."
}

system_prompt = persona_prompts[persona]

# -- Initialize session state ---------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -- Display chat history -------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -- Suggested questions (show only at start) -----------------
if len(st.session_state.messages) == 0:
    st.markdown("### Try asking:")
    cols = st.columns(2)
    suggestions = [
        "Explain RAG pipelines simply",
        "What skills do I need for AI jobs?",
        "How does attention mechanism work?",
        "Give me a 30-day AI learning plan"
    ]
    for i, suggestion in enumerate(suggestions):
        col = cols[i % 2]
        if col.button(suggestion, key=f"suggest_{i}"):
            st.session_state.pending_question = suggestion
            st.rerun()

# -- Handle pending suggestion clicks ------------------------
if "pending_question" in st.session_state:
    user_input = st.session_state.pending_question
    del st.session_state.pending_question
else:
    user_input = None

# -- Chat Input -----------------------------------------------
chat_input = st.chat_input("Ask me anything about AI, ML, or your career...")
if chat_input:
    user_input = chat_input

# -- Process user input ---------------------------------------
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            if not api_key:
                response = \"\"\"Please enter your OpenAI API key in the sidebar to get AI responses.

**Get your free API key:**
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up / Log in
3. Go to API Keys -> Create new key
4. Paste it in the sidebar\"\"\"

            else:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)

                    messages = [{"role": "system", "content": system_prompt}]
                    for msg in st.session_state.messages:
                        messages.append({"role": msg["role"], "content": msg["content"]})

                    result = client.chat.completions.create(
                        model       = model,
                        messages    = messages,
                        temperature = temperature,
                        max_tokens  = 1000
                    )
                    response = result.choices[0].message.content

                except Exception as e:
                    response = f"Error: {str(e)}. Check your API key in the sidebar."

            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# -- Stats in sidebar -----------------------------------------
with st.sidebar:
    if st.session_state.messages:
        st.divider()
        st.markdown(f"**Messages:** {len(st.session_state.messages)}")
        user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
        st.markdown(f"**Your questions:** {user_msgs}")
"""

# Write app.py to disk with explicit utf-8 encoding
with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

print("  app.py written successfully!")
print("  File: app.py (your Streamlit web app)")


# -- SECTION 3: How to Run ------------------------------------
print("\n\nSECTION 3: How to Run Your App")
print("-" * 40)
print("""
  Step 1: Install Streamlit
  -------------------------
  pip install streamlit openai

  Step 2: Run the app
  -------------------
  streamlit run app.py

  Step 3: Open in browser
  -----------------------
  It opens automatically at: http://localhost:8501

  Step 4: Enter your API key
  --------------------------
  Paste your OpenAI API key in the sidebar -> start chatting!

  Step 5: Deploy for FREE
  -----------------------
  1. Push app.py to GitHub
  2. Go to: https://streamlit.io/cloud
  3. Sign in with GitHub
  4. Click "New app" -> select your repo -> select app.py
  5. Click Deploy -> get a live URL in 2 minutes!
""")


# -- SECTION 4: requirements.txt ------------------------------
print("\nSECTION 4: Creating requirements.txt")
print("-" * 40)

requirements = """streamlit>=1.28.0
openai>=1.0.0
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-community>=0.0.10
chromadb>=0.4.0
sentence-transformers>=2.2.0
numpy>=1.24.0
pandas>=2.0.0
"""

with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write(requirements)

print("  requirements.txt created!")
print(requirements)


# -- SECTION 5: n8n Automation Concepts -----------------------
print("\n\nSECTION 5: AI Automation with n8n (Concepts)")
print("-" * 40)
print("""
n8n = no-code automation tool (like Zapier but more powerful)
You connect services together visually -- no code needed.

Real automation workflows you can build TODAY:

  Workflow 1: Daily AI News Digest
  ---------------------------------
  Trigger: Every morning 8 AM
  -> Fetch top AI news from RSS feeds
  -> Send to GPT: "Summarize in 5 bullets"
  -> Send summary to your WhatsApp/Email

  Workflow 2: LinkedIn Post Generator
  -------------------------------------
  Trigger: Fill a Google Form with today's learning
  -> Send to GPT: "Write a LinkedIn post about this"
  -> Save draft to Google Docs
  -> Notify you on Telegram

  Workflow 3: Resume Screener
  ----------------------------
  Trigger: Email received with PDF attachment
  -> Extract text from PDF
  -> Send to GPT: "Score this resume 1-10 for AI engineer role"
  -> Save score + summary to Google Sheets

HOW TO SET UP n8n (Free):
--------------------------
Option A: n8n cloud free trial
  -> Go to: https://n8n.io
  -> Sign up -> 20 free executions/day

Option B: Self-hosted with Docker (unlimited free)
  -> Install Docker Desktop
  -> Run: docker run -it --rm --name n8n -p 5678:5678 n8nio/n8n
  -> Open: http://localhost:5678
""")


# -- SECTION 6: Summary ---------------------------------------
print("\n\nSECTION 6: Day 4 Complete -- What You Built")
print("=" * 60)
print("""
  Script 1: LangChain Basics
  -> Prompt Templates, LLM Chains, Memory, Output Parsers
  -> Mini project: AI Resume Helper

  Script 2: RAG Pipeline
  -> Embeddings, ChromaDB vector store, text splitting
  -> Full RAG Q&A system on custom documents

  Script 3: Streamlit App (THIS FILE)
  -> Built a complete AI chatbot web app (app.py)
  -> Ready to deploy on Streamlit Cloud for FREE

  FILES CREATED TODAY:
  [OK] script1_langchain_basics.py
  [OK] script2_rag_pipeline.py
  [OK] script3_streamlit_app.py
  [OK] app.py                    <- your deployable web app!
  [OK] requirements.txt          <- for Streamlit Cloud deployment
""")

print("=" * 60)
print("NEXT STEP: Run 'streamlit run app.py' to see your app!")
print("Then deploy it free at streamlit.io/cloud")
print("Share the live URL on LinkedIn for maximum recruiter impact!")
print("=" * 60)
