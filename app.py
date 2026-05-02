# ============================================================
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
                response = """Please enter your OpenAI API key in the sidebar to get AI responses.

**Get your free API key:**
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up / Log in
3. Go to API Keys -> Create new key
4. Paste it in the sidebar"""

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
