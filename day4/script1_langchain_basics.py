# ============================================================
# DAY 4 — SCRIPT 1: LangChain Basics
# Topics: LangChain chains, prompt templates, memory,
#         output parsers, LLM wrappers
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("DAY 4 — SCRIPT 1: LangChain Basics")
print("=" * 60)

print("""
WHAT IS LANGCHAIN?
──────────────────
LangChain is a framework that makes it easy to build
applications powered by LLMs.

Without LangChain:
  → You manually manage prompts, memory, API calls, parsing

With LangChain:
  → All of that is handled by pre-built components you just plug together

Think of it like: React for AI apps.

Core components:
  1. LLMs / ChatModels   → the AI brain (GPT, Claude, etc.)
  2. Prompt Templates    → reusable prompt structures
  3. Chains              → connect components together
  4. Memory              → remember conversation history
  5. Output Parsers      → extract structured data from responses
  6. Agents              → LLM that decides what tools to use
""")

# ── Check installations ───────────────────────────────────────
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, PromptTemplate
    from langchain.schema import HumanMessage, SystemMessage, AIMessage
    from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
    from langchain.chains import LLMChain, ConversationChain
    from langchain.output_parsers import CommaSeparatedListOutputParser
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    LANGCHAIN_AVAILABLE = True
    print("✓ LangChain installed successfully!")
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"LangChain not fully installed: {e}")
    print("Run: pip install langchain langchain-openai langchain-community")

# API Key setup
API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
API_AVAILABLE = API_KEY != "your-api-key-here" and LANGCHAIN_AVAILABLE


# ── SECTION 1: Prompt Templates ──────────────────────────────
print("\n\nSECTION 1: Prompt Templates")
print("-" * 40)
print("""
Prompt Templates = reusable prompts with variables.
Instead of writing the full prompt every time, define it once
and just fill in the variables.
""")

if LANGCHAIN_AVAILABLE:
    # Basic prompt template
    template = PromptTemplate(
        input_variables=["topic", "level"],
        template="Explain {topic} to a {level} student in 3 bullet points."
    )

    # Fill in the template
    prompt1 = template.format(topic="neural networks", level="beginner")
    prompt2 = template.format(topic="transformers",    level="intermediate")
    prompt3 = template.format(topic="RAG pipelines",   level="advanced")

    print("Template: 'Explain {topic} to a {level} student in 3 bullet points.'")
    print(f"\nFilled prompt 1: {prompt1}")
    print(f"Filled prompt 2: {prompt2}")
    print(f"Filled prompt 3: {prompt3}")

    # Chat Prompt Template (for ChatGPT-style models)
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert {domain} tutor. Be concise and clear."),
        ("human",  "Explain {concept} with one real-world example.")
    ])

    filled = chat_template.format_messages(
        domain  = "machine learning",
        concept = "overfitting"
    )
    print(f"\nChat template filled:")
    for msg in filled:
        print(f"  {msg.type.upper()}: {msg.content}")
else:
    print("  Install LangChain to run this section")
    print("  pip install langchain langchain-openai")


# ── SECTION 2: LLM + Chain ────────────────────────────────────
print("\n\nSECTION 2: LLM Chain — Prompt + Model + Parser")
print("-" * 40)
print("""
A Chain connects:
  Prompt Template → LLM → Output Parser

This is the simplest and most common LangChain pattern.
New style uses | (pipe) operator: prompt | llm | parser
""")

if API_AVAILABLE:
    llm = ChatOpenAI(
        openai_api_key = API_KEY,
        model          = "gpt-3.5-turbo",
        temperature    = 0.7
    )

    # Simple chain using LCEL (LangChain Expression Language)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI career advisor for engineering students."),
        ("human",  "{question}")
    ])

    parser = StrOutputParser()

    # Chain = prompt | llm | parser
    chain = prompt | llm | parser

    questions = [
        "What are the top 5 skills needed to become an AI engineer?",
        "How long does it take to learn machine learning from scratch?",
    ]

    for q in questions:
        print(f"\n  Q: {q}")
        response = chain.invoke({"question": q})
        print(f"  A: {response[:200]}...")

else:
    print("  [API key needed to run live chains]")
    print("""
  Example output:

  Q: What are the top 5 skills to become an AI engineer?
  A: 1. Python programming
     2. Machine learning fundamentals (scikit-learn)
     3. Deep learning (TensorFlow/PyTorch)
     4. LLM APIs (OpenAI, HuggingFace)
     5. MLOps & deployment (Docker, cloud platforms)

  Q: How long to learn ML from scratch?
  A: With dedicated study (6-8 hours/day), you can get job-ready
     in 3-6 months. Focus on projects over theory.
  """)


# ── SECTION 3: Memory ─────────────────────────────────────────
print("\n\nSECTION 3: Memory — Making AI Remember Conversations")
print("-" * 40)
print("""
Without memory: every message is independent — AI forgets everything.
With memory   : AI remembers what was said earlier in conversation.

Types of memory:
  ConversationBufferMemory  → stores ALL messages (simple, gets expensive)
  ConversationSummaryMemory → summarizes old messages (efficient for long chats)
  ConversationWindowMemory  → remembers last K messages only
""")

if LANGCHAIN_AVAILABLE:
    # Show how memory works without API
    memory = ConversationBufferMemory()

    # Simulate adding conversation turns manually
    memory.chat_memory.add_user_message("My name is Sharvaani and I'm learning AI.")
    memory.chat_memory.add_ai_message("Great to meet you Sharvaani! AI is a fantastic field to learn.")
    memory.chat_memory.add_user_message("I want to become an AI engineer in Hyderabad.")
    memory.chat_memory.add_ai_message("Hyderabad has a growing AI ecosystem. Focus on LLMs and RAG.")

    print("  Memory contents:")
    for msg in memory.chat_memory.messages:
        role = "Human" if msg.type == "human" else "AI"
        print(f"    {role}: {msg.content[:80]}")

    print(f"\n  Total messages stored: {len(memory.chat_memory.messages)}")
    print("  The LLM receives ALL of this as context on next message.")

if API_AVAILABLE:
    print("\n  Running live conversation with memory...")
    conversation = ConversationChain(
        llm    = ChatOpenAI(openai_api_key=API_KEY, model="gpt-3.5-turbo"),
        memory = ConversationBufferMemory(),
        verbose= False
    )

    turns = [
        "My name is Sharvaani. I'm a BE Computer Engineering final year student.",
        "I want to become an AI engineer. What should I focus on?",
        "What was my name again? And what did I say I wanted to become?"
    ]

    for turn in turns:
        response = conversation.predict(input=turn)
        print(f"\n  Human: {turn}")
        print(f"  AI   : {response[:150]}...")
else:
    print("""
  [Live conversation example — needs API key]

  Human: My name is Sharvaani. I'm a BE CS final year student.
  AI   : Nice to meet you, Sharvaani! What can I help you with?

  Human: I want to become an AI engineer. What should I focus on?
  AI   : Great goal! For an AI engineer role, focus on Python,
         LLMs, RAG pipelines, and build 3 solid projects...

  Human: What was my name again?
  AI   : Your name is Sharvaani! You mentioned you're a BE CS
         final year student aiming to become an AI engineer.
         ← Memory working! It remembered from earlier.
  """)


# ── SECTION 4: Output Parsers ──────────────────────────────────
print("\n\nSECTION 4: Output Parsers — Get Structured Data")
print("-" * 40)
print("""
By default, LLMs return raw text strings.
Output parsers extract structured data (lists, JSON, objects).
""")

if LANGCHAIN_AVAILABLE:
    # CommaSeparatedListOutputParser
    list_parser = CommaSeparatedListOutputParser()

    format_instructions = list_parser.get_format_instructions()
    print(f"  Format instructions: {format_instructions}")

    if API_AVAILABLE:
        prompt = PromptTemplate(
            template = "List the top 5 Python libraries for machine learning.\n{format_instructions}",
            input_variables    = [],
            partial_variables  = {"format_instructions": format_instructions}
        )

        chain  = prompt | ChatOpenAI(openai_api_key=API_KEY) | list_parser
        result = chain.invoke({})

        print(f"\n  Parsed as Python list:")
        for i, item in enumerate(result, 1):
            print(f"    {i}. {item}")
    else:
        print("""
  Example parsed output (list):
    1. scikit-learn
    2. TensorFlow
    3. PyTorch
    4. Pandas
    5. NumPy
  """)


# ── SECTION 5: Sequential Chain ───────────────────────────────
print("\n\nSECTION 5: Sequential Chain — Multi-Step AI Pipeline")
print("-" * 40)
print("""
Sequential chains pass output of one chain as input to next.

Example pipeline:
  Step 1: Generate a resume bullet point from job description
  Step 2: Improve the bullet point to be more impactful
  Step 3: Translate to a specific tone (confident, humble, etc.)
""")

if API_AVAILABLE:
    llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-3.5-turbo", temperature=0.7)
    parser = StrOutputParser()

    # Step 1: Generate bullet
    prompt1 = ChatPromptTemplate.from_template(
        "Write ONE resume bullet point for this achievement: {achievement}"
    )

    # Step 2: Improve it
    prompt2 = ChatPromptTemplate.from_template(
        "Make this resume bullet point more impactful with numbers/metrics: {bullet}"
    )

    # Full pipeline
    chain1 = prompt1 | llm | parser
    chain2 = prompt2 | llm | parser

    achievement = "Built a chatbot using Python and OpenAI API that answers student questions"

    bullet   = chain1.invoke({"achievement": achievement})
    improved = chain2.invoke({"bullet": bullet})

    print(f"  Achievement : {achievement}")
    print(f"\n  Step 1 (raw bullet)    : {bullet}")
    print(f"\n  Step 2 (improved)      : {improved}")

else:
    print("""
  Example pipeline output:

  Achievement: Built a chatbot using Python and OpenAI API

  Step 1: Developed an AI-powered chatbot using Python and OpenAI API
          to automate student question answering.

  Step 2: Engineered an AI chatbot using Python & OpenAI API,
          reducing student query response time by 80% and handling
          200+ questions/day with 94% accuracy.
  """)


# ── SECTION 6: Build a Resume Helper App ─────────────────────
print("\n\nSECTION 6: Mini Project — AI Resume Helper")
print("-" * 40)
print("A complete LangChain app: takes job description → generates resume points")

if API_AVAILABLE:
    llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-3.5-turbo", temperature=0.7)

    resume_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert resume writer for AI/ML roles.
        Given a job description, generate 5 strong resume bullet points
        that a fresher BE Computer Engineering student should highlight.
        Make each bullet point start with a strong action verb.
        Include metrics where possible."""),
        ("human", "Job Description: {job_description}\nStudent Skills: {skills}")
    ])

    chain  = resume_prompt | llm | StrOutputParser()

    job_desc = """
    We are looking for an AI Engineer who can:
    - Build and deploy LLM-powered applications
    - Work with Python, LangChain, and vector databases
    - Implement RAG pipelines for document Q&A
    - Collaborate with product teams to ship AI features
    """

    skills = "Python, NumPy, Pandas, scikit-learn, TensorFlow, LangChain, OpenAI API"

    result = chain.invoke({"job_description": job_desc, "skills": skills})
    print(f"\n  Generated resume bullets:")
    print(result)
else:
    print("""
  [Needs API key — example output:]

  Generated resume bullets for AI Engineer role:

  • Engineered 3 end-to-end ML pipelines using Python and scikit-learn,
    achieving 94% classification accuracy on 10,000+ sample datasets.

  • Developed LangChain-powered document Q&A system using RAG architecture
    with ChromaDB vector store, reducing information retrieval time by 70%.

  • Implemented neural network models using TensorFlow/Keras, training on
    MNIST achieving 97.2% test accuracy across 60,000 samples.

  • Built OpenAI API-integrated chatbot handling multi-turn conversations
    with persistent memory using ConversationBufferMemory.

  • Automated data preprocessing pipelines with Pandas and NumPy, processing
    120+ student records with missing value handling and feature engineering.
  """)

print()
print("=" * 60)
print("Script 1 complete! LangChain basics covered.")
print("Key concepts:")
print("  ✓ Prompt Templates (reusable prompts)")
print("  ✓ LLM Chains (prompt | llm | parser)")
print("  ✓ Memory types (Buffer, Summary)")
print("  ✓ Output Parsers (list, JSON)")
print("  ✓ Sequential chains (multi-step pipelines)")
print("  ✓ Mini project: AI Resume Helper")
print("=" * 60)
