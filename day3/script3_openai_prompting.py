# ============================================================
# DAY 3 — SCRIPT 3: OpenAI API + Prompt Engineering
# Topics: API setup, chat completions, system prompts,
#         temperature, few-shot, chain-of-thought, roles
# ============================================================

import os
import json

print("=" * 60)
print("DAY 3 — SCRIPT 3: OpenAI API + Prompt Engineering")
print("=" * 60)

# ── SECTION 1: Setup ─────────────────────────────────────────
print("""
SECTION 1: Setup — Getting Your OpenAI API Key
───────────────────────────────────────────────
1. Go to: https://platform.openai.com/api-keys
2. Sign up / Log in
3. Click "Create new secret key"
4. Copy the key (starts with sk-...)
5. Replace "your-api-key-here" below with your actual key

IMPORTANT: Never push your API key to GitHub!
We use environment variables to keep it safe.
""")

# Set your API key here (for learning — in production use .env file)
# Replace the string below with your actual key
API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key-here")

try:
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)
    API_AVAILABLE = API_KEY != "your-api-key-here"
except ImportError:
    print("openai not installed. Run: pip install openai")
    API_AVAILABLE = False


def call_gpt(messages, temperature=0.7, model="gpt-3.5-turbo"):
    """
    Helper function to call OpenAI API.
    messages = list of dicts with 'role' and 'content'
    """
    if not API_AVAILABLE:
        return "[API key not set — showing example output]"
    try:
        response = client.chat.completions.create(
            model       = model,
            messages    = messages,
            temperature = temperature,
            max_tokens  = 500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error: {e}]"


# ── SECTION 2: Basic API Call ──────────────────────────────────
print("\nSECTION 2: Basic API Call Structure")
print("-" * 40)
print("""
Every OpenAI API call has 3 message roles:

  SYSTEM  → sets the personality/context of the AI
  USER    → the human's message
  ASSISTANT → the AI's previous responses (for multi-turn)

Think of it like directing an actor:
  System = "You are a strict teacher"
  User   = "Explain neural networks"
  → The AI responds AS a strict teacher
""")

basic_messages = [
    {"role": "system",  "content": "You are a helpful AI tutor for engineering students."},
    {"role": "user",    "content": "Explain what a neural network is in 3 simple sentences."}
]

print("Basic API call:")
print(f"  System: {basic_messages[0]['content']}")
print(f"  User  : {basic_messages[1]['content']}")
print(f"\n  Response:")
response = call_gpt(basic_messages)
print(f"  {response}")

if not API_AVAILABLE:
    print("""
  EXAMPLE OUTPUT:
  A neural network is a system inspired by the human brain that
  consists of layers of connected nodes called neurons. Each neuron
  receives inputs, applies weights and an activation function, and
  passes the result to the next layer. By training on data, neural
  networks learn to recognize patterns and make predictions.
  """)


# ── SECTION 3: Temperature — Controlling Creativity ───────────
print("\n\nSECTION 3: Temperature — Controlling Creativity")
print("-" * 40)
print("""
Temperature controls randomness:
  0.0 → deterministic, always same answer (use for facts/code)
  0.7 → balanced (most common, good default)
  1.5 → very creative/random (use for stories/brainstorming)
""")

prompt = "Give me a creative name for an AI startup."
temperatures = [0.1, 0.7, 1.4]

for temp in temperatures:
    msgs = [{"role": "user", "content": prompt}]
    result = call_gpt(msgs, temperature=temp)
    print(f"  Temperature {temp}: {result[:80]}")

if not API_AVAILABLE:
    print("""
  Temperature 0.1: NeuralEdge AI
  Temperature 0.7: CognifyLabs
  Temperature 1.4: Synaptic Prism Ventures
  """)


# ── SECTION 4: System Prompt Engineering ──────────────────────
print("\n\nSECTION 4: System Prompt Engineering")
print("-" * 40)
print("Same question, different system prompts = completely different responses\n")

question = "How should I prepare for an AI job interview?"

personas = [
    ("Strict professor", "You are a strict professor who gives blunt, no-nonsense advice."),
    ("Career coach",     "You are an encouraging career coach who motivates students."),
    ("Senior engineer",  "You are a senior AI engineer at Google with 10 years experience."),
]

for name, system_prompt in personas:
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question}
    ]
    result = call_gpt(msgs, temperature=0.7)
    print(f"  [{name}]")
    print(f"  {result[:150]}...")
    print()

if not API_AVAILABLE:
    print("""
  [Strict professor]
  Study the fundamentals. I don't care about your projects if you
  can't explain gradient descent. Master linear algebra and prob...

  [Career coach]
  You've got this! Start by showcasing your projects on GitHub.
  Practice speaking about your work confidently...

  [Senior engineer]
  Focus on system design — most candidates fail there. Know your
  ML fundamentals cold, and have at least 2 deployed projects...
  """)


# ── SECTION 5: Few-Shot Prompting ─────────────────────────────
print("\n\nSECTION 5: Few-Shot Prompting")
print("-" * 40)
print("""
Few-shot = give examples IN the prompt so the model follows your format.
This is the most powerful prompt engineering technique.
""")

few_shot_prompt = """
You classify student queries into categories.

Examples:
Query: "How do I fix a segmentation fault in C?"
Category: DEBUGGING

Query: "What is the difference between ML and DL?"
Category: CONCEPT_QUESTION

Query: "Can you write a Python function to sort a list?"
Category: CODE_REQUEST

Query: "I'm feeling overwhelmed with my studies"
Category: EMOTIONAL_SUPPORT

Now classify these:
Query: "Explain what gradient descent does"
Category:"""

msgs = [{"role": "user", "content": few_shot_prompt}]
result = call_gpt(msgs, temperature=0.1)
print(f"  Few-shot prompt output: {result}")

if not API_AVAILABLE:
    print("  Few-shot prompt output: CONCEPT_QUESTION")

print("\n  Why it works: Examples teach the FORMAT without extra instructions.")


# ── SECTION 6: Chain-of-Thought Prompting ─────────────────────
print("\n\nSECTION 6: Chain-of-Thought (CoT) Prompting")
print("-" * 40)
print("""
CoT = ask the model to THINK STEP BY STEP before answering.
This dramatically improves accuracy on reasoning tasks.
Just add "Let's think step by step" or "Think through this carefully."
""")

# Without CoT
simple_prompt = "A student scores 85, 92, and 78 on 3 tests. What is the minimum score needed on the 4th test to get an average of 88?"

msgs_simple = [{"role": "user", "content": simple_prompt}]
result_simple = call_gpt(msgs_simple, temperature=0)
print(f"  Without CoT: {result_simple[:100]}")

# With CoT
cot_prompt = simple_prompt + "\n\nLet's think step by step."
msgs_cot = [{"role": "user", "content": cot_prompt}]
result_cot = call_gpt(msgs_cot, temperature=0)
print(f"\n  With CoT: {result_cot[:300]}")

if not API_AVAILABLE:
    print("""
  Without CoT: The minimum score needed is 97.

  With CoT:
  Step 1: Current total = 85 + 92 + 78 = 255
  Step 2: Target total for 88 avg over 4 tests = 88 × 4 = 352
  Step 3: Score needed = 352 - 255 = 97
  Therefore, the student needs at least 97 on the 4th test.
  """)


# ── SECTION 7: Multi-Turn Conversation ────────────────────────
print("\n\nSECTION 7: Multi-Turn Conversation (Memory)")
print("-" * 40)
print("To maintain context, include the full conversation history each time\n")

conversation_history = [
    {"role": "system", "content": "You are an AI tutor helping a student learn machine learning."}
]

# Simulate a 3-turn conversation
turns = [
    "What is overfitting?",
    "How do I prevent it?",
    "Give me a Python code example of using dropout."
]

for turn in turns:
    conversation_history.append({"role": "user", "content": turn})
    response = call_gpt(conversation_history, temperature=0.5)
    conversation_history.append({"role": "assistant", "content": response})

    print(f"  User : {turn}")
    print(f"  GPT  : {response[:150]}...")
    print()

if not API_AVAILABLE:
    print("""
  User : What is overfitting?
  GPT  : Overfitting is when a model memorizes the training data instead
         of learning general patterns, causing it to perform poorly on new data...

  User : How do I prevent it?
  GPT  : Great follow-up! Since you now know what overfitting is, here are
         the main ways to prevent it: Dropout, Regularization (L1/L2)...

  User : Give me a Python code example of using dropout.
  GPT  : Building on our conversation, here's a Keras example:
         model.add(Dropout(0.3))  # drops 30% of neurons randomly...
  """)


# ── SECTION 8: 20 Prompt Patterns ────────────────────────────
print("\n\nSECTION 8: 20 Essential Prompt Patterns (Save This!)")
print("-" * 40)

patterns = [
    ("1.  Zero-shot",         "Just ask directly. 'Explain transformers.'"),
    ("2.  Few-shot",          "Give 2-3 examples before asking."),
    ("3.  Chain-of-Thought",  "Add 'Think step by step' to any question."),
    ("4.  Role prompting",    "'You are a senior ML engineer. Review my code.'"),
    ("5.  Format control",    "'Respond ONLY in JSON with keys: name, score.'"),
    ("6.  Constraint prompting","'Explain ML in exactly 3 bullet points, max 10 words each.'"),
    ("7.  Persona prompting", "'Explain like I am a 5 year old.'"),
    ("8.  Negative prompting","'Do NOT use jargon. Do NOT exceed 100 words.'"),
    ("9.  Template filling",  "'Fill this template: Name: ___, Skills: ___'"),
    ("10. Iterative refining","'Make it shorter.' 'Make it more technical.'"),
    ("11. Self-consistency",  "Ask same question 3 times → pick most common answer."),
    ("12. Decomposition",     "'Break this complex task into 5 subtasks.'"),
    ("13. Verification",      "'Double check your answer. Is it correct?'"),
    ("14. Analogical",        "'Explain backprop using a real-world analogy.'"),
    ("15. Socratic",          "'Ask me questions to understand my problem better.'"),
    ("16. Pros/Cons",         "'List pros and cons of using PyTorch vs TensorFlow.'"),
    ("17. Tree of Thought",   "'Think of 3 different approaches, then pick the best.'"),
    ("18. ReAct",             "'Think, then Act. Reason before every step.'"),
    ("19. Output anchoring",  "'Start your response with: Here are the exact steps:'"),
    ("20. Context injection", "Paste relevant docs/code BEFORE asking your question."),
]

for pattern, description in patterns:
    print(f"  {pattern:<25}: {description}")


# ── SECTION 9: Build a Simple Chatbot ────────────────────────
print("\n\nSECTION 9: Building a Simple CLI Chatbot")
print("-" * 40)

def run_chatbot():
    """Simple command-line chatbot using OpenAI API"""
    print("  Simple AI Chatbot (type 'quit' to exit)\n")

    history = [
        {"role": "system", "content":
         "You are a helpful AI tutor for a computer engineering student "
         "learning AI/ML. Be concise, friendly, and use examples."}
    ]

    while True:
        user_input = input("  You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("  Chatbot: Good luck with your AI journey! 🚀")
            break

        history.append({"role": "user", "content": user_input})
        response = call_gpt(history, temperature=0.7)
        history.append({"role": "assistant", "content": response})

        print(f"  Bot: {response}\n")

if API_AVAILABLE:
    print("  Starting chatbot... (type 'quit' to exit)")
    run_chatbot()
else:
    print("""
  [Chatbot Demo — set your API key to run live]

  You: What is the difference between BERT and GPT?
  Bot: Great question! Here's a quick comparison:
       BERT  = Bidirectional, reads text left+right simultaneously
               Best for: classification, NER, Q&A
       GPT   = Unidirectional, reads left to right, generates text
               Best for: chatbots, code generation, summarization

  You: Which one should I learn first?
  Bot: Start with GPT via the OpenAI API since you can build
       useful apps immediately. Then explore BERT for NLP tasks.
  """)

print()
print("=" * 60)
print("Script 3 complete! OpenAI API + Prompt Engineering done.")
print("Key concepts covered:")
print("  ✓ OpenAI API structure (system/user/assistant)")
print("  ✓ Temperature control")
print("  ✓ System prompt engineering")
print("  ✓ Few-shot prompting")
print("  ✓ Chain-of-Thought prompting")
print("  ✓ Multi-turn conversations")
print("  ✓ 20 essential prompt patterns")
print("  ✓ Building a CLI chatbot")
print()
print("NEXT STEP: Get your OpenAI API key from platform.openai.com")
print("           Replace 'your-api-key-here' and run the chatbot live!")
print("=" * 60)
