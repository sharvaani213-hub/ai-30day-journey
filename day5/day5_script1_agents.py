# -*- coding: utf-8 -*-
# ============================================================
# DAY 5 -- SCRIPT 1: AI Agents
# Topics: What are agents, tools, ReAct pattern,
#         building a simple agent, tool calling
# ============================================================

import os
import warnings
import datetime
import math
import json
warnings.filterwarnings("ignore")

print("=" * 60)
print("DAY 5 -- SCRIPT 1: AI Agents")
print("=" * 60)

print("""
WHAT IS AN AI AGENT?
--------------------
Until now: You ask GPT a question -> GPT answers from memory.
           That's it. One shot. Static.

An Agent is different:
  -> The LLM can DECIDE what tools to use
  -> It can SEARCH the web, RUN code, READ files, CALL APIs
  -> It reasons step by step: Think -> Act -> Observe -> Repeat
  -> It keeps going until it solves the problem

Real world example:
  You: "What is the weather in Hyderabad and should I carry an umbrella?"

  Without agent: GPT guesses from training data (probably wrong)

  With agent:
    Step 1: Think  -> "I need current weather data"
    Step 2: Act    -> calls weather API tool
    Step 3: Observe-> gets "28C, 80% humidity, rain likely"
    Step 4: Think  -> "I have the data, now I can answer"
    Step 5: Answer -> "Yes carry an umbrella, 80% chance of rain"

This is called the ReAct pattern: Reasoning + Acting
""")


# -- SECTION 1: Build Tools Manually --------------------------
print("SECTION 1: Building Tools for an Agent")
print("-" * 40)
print("""
Tools = functions the agent can call to get information.
You define them, the agent decides WHEN to use them.
""")

# Tool 1: Calculator
def calculator(expression):
    """
    Evaluates a mathematical expression safely.
    Examples: "2 + 2", "sqrt(16)", "15 * 8 / 3"
    """
    try:
        # Safe evaluation with math functions
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result  = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# Tool 2: Current date/time
def get_current_datetime(query=""):
    """Returns the current date and time."""
    now = datetime.datetime.now()
    return f"Current date: {now.strftime('%A, %d %B %Y')} | Time: {now.strftime('%I:%M %p')}"

# Tool 3: Unit converter
def unit_converter(query):
    """
    Converts between common units.
    Format: "100 km to miles" or "50 celsius to fahrenheit"
    """
    query = query.lower().strip()
    try:
        parts = query.split()
        value = float(parts[0])
        unit_from = parts[1]
        unit_to   = parts[3]

        conversions = {
            ("km",         "miles")     : lambda x: x * 0.621371,
            ("miles",      "km")        : lambda x: x * 1.60934,
            ("kg",         "pounds")    : lambda x: x * 2.20462,
            ("pounds",     "kg")        : lambda x: x * 0.453592,
            ("celsius",    "fahrenheit"): lambda x: (x * 9/5) + 32,
            ("fahrenheit", "celsius")   : lambda x: (x - 32) * 5/9,
            ("meters",     "feet")      : lambda x: x * 3.28084,
            ("feet",       "meters")    : lambda x: x * 0.3048,
            ("lpa",        "monthly")   : lambda x: round(x * 100000 / 12, 2),
            ("monthly",    "lpa")       : lambda x: round(x * 12 / 100000, 2),
        }

        key = (unit_from, unit_to)
        if key in conversions:
            result = conversions[key](value)
            return f"{value} {unit_from} = {round(result, 4)} {unit_to}"
        else:
            return f"Conversion from {unit_from} to {unit_to} not supported."
    except Exception as e:
        return f"Error: {str(e)}. Format: '100 km to miles'"

# Tool 4: Study planner
def study_planner(topic):
    """Returns a quick study plan for a given AI/ML topic."""
    plans = {
        "neural networks": [
            "Day 1: Perceptron, activation functions, forward pass",
            "Day 2: Backpropagation, gradient descent",
            "Day 3: Build with Keras on MNIST",
            "Day 4: CNN for image classification",
            "Day 5: Deploy with Streamlit"
        ],
        "llm": [
            "Day 1: Transformer architecture, attention mechanism",
            "Day 2: HuggingFace pipelines, tokenizers",
            "Day 3: OpenAI API, prompt engineering",
            "Day 4: LangChain basics, chains, memory",
            "Day 5: RAG pipeline with ChromaDB"
        ],
        "python": [
            "Day 1: Variables, loops, functions, OOP",
            "Day 2: NumPy, Pandas basics",
            "Day 3: File I/O, APIs, error handling",
            "Day 4: Virtual environments, pip, modules",
            "Day 5: Build a complete project"
        ]
    }
    topic_lower = topic.lower()
    for key, plan in plans.items():
        if key in topic_lower:
            result = f"5-day study plan for {topic}:\n"
            for step in plan:
                result += f"  {step}\n"
            return result
    return f"No specific plan for '{topic}'. Try: neural networks, llm, python"

# Tool 5: Salary estimator
def salary_estimator(skills):
    """Estimates salary range for AI roles based on skills."""
    skills_lower = skills.lower()
    base = 6

    bonuses = {
        "langchain"   : 2,
        "rag"         : 2,
        "llm"         : 2,
        "pytorch"     : 1.5,
        "tensorflow"  : 1.5,
        "mlops"       : 2,
        "docker"      : 1,
        "aws"         : 1.5,
        "openai"      : 1.5,
        "agents"      : 2,
        "fine-tuning" : 2.5,
    }

    total_bonus = sum(v for k, v in bonuses.items() if k in skills_lower)
    min_sal = round(base + total_bonus * 0.5, 1)
    max_sal = round(base + total_bonus, 1)

    return f"Estimated salary range: {min_sal} - {max_sal} LPA\nKey skills found: {[k for k in bonuses if k in skills_lower]}"

# Test all tools
print("Testing all tools:\n")

tests = [
    ("Calculator",       calculator,       "sqrt(144) + 50 * 2"),
    ("DateTime",         get_current_datetime, ""),
    ("Unit Converter",   unit_converter,   "12 lpa to monthly"),
    ("Unit Converter 2", unit_converter,   "100 km to miles"),
    ("Study Planner",    study_planner,    "LLM and transformers"),
    ("Salary Estimator", salary_estimator, "python langchain rag openai agents"),
]

for name, tool, query in tests:
    result = tool(query)
    print(f"  Tool: {name}")
    print(f"  Input: '{query}'")
    print(f"  Output: {result}")
    print()


# -- SECTION 2: ReAct Agent Logic (Manual) --------------------
print("\nSECTION 2: ReAct Agent -- Manual Implementation")
print("-" * 40)
print("""
ReAct = Reasoning + Acting
The agent thinks step by step before taking action.

Pattern:
  Thought: what do I need to do?
  Action: which tool should I use?
  Action Input: what input to give the tool?
  Observation: what did the tool return?
  ... repeat ...
  Final Answer: the complete answer
""")

class SimpleAgent:
    """
    A simple rule-based agent that demonstrates the ReAct pattern.
    In production this uses an LLM to decide which tool to call.
    Here we simulate the decision logic manually.
    """

    def __init__(self):
        self.tools = {
            "calculator"      : calculator,
            "datetime"        : get_current_datetime,
            "unit_converter"  : unit_converter,
            "study_planner"   : study_planner,
            "salary_estimator": salary_estimator,
        }
        self.steps = []

    def think(self, query):
        """Decide which tool to use based on keywords in query."""
        query_lower = query.lower()

        if any(w in query_lower for w in ["calculate", "compute", "sqrt", "math", "+", "*", "/"]):
            return "calculator", query.split("calculate")[-1].strip() if "calculate" in query_lower else query

        elif any(w in query_lower for w in ["date", "time", "today", "day"]):
            return "datetime", query

        elif any(w in query_lower for w in ["convert", "to miles", "to km", "celsius", "fahrenheit", "lpa", "monthly"]):
            # Extract the conversion part
            for phrase in ["convert ", "what is "]:
                if phrase in query_lower:
                    return "unit_converter", query_lower.replace(phrase, "").strip()
            return "unit_converter", query_lower

        elif any(w in query_lower for w in ["study plan", "learn", "how to study"]):
            topic = query_lower.replace("give me a study plan for", "").replace("how to study", "").strip()
            return "study_planner", topic

        elif any(w in query_lower for w in ["salary", "earn", "pay", "ctc", "lpa"]):
            return "salary_estimator", query

        else:
            return None, None

    def run(self, query):
        """Run the agent on a query."""
        print(f"\n  Query: {query}")
        print(f"  {'─'*50}")

        # Step 1: Think
        tool_name, tool_input = self.think(query)
        print(f"  Thought: I need to figure out what this query needs...")

        if tool_name:
            print(f"  Action: Use '{tool_name}' tool")
            print(f"  Action Input: '{tool_input}'")

            # Step 2: Act
            tool_fn     = self.tools[tool_name]
            observation = tool_fn(tool_input)
            print(f"  Observation: {observation}")

            # Step 3: Final answer
            answer = f"Based on the {tool_name} tool: {observation}"
        else:
            print(f"  Thought: No specific tool needed, answering directly")
            answer = f"I can help with calculations, date/time, unit conversions, study plans, and salary estimation. Please rephrase your query."

        print(f"  Final Answer: {answer}")
        return answer

# Test the agent
agent = SimpleAgent()

queries = [
    "What is today's date and time?",
    "Convert 15 lpa to monthly salary",
    "Calculate sqrt(256) + 100",
    "Give me a study plan for LLM and transformers",
    "What salary can I expect with python langchain rag openai skills?",
    "Convert 200 km to miles",
]

print("\nRunning agent on test queries:")
for query in queries:
    agent.run(query)


# -- SECTION 3: LangChain Agent (with API) --------------------
print("\n\nSECTION 3: LangChain Agent with Real LLM")
print("-" * 40)

API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
API_AVAILABLE = API_KEY != "your-api-key-here"

if API_AVAILABLE:
    try:
        from langchain_openai import ChatOpenAI
        from langchain.agents import AgentExecutor, create_react_agent
        from langchain.tools import tool
        from langchain import hub

        llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-3.5-turbo", temperature=0)

        @tool
        def calc_tool(expression: str) -> str:
            """Evaluates a math expression. Input: math expression like '2+2' or 'sqrt(16)'"""
            return calculator(expression)

        @tool
        def date_tool(query: str) -> str:
            """Returns the current date and time."""
            return get_current_datetime()

        @tool
        def convert_tool(query: str) -> str:
            """Converts units. Format: '100 km to miles' or '50 celsius to fahrenheit'"""
            return unit_converter(query)

        tools = [calc_tool, date_tool, convert_tool]

        prompt   = hub.pull("hwchase17/react")
        agent    = create_react_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

        print("  Running LangChain agent with real LLM...\n")
        result = executor.invoke({"input": "What is today's date and what is 15 LPA divided by 12 months?"})
        print(f"\n  Final Answer: {result['output']}")

    except Exception as e:
        print(f"  Error: {e}")
        print("  Run: pip install langchain langchain-openai")
else:
    print("""
  [Needs API key -- example of what LangChain agent output looks like:]

  Query: "What is today's date and what is 15 LPA divided by 12 months?"

  Thought: I need today's date and also do a calculation.
  Action: date_tool
  Observation: Current date: Monday, 28 April 2025 | Time: 10:30 AM

  Thought: Now I need to calculate 15,00,000 / 12
  Action: calc_tool
  Action Input: 1500000 / 12
  Observation: Result: 125000.0

  Final Answer: Today is Monday, 28 April 2025.
                15 LPA = Rs. 1,25,000 per month.
  """)


# -- SECTION 4: Agent Use Cases for AI Engineer ---------------
print("\n\nSECTION 4: Real Agent Use Cases You Can Build")
print("-" * 40)
print("""
These are actual projects that impress recruiters:

  1. Job Research Agent
     -> Searches LinkedIn/Naukri for AI jobs
     -> Extracts required skills from job descriptions
     -> Compares with your skills
     -> Tells you exactly what to learn next

  2. Code Review Agent
     -> You paste your Python code
     -> Agent runs it, checks for bugs
     -> Searches StackOverflow for solutions
     -> Returns fixed code with explanation

  3. Study Assistant Agent
     -> You ask "explain backpropagation"
     -> Agent searches latest papers
     -> Summarizes in simple language
     -> Generates a quiz to test you

  4. Resume Optimizer Agent
     -> You paste job description + your resume
     -> Agent analyzes skill gaps
     -> Rewrites your resume bullets
     -> Suggests projects to add

  5. Daily News Summarizer Agent
     -> Runs every morning
     -> Fetches top AI news
     -> Summarizes to 5 bullets
     -> Sends to your WhatsApp
""")

print("=" * 60)
print("Script 1 complete! AI Agents covered.")
print("Key concepts:")
print("  [OK] What agents are and why they matter")
print("  [OK] Tools -- functions the agent can call")
print("  [OK] ReAct pattern -- Think, Act, Observe")
print("  [OK] Manual agent implementation")
print("  [OK] LangChain agent with real LLM")
print("  [OK] Real-world agent use cases")
print("=" * 60)
