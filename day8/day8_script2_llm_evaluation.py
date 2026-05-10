# -*- coding: utf-8 -*-
# ============================================================
# DAY 8 -- SCRIPT 2: LLM Evaluation
# Topics: how to measure LLM output quality, BLEU, ROUGE,
#         G-Eval, LLM-as-judge, building an eval pipeline
# ============================================================

import numpy as np
import json
import re
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("DAY 8 -- SCRIPT 2: LLM Evaluation")
print("=" * 60)

print("""
WHY LLM EVALUATION MATTERS
---------------------------
You built a RAG chatbot. But HOW GOOD is it?

Without evaluation:
  -> You do not know if it halluccinates
  -> You cannot compare two versions of your app
  -> You cannot convince a company your app is reliable
  -> You cannot improve what you cannot measure

With evaluation:
  -> "Version 2 of our RAG has 94% faithfulness vs 71% before"
  -> "Our chatbot answers 87% of questions correctly"
  -> "After re-ranking, retrieval accuracy went from 65% to 91%"

These numbers get you hired and promoted.

Types of LLM evaluation:
  1. Reference-based  -> compare to a known correct answer
  2. Reference-free   -> judge quality without a correct answer
  3. LLM-as-judge     -> use another LLM to score the output
  4. Human evaluation -> actual humans rate the outputs
""")


# -- SECTION 1: Reference-Based Metrics ---------------------
print("SECTION 1: Reference-Based Metrics")
print("-" * 40)
print("""
These metrics compare generated text to a reference (correct) answer.
Used when you KNOW what the correct answer should be.

BLEU  -> measures n-gram overlap (used for translation)
ROUGE -> measures recall of reference words (used for summarization)
Exact Match -> 1 if exactly matches, 0 otherwise
F1 Token    -> token-level F1 between prediction and reference
""")

def tokenize(text):
    """Simple tokenizer - split into words, lowercase."""
    return re.findall(r'\b\w+\b', text.lower())

def exact_match(prediction, reference):
    """1 if prediction exactly matches reference, else 0."""
    return 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0

def f1_token_score(prediction, reference):
    """Token-level F1 score between prediction and reference."""
    pred_tokens = set(tokenize(prediction))
    ref_tokens  = set(tokenize(reference))

    if not pred_tokens or not ref_tokens:
        return 0.0

    common    = pred_tokens.intersection(ref_tokens)
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)

def rouge_l(prediction, reference):
    """
    ROUGE-L: Longest Common Subsequence based recall.
    Measures how much of the reference appears in the prediction.
    """
    pred_tokens = tokenize(prediction)
    ref_tokens  = tokenize(reference)

    if not ref_tokens:
        return 0.0

    # Count matching tokens (order matters for LCS, simplifying here)
    pred_set    = set(pred_tokens)
    matches     = sum(1 for t in ref_tokens if t in pred_set)
    recall      = matches / len(ref_tokens)
    precision   = matches / len(pred_tokens) if pred_tokens else 0
    f1          = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0)
    return round(f1, 4)

def bleu_1gram(prediction, reference):
    """
    BLEU-1: unigram precision.
    Measures what fraction of prediction words appear in reference.
    """
    pred_tokens = tokenize(prediction)
    ref_tokens  = set(tokenize(reference))

    if not pred_tokens:
        return 0.0

    matches   = sum(1 for t in pred_tokens if t in ref_tokens)
    precision = matches / len(pred_tokens)
    return round(precision, 4)

# Test with example QA pairs
qa_examples = [
    {
        "question"  : "What is RAG?",
        "reference" : "RAG stands for Retrieval Augmented Generation. It combines document retrieval with LLM generation to reduce hallucinations.",
        "prediction": "RAG is Retrieval Augmented Generation, which retrieves relevant documents before generating an answer with an LLM.",
    },
    {
        "question"  : "What is the salary for an AI engineer fresher in India?",
        "reference" : "AI engineers in India earn 8-15 LPA as freshers.",
        "prediction": "Fresh AI engineers can expect salaries between 8 to 12 LPA in India, depending on skills and location.",
    },
    {
        "question"  : "What is ChromaDB?",
        "reference" : "ChromaDB is an open-source vector database for storing embeddings locally.",
        "prediction": "ChromaDB is used for cooking recipes and meal planning.",  # wrong answer
    },
]

print("\n  Evaluation results on 3 QA examples:\n")
print(f"  {'Q':<45} {'EM':>5} {'F1':>7} {'ROUGE':>7} {'BLEU':>7}")
print(f"  {'─'*45} {'─'*5} {'─'*7} {'─'*7} {'─'*7}")

all_scores = []
for ex in qa_examples:
    em    = exact_match(ex["prediction"], ex["reference"])
    f1    = f1_token_score(ex["prediction"], ex["reference"])
    rouge = rouge_l(ex["prediction"], ex["reference"])
    bleu  = bleu_1gram(ex["prediction"], ex["reference"])
    all_scores.append({"em": em, "f1": f1, "rouge": rouge, "bleu": bleu})
    q_short = ex["question"][:43]
    print(f"  {q_short:<45} {em:>5.1f} {f1:>7.3f} {rouge:>7.3f} {bleu:>7.3f}")

print(f"\n  Average scores:")
for metric in ["em", "f1", "rouge", "bleu"]:
    avg = sum(s[metric] for s in all_scores) / len(all_scores)
    print(f"    {metric.upper():<10}: {avg:.3f}")


# -- SECTION 2: Reference-Free Metrics ----------------------
print("\n\nSECTION 2: Reference-Free Metrics")
print("-" * 40)
print("""
When you do NOT have a reference answer, you can still evaluate:

  Faithfulness    -> Is the answer supported by the context?
  Relevance       -> Does it answer the actual question?
  Completeness    -> Does it cover all aspects of the question?
  Conciseness     -> Is it appropriately brief (not too long)?
  Fluency         -> Is it grammatically correct and readable?
""")

def faithfulness_score(answer, context):
    """
    Check if answer claims are supported by context.
    Score: fraction of answer sentences supported by context.
    """
    answer_sentences  = [s.strip() for s in answer.split(".") if s.strip()]
    context_words     = set(tokenize(context))
    supported         = 0

    for sentence in answer_sentences:
        sent_words = set(tokenize(sentence))
        overlap    = len(sent_words.intersection(context_words))
        support    = overlap / len(sent_words) if sent_words else 0
        if support > 0.3:   # 30% word overlap = considered supported
            supported += 1

    score = supported / len(answer_sentences) if answer_sentences else 0
    return round(score, 3)

def relevance_score(question, answer):
    """Check if the answer is relevant to the question."""
    q_words  = set(tokenize(question))
    a_words  = set(tokenize(answer))
    overlap  = len(q_words.intersection(a_words))
    score    = overlap / len(q_words) if q_words else 0
    return round(min(score * 2, 1.0), 3)  # scale up

def conciseness_score(answer, ideal_length=150):
    """Penalize answers that are too long or too short."""
    length = len(answer.split())
    if length < 10:
        return 0.3   # too short
    elif length <= ideal_length:
        return 1.0   # just right
    else:
        penalty = (length - ideal_length) / ideal_length
        return round(max(0.0, 1.0 - penalty * 0.5), 3)

# Test reference-free metrics
context = """
RAG stands for Retrieval Augmented Generation. It is an AI architecture
that combines document retrieval with language model generation.
RAG reduces hallucinations by grounding answers in retrieved context.
ChromaDB and Pinecone are popular vector databases used in RAG systems.
"""

test_answers = [
    {
        "label" : "Good answer",
        "answer": "RAG is Retrieval Augmented Generation, an architecture that retrieves relevant documents and uses them to ground LLM responses, reducing hallucinations."
    },
    {
        "label" : "Hallucinated answer",
        "answer": "RAG was invented by Google in 2018 and is primarily used for image recognition tasks in autonomous vehicles."
    },
    {
        "label" : "Too vague answer",
        "answer": "RAG is a type of AI system used in many applications."
    },
    {
        "label" : "Too long answer",
        "answer": "RAG stands for Retrieval Augmented Generation. " * 20
    },
]

question = "What is RAG and why is it used?"

print(f"\n  Context: {context[:100]}...")
print(f"  Question: {question}\n")
print(f"  {'Label':<22} {'Faithful':>10} {'Relevant':>10} {'Concise':>10} {'Overall':>10}")
print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

for ans in test_answers:
    faith   = faithfulness_score(ans["answer"], context)
    rel     = relevance_score(question, ans["answer"])
    conc    = conciseness_score(ans["answer"])
    overall = round((faith + rel + conc) / 3, 3)
    print(f"  {ans['label']:<22} {faith:>10.3f} {rel:>10.3f} {conc:>10.3f} {overall:>10.3f}")


# -- SECTION 3: LLM-as-Judge ---------------------------------
print("\n\nSECTION 3: LLM-as-Judge Pattern")
print("-" * 40)
print("""
The most powerful evaluation method: use an LLM to score another LLM.

How it works:
  1. Generate an answer with your RAG system
  2. Send the question + context + answer to a judge LLM (GPT-4)
  3. Ask the judge to score it on specific criteria (1-10)
  4. Aggregate scores across your test set

Why it works:
  -> More nuanced than keyword matching
  -> Correlates well with human judgment
  -> Can explain WHY the score is what it is
  -> Scalable to thousands of examples

The G-Eval prompt pattern:
  "You are an expert evaluator. Score the following answer
  on a scale of 1-10 for [criterion]. 
  Question: {question}
  Context: {context}
  Answer: {answer}
  Score (1-10) and brief explanation:"
""")

def llm_judge_prompt(question, context, answer, criterion):
    """Generate a G-Eval style prompt for LLM judging."""
    return f"""You are an expert evaluator for AI systems.

Evaluate the following answer on the criterion of {criterion}.
Score from 1 to 10 where:
  1-3  = Poor (major issues)
  4-6  = Acceptable (some issues)
  7-9  = Good (minor issues)
  10   = Perfect

Question : {question}
Context  : {context[:300]}
Answer   : {answer}

Respond with ONLY a JSON object:
{{"score": <number 1-10>, "reason": "<one sentence explanation>"}}"""

# Show example prompts
test_case = {
    "question": "What is the salary range for AI engineers in India?",
    "context" : "AI engineers in India earn 8-25 LPA. Freshers start at 8-12 LPA. Senior engineers can earn 20-25 LPA.",
    "answer"  : "AI engineers in India typically earn between 8 and 25 LPA, with freshers starting around 8-12 LPA."
}

for criterion in ["faithfulness", "relevance", "completeness"]:
    prompt = llm_judge_prompt(
        test_case["question"],
        test_case["context"],
        test_case["answer"],
        criterion
    )
    print(f"\n  G-Eval prompt for [{criterion}]:")
    print(f"  {prompt[:200]}...")

# Simulate LLM judge scores (in production you'd call the API)
simulated_scores = {
    "faithfulness" : {"score": 9, "reason": "Answer is well grounded in the provided context"},
    "relevance"    : {"score": 10, "reason": "Directly answers the salary question"},
    "completeness" : {"score": 7, "reason": "Could mention senior engineer salaries explicitly"}
}

print(f"\n  Simulated LLM Judge Scores:")
print(f"  {'Criterion':<20} {'Score':>7} {'Reason'}")
print(f"  {'─'*20} {'─'*7} {'─'*40}")
for criterion, result in simulated_scores.items():
    print(f"  {criterion:<20} {result['score']:>7}/10  {result['reason']}")

avg_score = sum(r["score"] for r in simulated_scores.values()) / len(simulated_scores)
print(f"\n  Average score: {avg_score:.1f}/10")


# -- SECTION 4: Build Complete Eval Pipeline -----------------
print("\n\nSECTION 4: Complete Evaluation Pipeline")
print("-" * 40)

class RAGEvaluator:
    """Complete RAG evaluation pipeline."""

    def __init__(self, name):
        self.name    = name
        self.results = []

    def evaluate_example(self, question, context, answer, reference=None):
        """Evaluate a single RAG output."""
        scores = {
            "faithfulness": faithfulness_score(answer, context),
            "relevance"   : relevance_score(question, answer),
            "conciseness" : conciseness_score(answer),
        }

        if reference:
            scores["f1_vs_reference"]   = f1_token_score(answer, reference)
            scores["rouge_vs_reference"]= rouge_l(answer, reference)

        scores["overall"] = round(sum(scores.values()) / len(scores), 3)
        self.results.append({"question": question, **scores})
        return scores

    def summary(self):
        """Print evaluation summary."""
        if not self.results:
            print("No results yet.")
            return

        print(f"\n  Evaluation Summary: {self.name}")
        print(f"  Total examples: {len(self.results)}\n")

        metrics = [k for k in self.results[0].keys() if k != "question"]
        for metric in metrics:
            values = [r[metric] for r in self.results]
            avg    = sum(values) / len(values)
            mn     = min(values)
            mx     = max(values)
            bar    = "█" * int(avg * 20)
            print(f"  {metric:<25}: avg={avg:.3f} min={mn:.3f} max={mx:.3f}  {bar}")

    def save(self, filepath):
        """Save results to JSON."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"evaluator": self.name, "results": self.results}, f, indent=2)
        print(f"\n  Results saved to {filepath}")

# Run evaluation
evaluator = RAGEvaluator("Day8_RAG_System_v1")

test_examples = [
    {
        "question" : "What is RAG and why is it important?",
        "context"  : "RAG stands for Retrieval Augmented Generation. It reduces LLM hallucinations by grounding responses in retrieved documents. Essential for enterprise AI applications.",
        "answer"   : "RAG is Retrieval Augmented Generation, which reduces hallucinations by grounding LLM answers in retrieved documents. It is essential for enterprise AI.",
        "reference": "RAG reduces hallucinations by combining retrieval with generation."
    },
    {
        "question" : "How much do AI engineers earn in India?",
        "context"  : "AI engineers in India earn 8-25 LPA. Hyderabad and Bangalore are top cities. Freshers start at 8-12 LPA.",
        "answer"   : "AI engineers earn 8-25 LPA in India. Fresh graduates typically start at 8-12 LPA in cities like Hyderabad and Bangalore.",
        "reference": "AI engineers in India earn 8-25 LPA."
    },
    {
        "question" : "What is Docker used for?",
        "context"  : "Docker containerizes applications so they run consistently across environments. Used for deployment and DevOps.",
        "answer"   : "Docker is a tool for baking cakes and managing restaurant orders.",  # bad answer
        "reference": "Docker containerizes applications for consistent deployment."
    },
]

print("  Running evaluation pipeline...\n")
for ex in test_examples:
    scores = evaluator.evaluate_example(
        ex["question"],
        ex["context"],
        ex["answer"],
        ex.get("reference")
    )

evaluator.summary()
evaluator.save("rag_evaluation_results.json")

print()
print("=" * 60)
print("Script 2 complete! LLM Evaluation covered.")
print("Key concepts:")
print("  [OK] Reference-based metrics (BLEU, ROUGE, F1, EM)")
print("  [OK] Reference-free metrics (faithfulness, relevance)")
print("  [OK] LLM-as-judge pattern (G-Eval)")
print("  [OK] Complete evaluation pipeline")
print("  [OK] Saving evaluation results")
print("=" * 60)
