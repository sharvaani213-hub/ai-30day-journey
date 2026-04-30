# ============================================================
# DAY 3 — SCRIPT 2: Transformers & HuggingFace
# Topics: transformer intuition, attention mechanism,
#         HuggingFace pipeline, tokenizers, BERT, GPT-2
# ============================================================

import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("DAY 3 — SCRIPT 2: Transformers & HuggingFace")
print("=" * 60)

# ── SECTION 1: Why Transformers? ──────────────────────────────
print("""
SECTION 1: Why Transformers Changed Everything
─────────────────────────────────────────────
Before Transformers (2017), we used RNNs/LSTMs which:
  ✗ Processed words one by one (slow)
  ✗ Forgot earlier words in long sentences
  ✗ Couldn't be parallelized (couldn't use GPUs fully)

Transformers introduced ATTENTION — reads ALL words at once
and learns which words to "pay attention to" for each word.

Example:
  "The cat sat on the mat because IT was tired"
  → What does "IT" refer to? The cat!
  → Attention mechanism figures this out automatically.

GPT, BERT, Claude, Gemini — ALL are transformers.
""")


# ── SECTION 2: Attention Mechanism (Simplified) ───────────────
print("SECTION 2: Self-Attention — Manual Intuition")
print("-" * 40)
print("Attention score tells each word how much to focus on other words\n")

# Simplified attention for 4 words: "I love deep learning"
words = ["I", "love", "deep", "learning"]
n = len(words)

np.random.seed(7)

# In real transformers these come from learned weight matrices
# Here we simulate small vectors for each word
Q = np.random.randn(n, 3)   # Query  — "what am I looking for?"
K = np.random.randn(n, 3)   # Key    — "what do I have?"
V = np.random.randn(n, 3)   # Value  — "what do I give?"

# Attention scores = Q @ K^T  (how much each word attends to others)
d_k    = Q.shape[1]
scores = (Q @ K.T) / np.sqrt(d_k)   # scale by sqrt(d_k) for stability

# Softmax to get probabilities
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

attention_weights = softmax(scores)

print("Attention weights (how much each word attends to others):")
print(f"{'':10}", end="")
for w in words:
    print(f"{w:>10}", end="")
print()
for i, word in enumerate(words):
    print(f"{word:<10}", end="")
    for j in range(n):
        print(f"{attention_weights[i,j]:>10.3f}", end="")
    print()

# Output = weighted sum of Values
output = attention_weights @ V
print(f"\nOutput shape: {output.shape} (each word now has context from all others)")
print("\nThis is Self-Attention. Transformers stack 8-16 of these in parallel (Multi-Head Attention)")


# ── SECTION 3: Transformer Architecture Overview ──────────────
print("\n\nSECTION 3: Transformer Architecture")
print("-" * 40)
print("""
Full Transformer (simplified):

  INPUT TEXT
      ↓
  [Tokenizer]          "Hello world" → [15496, 995]
      ↓
  [Embedding Layer]    token IDs → dense vectors (e.g. 768 dims)
      ↓
  [Positional Encoding] adds position info (word order)
      ↓
  [Multi-Head Attention] × N layers   ← THE KEY INNOVATION
      ↓
  [Feed Forward Network] × N layers
      ↓
  [Layer Norm + Residual connections]
      ↓
  OUTPUT (logits over vocabulary)

BERT  = Encoder only  → great for classification, understanding
GPT   = Decoder only  → great for text generation
T5    = Encoder+Decoder → great for translation, summarization
""")


# ── SECTION 4: HuggingFace Pipelines ─────────────────────────
print("\nSECTION 4: HuggingFace Pipelines — Use LLMs in 3 lines")
print("-" * 40)
print("HuggingFace = largest open-source AI model hub")
print("Pipeline = download a model + run inference in 3 lines of code\n")

try:
    from transformers import pipeline

    # ── Task 1: Sentiment Analysis ────────────────────────────
    print("Task 1: Sentiment Analysis")
    print("  (First run downloads model ~67MB — takes 1-2 minutes)\n")

    sentiment = pipeline("sentiment-analysis",
                         model="distilbert-base-uncased-finetuned-sst-2-english")

    texts = [
        "I love Python and machine learning!",
        "This code has so many bugs, I hate it.",
        "The model accuracy is decent but could be better.",
        "Getting placed at a top AI company would be amazing!",
        "I failed my interview. I'm really disappointed."
    ]

    print(f"  {'Text':<50} {'Label':>10} {'Score':>8}")
    print(f"  {'─'*50} {'─'*10} {'─'*8}")
    results = sentiment(texts)
    for text, result in zip(texts, results):
        short = text[:48] + ".." if len(text) > 48 else text
        print(f"  {short:<50} {result['label']:>10} {result['score']:>7.2%}")

    # ── Task 2: Text Generation (GPT-2) ──────────────────────
    print("\n\nTask 2: Text Generation with GPT-2")
    print("  GPT-2 = older version of ChatGPT — still great for learning\n")

    generator = pipeline("text-generation", model="gpt2", max_new_tokens=60)

    prompts = [
        "Machine learning is important because",
        "To become an AI engineer, you need to",
    ]

    for prompt in prompts:
        print(f"  Prompt   : {prompt}")
        result = generator(prompt, num_return_sequences=1, do_sample=True,
                           temperature=0.7, pad_token_id=50256)
        generated = result[0]["generated_text"]
        print(f"  Generated: {generated[:200]}")
        print()

    # ── Task 3: Zero-shot Classification ─────────────────────
    print("\nTask 3: Zero-Shot Classification")
    print("  Classify text into categories WITHOUT training on them!\n")

    classifier = pipeline("zero-shot-classification",
                           model="facebook/bart-large-mnli")

    text   = "I want to learn how to build a chatbot using Python and OpenAI API"
    labels = ["programming", "cooking", "sports", "technology", "finance"]

    result = classifier(text, candidate_labels=labels)
    print(f"  Text  : {text}")
    print(f"  Labels: {labels}")
    print(f"\n  Results:")
    for label, score in zip(result["labels"], result["scores"]):
        bar = "█" * int(score * 30)
        print(f"    {label:<15}: {score:.2%}  {bar}")

    # ── Task 4: Named Entity Recognition ─────────────────────
    print("\n\nTask 4: Named Entity Recognition (NER)")
    print("  Extracts people, places, organizations from text\n")

    ner = pipeline("ner", grouped_entities=True)
    text = "Elon Musk founded Tesla and SpaceX in California. Sundar Pichai leads Google in Mountain View."

    entities = ner(text)
    print(f"  Text: {text}\n")
    print(f"  {'Entity':<25} {'Type':<15} {'Score':>8}")
    print(f"  {'─'*25} {'─'*15} {'─'*8}")
    for ent in entities:
        print(f"  {ent['word']:<25} {ent['entity_group']:<15} {ent['score']:>7.2%}")

    # ── Task 5: Question Answering ────────────────────────────
    print("\n\nTask 5: Question Answering")
    print("  Give it a passage + question → it finds the answer\n")

    qa = pipeline("question-answering",
                  model="distilbert-base-cased-distilled-squad")

    context = """
    Python is a high-level programming language created by Guido van Rossum in 1991.
    It is widely used in artificial intelligence, machine learning, web development,
    and data science. Python's simple syntax makes it ideal for beginners.
    TensorFlow and PyTorch are the two most popular deep learning frameworks for Python.
    """

    questions = [
        "Who created Python?",
        "When was Python created?",
        "What are the popular deep learning frameworks?",
        "Why is Python ideal for beginners?"
    ]

    for q in questions:
        result = qa(question=q, context=context)
        print(f"  Q: {q}")
        print(f"  A: {result['answer']}  (confidence: {result['score']:.2%})\n")

except ImportError:
    print("\n  transformers not installed!")
    print("  Run: pip install transformers torch")
    print("\n  Showing what the output WOULD look like:\n")
    print("  Sentiment: 'I love Python!' → POSITIVE (99.8%)")
    print("  GPT-2 generation: 'ML is important because it enables...'")
    print("  Zero-shot: 'Build a chatbot' → technology (94.2%)")
    print("  NER: 'Elon Musk' → PERSON, 'Tesla' → ORG, 'California' → LOC")

except Exception as e:
    print(f"\n  Note: {e}")
    print("  This is normal if running offline or models are downloading.")


# ── SECTION 5: Tokenizers ─────────────────────────────────────
print("\n\nSECTION 5: Tokenizers — How Text Becomes Numbers")
print("-" * 40)
print("""
LLMs don't read text — they read TOKEN IDs (numbers).

"Hello, how are you?" → [15496, 11, 703, 389, 345, 30]

Each token = a word or subword piece.
"unhappiness" → ["un", "happiness"] → [2, IDs]

WHY subwords? Handles rare words + new words gracefully.
""")

try:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    texts = [
        "Hello world",
        "Machine learning is fascinating",
        "I am learning AI to become an AI Engineer",
        "Hyderabad is a great city for tech jobs"
    ]

    print(f"  {'Text':<45} {'Tokens':>7} {'Token IDs'}")
    print(f"  {'─'*45} {'─'*7} {'─'*30}")
    for text in texts:
        tokens   = tokenizer.tokenize(text)
        token_ids= tokenizer.encode(text)
        print(f"  {text:<45} {len(tokens):>7} {token_ids[:6]}...")

    print("\nTokenizer encodes and decodes:")
    sample = "I want to get placed as an AI Engineer"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    print(f"  Original : {sample}")
    print(f"  Encoded  : {encoded}")
    print(f"  Decoded  : {decoded}")

except Exception as e:
    print(f"  (Install transformers to see tokenizer demo)")
    print(f"  pip install transformers")

print()
print("=" * 60)
print("Script 2 complete! Transformers & HuggingFace covered.")
print("Key concepts covered:")
print("  ✓ Why transformers replaced RNNs")
print("  ✓ Self-attention mechanism (manual)")
print("  ✓ Transformer architecture overview")
print("  ✓ HuggingFace pipelines (5 NLP tasks)")
print("  ✓ Tokenizers — how text becomes numbers")
print("=" * 60)
