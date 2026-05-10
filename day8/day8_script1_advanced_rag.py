# -*- coding: utf-8 -*-
# ============================================================
# DAY 8 -- SCRIPT 1: Advanced RAG Techniques
# Topics: multi-document RAG, hybrid search, re-ranking,
#         metadata filtering, RAG evaluation, chunking strategies
# Week 2 -- Building Real Projects
# ============================================================

import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("DAY 8 -- SCRIPT 1: Advanced RAG Techniques")
print("Week 2 -- Building Real Projects")
print("=" * 60)

print("""
WEEK 2 MINDSET SHIFT
---------------------
Week 1 = learning tools one by one
Week 2 = combining tools into REAL projects people can use

Day 8 focus: Making your RAG system actually GOOD.

Basic RAG (what you built on Day 4):
  User asks question
  -> search vector DB
  -> return top 3 chunks
  -> send to GPT
  -> return answer

Problems with basic RAG:
  -> Returns wrong chunks sometimes (keyword mismatch)
  -> No way to filter by date, topic, source
  -> Cannot handle multiple document types
  -> No way to know if the answer is actually good
  -> Retrieves similar-looking but irrelevant chunks

Advanced RAG fixes all of this:
  -> Hybrid search (keyword + semantic combined)
  -> Metadata filtering (only search recent docs)
  -> Re-ranking (reorder results by true relevance)
  -> Multi-document with source tracking
  -> Self-evaluation (model checks its own answer)
""")


# -- SECTION 1: Chunking Strategies --------------------------
print("SECTION 1: Chunking Strategies -- Size Matters")
print("-" * 40)
print("""
Chunk size is the most important RAG tuning parameter.

Too small (50 chars):  loses context, fragments ideas
Too large (2000 chars): too much noise, retrieval is less precise
Just right (300-500 chars): preserves context, precise retrieval

Different strategies:
  Fixed size    -> split every N characters (simple, not smart)
  Recursive     -> split on paragraphs, then sentences, then words
  Semantic      -> split when topic changes (best quality, slower)
  Sentence      -> split on sentences (good for Q&A)
  Sliding window-> overlapping chunks (preserves context at boundaries)
""")

sample_document = """
Artificial Intelligence (AI) is transforming every industry in 2025.
Machine learning models can now understand text, images, and audio with
superhuman accuracy. Companies are racing to integrate AI into their products.

Large Language Models (LLMs) like GPT-4 and Claude represent the cutting
edge of AI capabilities. These models are trained on trillions of tokens
of text and can perform complex reasoning tasks. Fine-tuning these models
on domain-specific data allows companies to create specialized AI assistants.

Retrieval Augmented Generation (RAG) is the most popular architecture for
enterprise AI applications. RAG combines the broad knowledge of LLMs with
specific company documents, reducing hallucinations significantly.
Vector databases like ChromaDB and Pinecone store document embeddings
for fast semantic search.

The job market for AI engineers is booming. Companies in Hyderabad,
Bangalore, and Mumbai are actively hiring freshers with Python, LangChain,
and RAG skills. Salaries for entry-level AI engineers range from 8 to 15 LPA.
Building and deploying real projects is the fastest way to get hired.
"""

def fixed_size_chunking(text, chunk_size=200, overlap=50):
    """Split text into fixed-size chunks with overlap."""
    chunks = []
    start  = 0
    while start < len(text):
        end   = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks

def sentence_chunking(text):
    """Split text into sentence-based chunks."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    chunks    = []
    current   = ""
    for sentence in sentences:
        if len(current) + len(sentence) < 300:
            current += sentence + ". "
        else:
            if current:
                chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks

def paragraph_chunking(text):
    """Split text into paragraph-based chunks."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs

# Compare strategies
print("\n  Comparing chunking strategies on sample document:")
print(f"  Document length: {len(sample_document)} characters\n")

strategies = [
    ("Fixed size (200 chars)", fixed_size_chunking(sample_document)),
    ("Sentence-based",         sentence_chunking(sample_document)),
    ("Paragraph-based",        paragraph_chunking(sample_document)),
]

for name, chunks in strategies:
    avg_len = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
    print(f"  Strategy: {name}")
    print(f"  Chunks  : {len(chunks)}")
    print(f"  Avg size: {avg_len:.0f} chars")
    print(f"  Sample  : {chunks[0][:80]}...")
    print()


# -- SECTION 2: Multi-Document RAG with Metadata -------------
print("\nSECTION 2: Multi-Document RAG with Metadata Filtering")
print("-" * 40)
print("""
Real RAG systems handle multiple documents with metadata.
Metadata lets you filter BEFORE searching:
  "Only search documents from last 30 days"
  "Only search from the engineering department"
  "Only search Python tutorials, not JavaScript"
""")

# Create a rich knowledge base with metadata
knowledge_base = [
    {
        "id"      : "doc_001",
        "title"   : "Python for AI Engineers",
        "content" : "Python is essential for AI development. Key libraries include NumPy, Pandas, scikit-learn, TensorFlow, and LangChain. Python's simple syntax makes it the top choice for ML engineers.",
        "metadata": {"category": "programming", "level": "beginner", "date": "2025-01-15", "source": "tutorial"}
    },
    {
        "id"      : "doc_002",
        "title"   : "LangChain RAG Tutorial",
        "content" : "RAG combines retrieval with generation. Use ChromaDB for local vector storage. Split documents into 300-500 char chunks with 50 char overlap. Use HuggingFace embeddings for free semantic search.",
        "metadata": {"category": "llm", "level": "intermediate", "date": "2025-02-20", "source": "tutorial"}
    },
    {
        "id"      : "doc_003",
        "title"   : "AI Jobs in India 2025",
        "content" : "AI engineer salaries in India range from 8-25 LPA. Hyderabad and Bangalore are top cities. Companies like TCS, Infosys, startups like Sarvam AI, and product companies are all hiring. Freshers need Python, LLMs, and deployed projects.",
        "metadata": {"category": "career", "level": "beginner", "date": "2025-03-10", "source": "article"}
    },
    {
        "id"      : "doc_004",
        "title"   : "Fine-tuning vs RAG Decision Guide",
        "content" : "Use RAG when data changes frequently or you need citations. Use fine-tuning when you need specific tone or consistent classification. RAG is cheaper and faster to implement for most enterprise use cases.",
        "metadata": {"category": "llm", "level": "advanced", "date": "2025-03-25", "source": "guide"}
    },
    {
        "id"      : "doc_005",
        "title"   : "FastAPI for ML Deployment",
        "content" : "FastAPI is the best framework for serving ML models. It provides automatic Swagger docs, Pydantic validation, async support, and is 300% faster than Flask. Use uvicorn as the ASGI server.",
        "metadata": {"category": "deployment", "level": "intermediate", "date": "2025-04-01", "source": "tutorial"}
    },
    {
        "id"      : "doc_006",
        "title"   : "Docker for AI Engineers",
        "content" : "Docker containerizes your AI app so it runs identically everywhere. Write a Dockerfile, build an image, run a container. Use docker-compose for multi-service apps. Deploy to Render or Railway for free.",
        "metadata": {"category": "deployment", "level": "intermediate", "date": "2025-04-05", "source": "tutorial"}
    },
    {
        "id"      : "doc_007",
        "title"   : "Prompt Engineering Patterns",
        "content" : "Key prompt patterns: zero-shot, few-shot, chain-of-thought, role prompting, output anchoring. Temperature 0 for factual tasks, 0.7-1.0 for creative tasks. System prompts define AI behavior. Good prompts = better results without fine-tuning.",
        "metadata": {"category": "llm", "level": "beginner", "date": "2025-01-30", "source": "guide"}
    },
    {
        "id"      : "doc_008",
        "title"   : "Vector Databases Compared",
        "content" : "ChromaDB: free, local, perfect for learning and small projects. Pinecone: cloud-managed, scales to billions of vectors, has free tier. FAISS: Facebook's library, fastest for CPU search, no built-in persistence. Weaviate: open source, good GraphQL support.",
        "metadata": {"category": "infrastructure", "level": "intermediate", "date": "2025-02-10", "source": "comparison"}
    },
]

print(f"  Knowledge base: {len(knowledge_base)} documents")
print(f"\n  {'ID':<10} {'Title':<35} {'Category':<15} {'Level':<12} {'Date'}")
print(f"  {'─'*10} {'─'*35} {'─'*15} {'─'*12} {'─'*12}")
for doc in knowledge_base:
    m = doc["metadata"]
    print(f"  {doc['id']:<10} {doc['title'][:33]:<35} {m['category']:<15} {m['level']:<12} {m['date']}")


# -- SECTION 3: Simple Embedding Function --------------------
print("\n\nSECTION 3: Building Embeddings Without API")
print("-" * 40)

def simple_embed(text, dim=50):
    """
    Simple character-frequency embedding for demo.
    In production use: HuggingFace sentence-transformers
    or OpenAI text-embedding-3-small
    """
    text  = text.lower()
    vec   = np.zeros(dim)
    words = text.split()

    for i, word in enumerate(words[:dim]):
        for char in word:
            if char.isalpha():
                idx      = (ord(char) - ord('a') + i) % dim
                vec[idx] += 1

    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

# Build index
print("  Building document index...")
for doc in knowledge_base:
    doc["embedding"] = simple_embed(doc["title"] + " " + doc["content"])
print(f"  Indexed {len(knowledge_base)} documents")


# -- SECTION 4: Hybrid Search --------------------------------
print("\n\nSECTION 4: Hybrid Search -- Keyword + Semantic")
print("-" * 40)
print("""
Basic RAG uses only semantic search (vector similarity).
Problem: "What is FAISS?" -> semantic search might miss it
         because FAISS is a specific keyword.

Hybrid search combines:
  Semantic search  -> finds conceptually similar docs (good for meaning)
  Keyword search   -> finds exact term matches (good for names, acronyms)
  
Final score = alpha * semantic_score + (1-alpha) * keyword_score
alpha = 0.7 means 70% semantic, 30% keyword
""")

def keyword_search(query, docs, top_k=3):
    """BM25-style keyword matching."""
    query_words = set(query.lower().split())
    scores      = []
    for doc in docs:
        content_words = set((doc["title"] + " " + doc["content"]).lower().split())
        overlap       = len(query_words.intersection(content_words))
        score         = overlap / (len(query_words) + 1)
        scores.append((score, doc))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]

def semantic_search(query, docs, top_k=3):
    """Vector similarity search."""
    query_emb = simple_embed(query)
    scores    = []
    for doc in docs:
        score = cosine_similarity(query_emb, doc["embedding"])
        scores.append((score, doc))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]

def hybrid_search(query, docs, top_k=3, alpha=0.7):
    """Combine semantic and keyword search."""
    query_emb    = simple_embed(query)
    query_words  = set(query.lower().split())
    scores       = []

    for doc in docs:
        # Semantic score
        sem_score = cosine_similarity(query_emb, doc["embedding"])

        # Keyword score
        content_words = set((doc["title"] + " " + doc["content"]).lower().split())
        overlap       = len(query_words.intersection(content_words))
        kw_score      = overlap / (len(query_words) + 1)

        # Combined score
        final_score = alpha * sem_score + (1 - alpha) * kw_score
        scores.append((final_score, sem_score, kw_score, doc))

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]

def metadata_filter(docs, category=None, level=None, after_date=None):
    """Filter documents by metadata before searching."""
    filtered = docs
    if category:
        filtered = [d for d in filtered if d["metadata"]["category"] == category]
    if level:
        filtered = [d for d in filtered if d["metadata"]["level"] == level]
    if after_date:
        filtered = [d for d in filtered if d["metadata"]["date"] >= after_date]
    return filtered

# Test different search methods
test_queries = [
    ("How do I deploy my ML model?",       None,      None),
    ("What salary can I earn as fresher?", "career",  None),
    ("What is FAISS vector database?",     None,      None),
    ("LLM techniques for beginners",       "llm",     "beginner"),
]

for query, cat_filter, level_filter in test_queries:
    print(f"\n  Query: '{query}'")
    if cat_filter or level_filter:
        print(f"  Filter: category={cat_filter}, level={level_filter}")

    # Apply metadata filter
    filtered_docs = metadata_filter(
        knowledge_base,
        category = cat_filter,
        level    = level_filter
    )

    # Hybrid search
    results = hybrid_search(query, filtered_docs, top_k=2)

    print(f"  {'Rank':<5} {'Score':>7} {'Sem':>7} {'KW':>7} {'Document'}")
    print(f"  {'─'*5} {'─'*7} {'─'*7} {'─'*7} {'─'*35}")
    for rank, (score, sem, kw, doc) in enumerate(results, 1):
        print(f"  {rank:<5} {score:>7.3f} {sem:>7.3f} {kw:>7.3f} {doc['title'][:35]}")


# -- SECTION 5: Re-ranking -----------------------------------
print("\n\nSECTION 5: Re-ranking -- Get Better Results")
print("-" * 40)
print("""
Re-ranking = after getting top 10 results, reorder them
             using a more accurate (but slower) model.

Two-stage retrieval:
  Stage 1: Fast vector search -> top 10 candidates (fast, cheap)
  Stage 2: Re-ranker model   -> reorder top 10 (slower, more accurate)

Why? Vector search is fast but not perfectly accurate.
Re-ranker compares query+document together (cross-encoder)
which is much more accurate but too slow to run on all docs.

Popular re-rankers:
  cross-encoder/ms-marco-MiniLM-L-6-v2  (HuggingFace, free)
  Cohere Rerank API
  Jina Reranker
""")

def simple_reranker(query, candidates):
    """
    Simple re-ranker based on query-document overlap scoring.
    In production use: cross-encoder from HuggingFace
    """
    query_words = set(query.lower().split())
    reranked    = []

    for score, doc in candidates:
        content = (doc["title"] + " " + doc["content"]).lower()

        # Count exact phrase matches (more accurate than word overlap)
        exact_matches = sum(1 for word in query_words if word in content)

        # Boost score if title matches query
        title_boost = 0.2 if any(w in doc["title"].lower() for w in query_words) else 0

        # Position of first keyword match (earlier = more relevant)
        first_pos = min(
            (content.find(w) for w in query_words if w in content),
            default = len(content)
        )
        position_score = 1.0 - (first_pos / len(content))

        rerank_score = (
            score * 0.4 +
            (exact_matches / (len(query_words) + 1)) * 0.4 +
            title_boost +
            position_score * 0.1
        )
        reranked.append((rerank_score, doc))

    reranked.sort(key=lambda x: x[0], reverse=True)
    return reranked

# Demo re-ranking
query     = "how to deploy FastAPI ML model with Docker"
initial   = semantic_search(query, knowledge_base, top_k=5)
reranked  = simple_reranker(query, initial)

print(f"\n  Query: '{query}'\n")
print(f"  Before re-ranking (semantic only):")
for i, (score, doc) in enumerate(initial, 1):
    print(f"    {i}. [{score:.3f}] {doc['title']}")

print(f"\n  After re-ranking:")
for i, (score, doc) in enumerate(reranked, 1):
    print(f"    {i}. [{score:.3f}] {doc['title']}")


# -- SECTION 6: RAG Self-Evaluation --------------------------
print("\n\nSECTION 6: RAG Evaluation -- Is Your Answer Good?")
print("-" * 40)
print("""
How do you know if your RAG system is working well?
You need to MEASURE it.

Key RAG metrics:
  Faithfulness    -> Is the answer based on retrieved context? (no hallucination)
  Answer Relevance-> Does the answer actually address the question?
  Context Recall  -> Did we retrieve the RIGHT documents?
  Context Precision-> Are the retrieved docs actually relevant?

Simple evaluation approach:
  1. Create a test set: (question, expected_answer, source_doc) pairs
  2. Run your RAG on each question
  3. Compare generated answer with expected answer
  4. Score each metric
""")

# Create evaluation dataset
eval_dataset = [
    {
        "question"       : "What salary can a fresher AI engineer expect?",
        "expected_answer": "8-15 LPA",
        "source_doc"     : "doc_003",
        "keywords"       : ["salary", "lpa", "fresher", "8", "15"]
    },
    {
        "question"       : "What is ChromaDB used for?",
        "expected_answer": "local vector storage for RAG",
        "source_doc"     : "doc_008",
        "keywords"       : ["chromadb", "vector", "local", "free"]
    },
    {
        "question"       : "When should I use RAG instead of fine-tuning?",
        "expected_answer": "when data changes frequently or you need citations",
        "source_doc"     : "doc_004",
        "keywords"       : ["rag", "fine-tuning", "data", "changes", "citations"]
    },
]

def evaluate_rag(eval_set, docs):
    """Simple RAG evaluation."""
    results = []

    for item in eval_set:
        # Retrieve docs
        retrieved = hybrid_search(item["question"], docs, top_k=3)
        top_doc   = retrieved[0][3] if retrieved else None

        # Check context recall (did we get the right document?)
        correct_doc_retrieved = any(
            r[3]["id"] == item["source_doc"] for r in retrieved
        )

        # Simulate answer from retrieved context
        simulated_answer = top_doc["content"][:200] if top_doc else ""

        # Check faithfulness (are answer keywords in retrieved context?)
        answer_in_context = sum(
            1 for kw in item["keywords"]
            if kw.lower() in simulated_answer.lower()
        ) / len(item["keywords"])

        results.append({
            "question"            : item["question"][:50],
            "correct_doc_found"   : correct_doc_retrieved,
            "faithfulness_score"  : round(answer_in_context, 2),
            "top_retrieved_doc"   : top_doc["title"] if top_doc else "None",
            "expected_source"     : item["source_doc"]
        })

    return results

eval_results = evaluate_rag(eval_dataset, knowledge_base)

print("\n  RAG Evaluation Results:")
print(f"\n  {'Question':<45} {'Correct Doc':>12} {'Faithful':>10}")
print(f"  {'─'*45} {'─'*12} {'─'*10}")
for r in eval_results:
    found = "YES" if r["correct_doc_found"] else "NO"
    print(f"  {r['question']:<45} {found:>12} {r['faithfulness_score']:>10.0%}")

avg_faithfulness = sum(r["faithfulness_score"] for r in eval_results) / len(eval_results)
correct_retrieval = sum(1 for r in eval_results if r["correct_doc_found"])
print(f"\n  Overall Retrieval Accuracy: {correct_retrieval}/{len(eval_results)}")
print(f"  Average Faithfulness      : {avg_faithfulness:.0%}")


# -- SECTION 7: Save Advanced RAG Config ---------------------
print("\n\nSECTION 7: Saving Advanced RAG Configuration")
print("-" * 40)

rag_config = {
    "chunking": {
        "strategy"  : "recursive",
        "chunk_size": 400,
        "overlap"   : 50
    },
    "retrieval": {
        "method"         : "hybrid",
        "alpha"          : 0.7,
        "top_k_retrieval": 10,
        "top_k_rerank"   : 3
    },
    "embedding": {
        "model"    : "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "device"   : "cpu"
    },
    "reranking": {
        "enabled": True,
        "model"  : "cross-encoder/ms-marco-MiniLM-L-6-v2"
    },
    "evaluation": {
        "metrics": ["faithfulness", "context_recall", "answer_relevance"]
    }
}

with open("advanced_rag_config.json", "w", encoding="utf-8") as f:
    json.dump(rag_config, f, indent=2)

print("  advanced_rag_config.json saved!")
print("  Use this config as a template for your production RAG systems.")

print()
print("=" * 60)
print("Script 1 complete! Advanced RAG covered.")
print("Key concepts:")
print("  [OK] Chunking strategies compared")
print("  [OK] Multi-document RAG with metadata")
print("  [OK] Hybrid search (semantic + keyword)")
print("  [OK] Metadata filtering")
print("  [OK] Re-ranking for better accuracy")
print("  [OK] RAG evaluation metrics")
print("=" * 60)
