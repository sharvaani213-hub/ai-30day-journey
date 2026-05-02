# ============================================================
# DAY 4 — SCRIPT 2: RAG Pipeline — Chat with Your Documents
# Topics: embeddings, vector databases, ChromaDB, FAISS,
#         retrieval, RAG chain, document Q&A
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("DAY 4 — SCRIPT 2: RAG Pipeline")
print("Chat with Your Documents using LangChain + ChromaDB")
print("=" * 60)

print("""
WHAT IS RAG?
────────────
RAG = Retrieval Augmented Generation

Problem: LLMs have a knowledge cutoff. They don't know YOUR documents.
Solution: RAG = Search relevant document chunks → Feed to LLM → Get answer

How it works:
  1. INDEXING (done once):
     → Split documents into chunks
     → Convert chunks to vectors (embeddings)
     → Store vectors in a vector database

  2. QUERYING (done every time):
     → Convert user question to a vector
     → Find most similar chunks in the database
     → Send chunks + question to LLM
     → LLM answers based on your documents

This is how ChatPDF, NotebookLM, and most enterprise AI apps work.
""")


# ── Check installations ───────────────────────────────────────
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
    print("✓ LangChain + ChromaDB available!")
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"Some packages missing: {e}")
    print("Run: pip install langchain langchain-community chromadb sentence-transformers")

API_KEY       = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
API_AVAILABLE = API_KEY != "your-api-key-here" and LANGCHAIN_AVAILABLE


# ── SECTION 1: What are Embeddings? ──────────────────────────
print("\n\nSECTION 1: Embeddings — Words as Vectors")
print("-" * 40)
print("""
Embeddings convert text into numbers (vectors) that capture MEANING.

"king" - "man" + "woman" ≈ "queen"  ← embeddings understand relationships!

Similar meanings → similar vectors → close together in vector space.

Example:
  "Python is great for AI"     → [0.23, -0.41, 0.88, ...]  (768 numbers)
  "Python works well for ML"   → [0.21, -0.39, 0.85, ...]  (very similar!)
  "I love eating pizza"        → [-0.92, 0.15, -0.33, ...]  (very different)

Vector databases store millions of these and find similar ones instantly.
""")

# Manual cosine similarity demo (no API needed)
import numpy as np

def cosine_similarity(v1, v2):
    """Measure similarity between two vectors (1.0 = identical, 0.0 = opposite)"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Simulated embeddings (in real life these come from a model)
np.random.seed(42)
base = np.random.randn(10)

# Similar sentences have similar vectors
python_ai   = base + np.random.randn(10) * 0.1
python_ml   = base + np.random.randn(10) * 0.1
eating_pizza= np.random.randn(10)         # completely different topic

sim1 = cosine_similarity(python_ai, python_ml)
sim2 = cosine_similarity(python_ai, eating_pizza)

print("Cosine similarity demo:")
print(f"  'Python for AI' vs 'Python for ML'  : {sim1:.3f}  (HIGH — similar topic)")
print(f"  'Python for AI' vs 'I love pizza'   : {sim2:.3f}  (LOW — different topic)")
print("\nThis is how vector search finds relevant documents!")


# ── SECTION 2: Create Sample Documents ───────────────────────
print("\n\nSECTION 2: Creating Sample Documents")
print("-" * 40)

# We'll create a mini "knowledge base" about AI engineering
# In a real app, these would be PDFs, web pages, your notes, etc.

documents_text = [
    {
        "title": "Python for AI",
        "content": """
        Python is the most popular programming language for artificial intelligence
        and machine learning. Key libraries include NumPy for numerical computing,
        Pandas for data manipulation, scikit-learn for traditional ML algorithms,
        TensorFlow and PyTorch for deep learning, and LangChain for LLM applications.
        Python's simple syntax makes it ideal for rapid prototyping of AI systems.
        Every AI engineer must be proficient in Python.
        """
    },
    {
        "title": "LLMs and Transformers",
        "content": """
        Large Language Models (LLMs) like GPT-4, Claude, and Gemini are built on
        the transformer architecture introduced in 2017. The key innovation is the
        attention mechanism which allows models to focus on relevant parts of text.
        BERT is an encoder-only transformer good for understanding tasks.
        GPT models are decoder-only transformers good for text generation.
        LLMs are trained on massive text datasets using self-supervised learning.
        Fine-tuning adapts pre-trained LLMs to specific tasks.
        """
    },
    {
        "title": "RAG Systems",
        "content": """
        Retrieval Augmented Generation (RAG) combines information retrieval with
        text generation. RAG systems first retrieve relevant documents from a
        knowledge base using semantic search, then use an LLM to generate answers
        based on the retrieved context. This approach reduces hallucinations and
        allows LLMs to access up-to-date information. Vector databases like
        ChromaDB, Pinecone, and FAISS store document embeddings for fast retrieval.
        RAG is widely used in enterprise AI applications and chatbots.
        """
    },
    {
        "title": "AI Engineer Career",
        "content": """
        An AI engineer builds and deploys AI-powered applications. Key skills include
        Python programming, machine learning fundamentals, working with LLM APIs,
        building RAG pipelines, prompt engineering, and basic MLOps.
        In India, AI engineers earn between 8-25 LPA depending on experience.
        Freshers with strong project portfolios can expect 8-12 LPA.
        Companies hiring AI engineers include startups, product companies, and
        IT services firms. Hyderabad and Bangalore are the top cities for AI jobs.
        Having 3-5 deployed projects on GitHub significantly improves job prospects.
        """
    },
    {
        "title": "Prompt Engineering",
        "content": """
        Prompt engineering is the practice of designing effective inputs for LLMs
        to get desired outputs. Key techniques include zero-shot prompting,
        few-shot prompting with examples, chain-of-thought for reasoning tasks,
        and role prompting to set the AI persona. System prompts define the
        AI's behavior and constraints. Temperature controls output randomness.
        Good prompt engineers can significantly improve LLM output quality
        without any model training. This is a highly valued skill in 2024-2025.
        """
    },
    {
        "title": "Vector Databases",
        "content": """
        Vector databases store high-dimensional embeddings and enable fast
        similarity search. Popular options include ChromaDB (open source, local),
        Pinecone (cloud, managed), FAISS (Facebook, very fast), and Weaviate.
        ChromaDB is best for learning and prototyping — runs locally for free.
        Pinecone is best for production with large datasets.
        Vector search uses cosine similarity or dot product to find similar vectors.
        Vector databases are the backbone of RAG systems and semantic search.
        """
    }
]

print(f"Created {len(documents_text)} knowledge base documents:")
for doc in documents_text:
    word_count = len(doc["content"].split())
    print(f"  • {doc['title']:<30} ({word_count} words)")


# ── SECTION 3: Text Splitting ─────────────────────────────────
print("\n\nSECTION 3: Text Splitting")
print("-" * 40)
print("""
Documents are too long to fit in context window.
We split them into smaller chunks (200-500 words each).
Each chunk is stored separately in the vector database.
""")

if LANGCHAIN_AVAILABLE:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = 300,    # characters per chunk
        chunk_overlap = 50,     # overlap between chunks (maintains context)
        separators    = ["\n\n", "\n", ". ", " "]
    )

    # Convert to LangChain Document objects
    docs = []
    for item in documents_text:
        chunks = splitter.split_text(item["content"].strip())
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content = chunk,
                metadata     = {"title": item["title"], "chunk": i}
            ))

    print(f"  Original documents : {len(documents_text)}")
    print(f"  After splitting    : {len(docs)} chunks")
    print(f"\n  Sample chunks from 'RAG Systems' document:")
    rag_chunks = [d for d in docs if d.metadata["title"] == "RAG Systems"]
    for i, chunk in enumerate(rag_chunks):
        print(f"  Chunk {i+1}: {chunk.page_content[:100]}...")
else:
    print("  Install LangChain to see text splitting demo")


# ── SECTION 4: Create Vector Store ───────────────────────────
print("\n\nSECTION 4: Creating Vector Store with ChromaDB")
print("-" * 40)

if LANGCHAIN_AVAILABLE:
    print("  Loading embedding model (downloads ~90MB first time)...")
    print("  Using: sentence-transformers/all-MiniLM-L6-v2 (free, local)")

    try:
        # Free embeddings — no API key needed!
        embeddings = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs = {"device": "cpu"}
        )

        # Create ChromaDB vector store
        vectorstore = Chroma.from_documents(
            documents          = docs,
            embedding          = embeddings,
            persist_directory  = "./chroma_db"   # saves to disk
        )

        print(f"  ✓ Vector store created with {vectorstore._collection.count()} chunks!")
        print("  ✓ Saved to ./chroma_db folder")

        # Test similarity search
        print("\n  Testing similarity search:")
        queries = [
            "What salary can I expect as an AI engineer fresher?",
            "How does attention mechanism work?",
            "What is ChromaDB used for?"
        ]

        for query in queries:
            results = vectorstore.similarity_search(query, k=2)
            print(f"\n  Query: '{query}'")
            print(f"  Top result from: [{results[0].metadata['title']}]")
            print(f"  Content: {results[0].page_content[:120]}...")

    except Exception as e:
        print(f"  Note: {e}")
        print("  Run: pip install sentence-transformers chromadb")
else:
    print("  Install dependencies to create vector store")
    print("""
  Example similarity search output:

  Query: 'What salary can I expect as an AI engineer fresher?'
  Top result from: [AI Engineer Career]
  Content: Freshers with strong project portfolios can expect 8-12 LPA.
           Companies hiring AI engineers include startups, product companies...
  """)


# ── SECTION 5: RAG Chain ─────────────────────────────────────
print("\n\nSECTION 5: Full RAG Chain — Question Answering")
print("-" * 40)

if API_AVAILABLE and LANGCHAIN_AVAILABLE:
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        from langchain.prompts import ChatPromptTemplate

        llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-3.5-turbo", temperature=0)

        # RAG prompt template
        rag_prompt = ChatPromptTemplate.from_template("""
        Answer the question based ONLY on the following context.
        If the answer is not in the context, say "I don't have information about that."

        Context:
        {context}

        Question: {question}

        Answer:""")

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Full RAG chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        questions = [
            "What salary can a fresher AI engineer expect in India?",
            "What is the difference between BERT and GPT?",
            "How does ChromaDB work?",
            "What cities are best for AI jobs in India?",
            "What is the capital of France?"  # Not in our docs!
        ]

        print("  Running RAG Q&A on our knowledge base:\n")
        for q in questions:
            answer = rag_chain.invoke(q)
            print(f"  Q: {q}")
            print(f"  A: {answer[:200]}")
            print()

    except Exception as e:
        print(f"  Error: {e}")

else:
    print("""
  [Full RAG output — needs API key]

  Q: What salary can a fresher AI engineer expect?
  A: Freshers with strong project portfolios can expect 8-12 LPA.

  Q: What is the difference between BERT and GPT?
  A: BERT is an encoder-only transformer good for understanding tasks,
     while GPT is a decoder-only transformer good for text generation.

  Q: What is the capital of France?
  A: I don't have information about that.
     ← RAG only answers from YOUR documents, not general knowledge!
  """)


# ── SECTION 6: Manual RAG (No API Needed) ────────────────────
print("\n\nSECTION 6: Manual RAG — Understanding the Math")
print("-" * 40)
print("Let's build a TINY RAG system using just NumPy — no API needed!")

# Mini knowledge base
mini_kb = {
    "Python salary"   : "Python AI engineers earn 8-25 LPA in India. Freshers start at 8-12 LPA.",
    "LangChain"       : "LangChain is a framework for building LLM applications with chains and agents.",
    "RAG definition"  : "RAG stands for Retrieval Augmented Generation. It retrieves relevant documents before generating answers.",
    "ChromaDB"        : "ChromaDB is an open-source vector database for storing and searching embeddings locally.",
    "Transformers"    : "Transformers use attention mechanisms and are the basis of GPT, BERT, and Claude.",
}

def simple_embed(text):
    """Fake embedding using character frequencies — just for demo!"""
    text = text.lower()
    vec  = np.zeros(26)
    for char in text:
        if char.isalpha():
            vec[ord(char) - ord('a')] += 1
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

# Build index
kb_embeddings = {key: simple_embed(key + " " + val) for key, val in mini_kb.items()}

def manual_rag_search(query, top_k=2):
    """Find most relevant docs using cosine similarity"""
    query_emb = simple_embed(query)
    scores    = {}
    for key, emb in kb_embeddings.items():
        scores[key] = cosine_similarity(query_emb, emb)
    sorted_keys = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [(k, mini_kb[k], scores[k]) for k in sorted_keys]

test_queries = [
    "how much money does an AI engineer make",
    "what is retrieval augmented generation",
    "vector database for embeddings",
]

print("\n  Manual RAG search results:")
for query in test_queries:
    results = manual_rag_search(query)
    print(f"\n  Query: '{query}'")
    for doc_name, content, score in results:
        print(f"    [{doc_name}] (score: {score:.3f})")
        print(f"    {content[:80]}...")

print()
print("=" * 60)
print("Script 2 complete! RAG pipeline covered.")
print("Key concepts:")
print("  ✓ Embeddings — text as vectors")
print("  ✓ Cosine similarity — measuring text similarity")
print("  ✓ Text splitting — chunking documents")
print("  ✓ ChromaDB — local vector store")
print("  ✓ RAG chain — retrieval + generation")
print("  ✓ Manual RAG with pure NumPy")
print("=" * 60)
