# -*- coding: utf-8 -*-
# ============================================================
# DAY 7 -- SCRIPT 2: Docker -- Containerize Your AI App
# Topics: What is Docker, Dockerfile, images, containers,
#         dockerizing FastAPI app, docker-compose
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("DAY 7 -- SCRIPT 2: Docker -- Containerize Your AI App")
print("=" * 60)

print("""
WHAT IS DOCKER?
---------------
The classic problem: "It works on my laptop but not on the server!"

Docker solves this by packaging your app + all dependencies
into a container that runs IDENTICALLY everywhere.

Without Docker:
  Your laptop: Python 3.12, numpy 1.24, works fine
  Server:      Python 3.9,  numpy 1.20, CRASHES

With Docker:
  Container has Python 3.12 + numpy 1.24 built in
  Runs the same on your laptop, server, AWS, Google Cloud

Key concepts:
  Image     -> blueprint/recipe (like a class in Python)
  Container -> running instance of an image (like an object)
  Dockerfile-> instructions to build an image
  Registry  -> storage for images (Docker Hub, AWS ECR)

Real world use:
  Every major tech company uses Docker.
  "Dockerize your app" is a standard job requirement.
  Kubernetes (K8s) orchestrates Docker containers at scale.
""")


# -- SECTION 1: Write Dockerfile ------------------------------
print("SECTION 1: Writing a Dockerfile")
print("-" * 40)
print("""
A Dockerfile is a text file with step-by-step instructions
to build a Docker image. Think of it like a recipe.
""")

dockerfile_content = """# ============================================================
# Dockerfile for AI Placement Predictor API
# ============================================================

# Step 1: Start from an official Python image
# python:3.11-slim = Python 3.11 on minimal Linux (smaller size)
FROM python:3.11-slim

# Step 2: Set working directory inside container
WORKDIR /app

# Step 3: Copy requirements file first (for caching)
# Docker caches layers -- if requirements.txt didn't change,
# it won't reinstall packages (saves build time!)
COPY requirements.txt .

# Step 4: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy your application code
COPY . .

# Step 6: Expose the port your app runs on
EXPOSE 8000

# Step 7: Command to run when container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

with open("Dockerfile", "w", encoding="utf-8") as f:
    f.write(dockerfile_content)

print("  Dockerfile created!")
print("\n  Dockerfile explanation line by line:")
explanations = [
    ("FROM python:3.11-slim",      "Base image -- start with Python already installed"),
    ("WORKDIR /app",               "All commands run from /app folder inside container"),
    ("COPY requirements.txt .",    "Copy requirements first (Docker layer caching)"),
    ("RUN pip install ...",        "Install all Python packages"),
    ("COPY . .",                   "Copy all your code into the container"),
    ("EXPOSE 8000",                "Tell Docker this app uses port 8000"),
    ("CMD [uvicorn...]",           "Command that runs when container starts"),
]
for cmd, explanation in explanations:
    print(f"  {cmd:<35} -> {explanation}")


# -- SECTION 2: Requirements file ----------------------------
print("\n\nSECTION 2: requirements.txt for Docker")
print("-" * 40)

requirements = """fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
python-multipart>=0.0.6
"""

with open("requirements_api.txt", "w", encoding="utf-8") as f:
    f.write(requirements)

print("  requirements_api.txt created!")
print(f"\n  Contents:\n{requirements}")


# -- SECTION 3: Docker Compose --------------------------------
print("\nSECTION 3: Docker Compose -- Running Multiple Services")
print("-" * 40)
print("""
Docker Compose lets you define and run multi-container apps.

Real AI app might have:
  -> FastAPI backend (ML predictions)
  -> Streamlit frontend (UI)
  -> PostgreSQL database (store predictions)
  -> Redis cache (speed up responses)

docker-compose.yml connects all of these together.
""")

compose_content = """# ============================================================
# docker-compose.yml
# Run with: docker-compose up
# ============================================================

version: '3.8'

services:

  # FastAPI ML backend
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ENVIRONMENT=production
    volumes:
      - ./models:/app/models
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Streamlit frontend
  frontend:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    depends_on:
      - api
    environment:
      - API_URL=http://api:8000
    restart: always

  # Optional: Redis cache for fast responses
  # redis:
  #   image: redis:alpine
  #   ports:
  #     - "6379:6379"

"""

with open("docker-compose.yml", "w", encoding="utf-8") as f:
    f.write(compose_content)

print("  docker-compose.yml created!")


# -- SECTION 4: .dockerignore --------------------------------
print("\n\nSECTION 4: .dockerignore -- Keep Image Small")
print("-" * 40)
print("""
.dockerignore tells Docker what NOT to copy into the image.
Like .gitignore but for Docker.
Keeps your image small and fast to build.
""")

dockerignore = """# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.env

# Git
.git/
.gitignore

# IDE
.vscode/
.idea/

# Logs and temp files
*.log
*.tmp
chroma_db/

# Large data files (mount as volumes instead)
*.csv
*.parquet

# Test files
tests/
test_*.py
"""

with open(".dockerignore", "w", encoding="utf-8") as f:
    f.write(dockerignore)

print("  .dockerignore created!")


# -- SECTION 5: All Docker Commands You Need ------------------
print("\n\nSECTION 5: Essential Docker Commands")
print("-" * 40)

commands = [
    ("Build image",              "docker build -t my-ai-app ."),
    ("Run container",            "docker run -p 8000:8000 my-ai-app"),
    ("Run in background",        "docker run -d -p 8000:8000 my-ai-app"),
    ("List running containers",  "docker ps"),
    ("List all containers",      "docker ps -a"),
    ("Stop container",           "docker stop <container_id>"),
    ("Remove container",         "docker rm <container_id>"),
    ("List images",              "docker images"),
    ("Remove image",             "docker rmi my-ai-app"),
    ("View logs",                "docker logs <container_id>"),
    ("Enter container shell",    "docker exec -it <container_id> bash"),
    ("Start compose",            "docker-compose up"),
    ("Start compose background", "docker-compose up -d"),
    ("Stop compose",             "docker-compose down"),
    ("Rebuild compose",          "docker-compose up --build"),
    ("Push to Docker Hub",       "docker push yourusername/my-ai-app"),
]

print(f"\n  {'Action':<30} {'Command'}")
print(f"  {'─'*30} {'─'*40}")
for action, command in commands:
    print(f"  {action:<30} {command}")


# -- SECTION 6: Full Deployment Flow --------------------------
print("\n\nSECTION 6: Full Deployment Flow")
print("-" * 40)
print("""
How your AI app goes from laptop to production:

  Step 1: Build Docker image
  --------------------------
  docker build -t placement-predictor .

  Step 2: Test locally
  --------------------
  docker run -p 8000:8000 placement-predictor
  -> Open http://localhost:8000/docs to verify

  Step 3: Push to Docker Hub (free registry)
  -------------------------------------------
  docker login
  docker tag placement-predictor yourusername/placement-predictor
  docker push yourusername/placement-predictor

  Step 4: Deploy to cloud (multiple options)
  -------------------------------------------
  Option A: Render.com (FREE tier available!)
    -> Go to render.com
    -> New Web Service
    -> Connect GitHub repo
    -> Select Docker
    -> Deploy!

  Option B: Railway.app (FREE tier)
    -> Go to railway.app
    -> New Project -> Deploy from GitHub
    -> Auto-detects Dockerfile
    -> Deploy!

  Option C: Google Cloud Run (pay per request)
    gcloud run deploy --source .

  Option D: AWS App Runner
    -> Push to ECR
    -> Create App Runner service

FASTEST FREE OPTION FOR YOU RIGHT NOW:
  1. Push your code to GitHub (already doing this!)
  2. Go to render.com -> New Web Service
  3. Connect your GitHub repo
  4. Set start command: uvicorn main:app --host 0.0.0.0 --port 8000
  5. Click Deploy -> FREE live API in 5 minutes!
""")


# -- SECTION 7: Environment Variables (Security) -------------
print("\nSECTION 7: Environment Variables -- Keep Secrets Safe")
print("-" * 40)
print("""
NEVER put API keys directly in code or Dockerfile.
Use environment variables instead.
""")

env_example = """# .env.example (commit this to GitHub)
# .env (NEVER commit this -- add to .gitignore!)

# OpenAI
OPENAI_API_KEY=your-key-here

# App settings
ENVIRONMENT=development
DEBUG=true
MAX_BATCH_SIZE=100

# Database (if using one)
DATABASE_URL=postgresql://user:password@localhost/dbname
"""

with open(".env.example", "w", encoding="utf-8") as f:
    f.write(env_example)

print("  .env.example created!")
print("""
  How to use in Python:
  import os
  from dotenv import load_dotenv

  load_dotenv()  # reads .env file
  api_key = os.environ.get("OPENAI_API_KEY")

  How to pass to Docker:
  docker run -e OPENAI_API_KEY=sk-... my-ai-app

  Or use --env-file:
  docker run --env-file .env my-ai-app
""")

print("=" * 60)
print("Script 2 complete! Docker covered.")
print("Key concepts:")
print("  [OK] What Docker is and why it matters")
print("  [OK] Dockerfile written and explained")
print("  [OK] Docker Compose for multi-service apps")
print("  [OK] All essential Docker commands")
print("  [OK] Full deployment flow to production")
print("  [OK] Environment variables for security")
print("=" * 60)
