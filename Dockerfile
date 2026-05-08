# ============================================================
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
