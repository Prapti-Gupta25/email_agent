FROM python:3.11-slim

WORKDIR /app

# 1. Install system dependencies (git is often required for OpenEnv)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 2. Copy configuration files first for better layer caching
COPY pyproject.toml .
# If you still have a requirements.txt, keep this line; otherwise, delete it
COPY requirements.txt . 

# 3. Install dependencies and the project in editable mode
# This ensures all dependencies in pyproject.toml are met
RUN pip install --no-cache-dir -r requirements.txt && pip install -e .

# 4. Copy the rest of the project files
COPY . .

# 5. Required Environment Variables
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
# Note: Scaler infra usually expects 8000, but HF Spaces uses 7860. 
# OpenEnv server handles the mapping if you pass the --port flag.
ENV PORT=7860

EXPOSE 7860

# 6. Use the standardized OpenEnv server command
# This ensures your environment is correctly wrapped for the validator
CMD ["python", "-m", "openenv.server", "--host", "0.0.0.0", "--port", "7860"]
