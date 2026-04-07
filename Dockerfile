FROM python:3.11-slim

WORKDIR /app

# Dependencies install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project files copy
COPY . .

# Environment variables (HF Spaces mein set karna)
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
ENV PORT=7860

EXPOSE 7860

# FastAPI server start
CMD ["uvicorn", "email_triage_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]