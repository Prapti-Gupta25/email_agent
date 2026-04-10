# 📧 Email Triage RL Environment

An intelligent Reinforcement Learning environment for AI-driven email management 

## 🚀 Live Demo
🤗 HuggingFace Space: https://huggingface.co/spaces/prapt25/email_agent

## 📌 Problem Statement
Email overload is a real problem. Professionals spend hours triaging emails — classifying spam, prioritizing important ones, and deciding actions. This project builds an RL environment where an AI agent learns to do this automatically.

## ⚙️ How It Works
The environment exposes emails to an AI agent, which takes one of three actions:

| Action | Description |
|--------|-------------|
| `classify` | Label email as spam / important / normal |
| `prioritize` | Order emails by importance |
| `triage` | Classify + decide action (reply / archive / escalate) |

The agent receives a reward based on correctness of its decisions, and learns to improve over time.

## 🧠 Tech Stack
- **Framework:** OpenEnv Core
- **Backend:** FastAPI + Uvicorn
- **AI Model:** Qwen/Qwen2.5-72B-Instruct (via HuggingFace Router)
- **Data Validation:** Pydantic v2
- **Deployment:** Docker + HuggingFace Spaces

## 📁 Project Structure
email_triage_env/
├── server/
│   ├── app.py                            # FastAPI server
│   └── email_triage_env_environment.py   # RL Environment logic
├── models.py                             # Action & Observation schemas
├── tasks/                                # Task definitions
server/
└── app.py                                # Entry point for validator
inference.py                              # Agent inference logic
openenv.yaml                              # Environment config
Dockerfile                                # Container setup
pyproject.toml
uv.lock

## 🎯 Tasks
- **Easy:** Email Classification
- **Medium:** Email Prioritization
- **Hard:** Full Email Triage
