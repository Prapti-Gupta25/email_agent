import os
import json
import textwrap
from openai import OpenAI

# Importing your specific environment classes
from email_triage_env.server.email_triage_env_environment import EmailTriageEnvironment
from email_triage_env.models import EmailTriageAction

# Configuration from environment variables
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "EMPTY"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

MAX_STEPS = 5
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

def format_emails(observation):
    return "\n".join([
        f"ID: {e.id}, Subject: {e.subject}, Body: {e.body}, Sender: {e.sender}"
        for e in observation.emails
    ])

# Helper to clean LLM JSON response
def extract_json(text):
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except:
        return None

def get_classify_action(observation, remaining_ids):
    target_id = remaining_ids[0]
    prompt = f"Classify email ID {target_id}. Return JSON: {{\"type\": \"classify\", \"email_id\": {target_id}, \"label\": \"spam/important/normal\"}}\n\n{format_emails(observation)}"
    try:
        completion = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.1)
        data = extract_json(completion.choices[0].message.content)
        return EmailTriageAction(**data) if data else EmailTriageAction(type="classify", email_id=target_id, label="normal")
    except:
        return EmailTriageAction(type="classify", email_id=target_id, label="normal")

def get_prioritize_action(observation, remaining_ids):
    target_id = remaining_ids[0]
    ids = [e.id for e in observation.emails]
    prompt = f"Order IDs by priority. Return JSON: {{\"type\": \"prioritize\", \"email_id\": {target_id}, \"order\": {ids}}}\n\n{format_emails(observation)}"
    try:
        completion = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.1)
        data = extract_json(completion.choices[0].message.content)
        return EmailTriageAction(**data) if data else EmailTriageAction(type="prioritize", email_id=target_id, order=ids)
    except:
        return EmailTriageAction(type="prioritize", email_id=target_id, order=ids)

def get_triage_action(observation, remaining_ids):
    target_id = remaining_ids[0]
    prompt = f"Triage email {target_id}. Return JSON: {{\"type\": \"triage\", \"email_id\": {target_id}, \"label\": \"spam/important/normal\", \"action\": \"reply/archive\"}}\n\n{format_emails(observation)}"
    try:
        completion = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.1)
        data = extract_json(completion.choices[0].message.content)
        return EmailTriageAction(**data) if data else EmailTriageAction(type="triage", email_id=target_id, label="normal", action="archive")
    except:
        return EmailTriageAction(type="triage", email_id=target_id, label="normal", action="archive")

def run_task(task_id, task_name, get_action_fn):
    env = EmailTriageEnvironment()
    observation = env.reset()
    rewards = []
    done = False
    step = 0
    remaining_ids = [e.id for e in observation.emails]

    print(f"[START] task={task_id} env=email_triage_env model={MODEL_NAME}", flush=True)

    while not done and step < MAX_STEPS and remaining_ids:
        step += 1
        action = get_action_fn(observation, remaining_ids)
        observation = env.step(action)

        reward = float(observation.reward or 0.0)
        done = bool(observation.done)
        rewards.append(reward)

        if action.email_id in remaining_ids:
            remaining_ids.remove(action.email_id)
        if not remaining_ids: done = True

        print(f"[STEP] step={step} reward={reward:.2f} done={str(done).lower()}", flush=True)

    score = sum(rewards) / len(rewards) if rewards else 0.5
    score = max(0.01, min(0.99, score))
    success = "true" if score > 0.5 else "false"

    print(f"[END] score={score:.2f} success={success}", flush=True)
    return score

def run_inference(custom_email=None):
    results = {}
    results["easy"] = run_task("easy", "Email Classification", get_classify_action)
    results["Medium"] =run_task("medium", "Email Prioritization", get_prioritize_action)
    results["Hard"] =run_task("hard", "Full Email Triage", get_triage_action)

    return results

def process_single(email_text):
    prompt = f"""
    You are an AI email assistant
    Analyze the email and return JSON:
    {{
        "category": "spam/important/normal",
        "action": "reply/archive",
        "urgency": "low/medium/high",
        "risk": "safe/phishing/fraud",
        "tasks": ["List of actionable tasks from email"],
        "deadline": "date/time if mentioned or null",
        "reason": "short explanation",
        "reply": "professional reply if action is reply"
        
    }}

    Email:
    {email_text}
    """
    try:
        completion = client.chat.completions.create(
            model = MODEL_NAME,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2 
        )

        data = extract_json(completion.choices[0].message.content)

        return{
            "category": data.get("category", "normal"),
            "action": data.get("action", "archive"),
            "urgency": data.get("urgency", "low"),
            "risk": data.get("risk", "safe"),
            "tasks": data.get("tasks", []),
            "deadline": data.get("deadline", "none"),
            "reason": data.get("reason", ""),
            "reply": data.get("reply", "")
        }

    except Exception as e:
        return {"error:str(e)"}
    
def process_single(email_text):
    prompt = f"""
You are an intelligent email assistant.

Analyze the email STRICTLY and return JSON:

{{
    "category": "spam OR important OR normal",
    "action": "reply OR archive",
    "urgency": "low OR medium OR high",
    "risk": "safe OR phishing OR fraud",
    "tasks": ["clear actionable tasks"],
    "deadline": "exact time/date if present else null",
    "reason": "short explanation",
    "reply": "natural professional reply if needed"
}}

RULES:
- If meeting, deadline, or schedule → mark IMPORTANT + HIGH urgency
- If scam, prize, bank issue → mark SPAM + PHISHING
- ALWAYS extract at least 1 task if action required
- NEVER leave tasks empty if email contains action
- Reply should sound human

Email:
{email_text}
"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        data = extract_json(completion.choices[0].message.content)

        
        email_lower = email_text.lower()

        if not data:
            data = {}

        
        if "meeting" in email_lower:
            data["category"] = "important"
            data["urgency"] = "high"
            data["action"] = "reply"

       
        if not data.get("tasks"):
            tasks = []

            if "meeting" in email_lower:
                tasks.append("Attend meeting")
                tasks.append("Prepare for meeting")

            elif "deadline" in email_lower:
                tasks.append("Complete task before deadline")

            elif "verify" in email_lower:
                tasks.append("Check account security")

            elif "order" in email_lower:
                tasks.append("Track order status")

            else:
                tasks.append("No action required")

            data["tasks"] = tasks

        if not data.get("reply"):
            if "meeting" in email_lower:
                data["reply"] = "Got it, I will be there at the scheduled time."
            elif data.get("category") == "spam":
                data["reply"] = "This looks suspicious. I will ignore it."
            else:
                data["reply"] = "Noted. Thanks!"

        return {
            "category": data.get("category", "normal"),
            "action": data.get("action", "archive"),
            "urgency": data.get("urgency", "low"),
            "risk": data.get("risk", "safe"),
            "tasks": data.get("tasks", []),
            "deadline": data.get("deadline", "none"),
            "reason": data.get("reason", ""),
            "reply": data.get("reply", "")
        }

    except Exception as e:
        return {"error": str(e)}
