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

    # CLEAN START LOG
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

        # CLEAN STEP LOG - DO NOT ADD EXTRA FIELDS HERE
        print(f"[STEP] step={step} reward={reward:.2f} done={str(done).lower()}", flush=True)

    score = sum(rewards) / len(rewards) if rewards else 0.0
    success = "true" if score > 0.5 else "false"

    # CLEAN END LOG
    print(f"[END] score={score:.2f} success={success}", flush=True)
    return score

def run():
    # The task IDs must match your openenv.yaml
    run_task("easy", "Email Classification", get_classify_action)
    run_task("medium", "Email Prioritization", get_prioritize_action)
    run_task("hard", "Full Email Triage", get_triage_action)

if __name__ == "__main__":
    run()
